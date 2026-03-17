# Spec: task-006 — Google Text Embedder

## What

Add one new embedder implementation that calls Google's text-embedding-005 via the
`google-generativeai` SDK (free tier via Google AI Studio).

**`GoogleTextEmbedder`** — 768 dimensions, up to 2048 tokens. Primary cloud-hosted
embedding axis for comparison against local models (Ollama, HuggingFace).

Implements the existing `Embedder` protocol from `src/protocols.py`. No protocol
changes needed.

The multimodal embedder (`multimodalembedding@001`) has been moved to incubating — it
requires a paid GCP Vertex AI account. See architecture-decisions.md.

## Why

The experiment framework compares RAG configurations across 4 axes (chunker, embedder,
strategy, model). Currently all embedders are local (Ollama, HuggingFace). Adding Google's
cloud embedding:
- Provides a cloud-hosted comparison point against local models
- Tests whether Google's text-embedding-005 outperforms mxbai-embed-large or MiniLM on
  our retrieval tasks
- google-generativeai has a free tier, so no cost barrier for experimentation

Docs: https://ai.google.dev/gemini-api/docs/embeddings

## Files to Create

### `src/embedders/google_text.py`

```python
"""Embedding via Google's text-embedding-005 model.

Uses the google-generativeai SDK (free tier available via Google AI Studio).
768 dimensions, up to 2048 tokens context.

Docs: https://ai.google.dev/gemini-api/docs/embeddings
"""
```

Class: `GoogleTextEmbedder`

Constructor:
- `model: str = "models/text-embedding-005"` — the model ID
- `task_type: str = "retrieval_document"` — embedding task type. Use `"retrieval_document"`
  when embedding chunks for indexing, `"retrieval_query"` when embedding queries. See note
  below on how to handle this.
- `api_key: str | None = None` — if None, reads from `GOOGLE_API_KEY` environment variable.
  Use `os.environ.get("GOOGLE_API_KEY")`. If neither is set, raise `ValueError` with a
  clear message: "Set GOOGLE_API_KEY environment variable or pass api_key to GoogleTextEmbedder".

**Task type handling:** The Embedder protocol has a single `embed()` method used for both
documents and queries. The Google API performs better when you specify the task type. Solution:
- Default `task_type` to `"retrieval_document"` (used when building the index)
- Add an `embed_query(texts)` convenience method that calls the API with
  `task_type="retrieval_query"`. This is NOT part of the Embedder protocol — it's an
  optional enhancement. The Retriever doesn't use it today, but it's available for future
  optimization. Do NOT modify the Retriever or protocol to use it.

Properties:
- `name` → `f"google:{model_name}"` where `model_name` is the part after `models/`
  (e.g., `"google:text-embedding-005"`)
- `dimension` → `768` (hardcoded; this model always returns 768-dim vectors)

`embed(self, texts: list[str]) -> np.ndarray`:
- Call `genai.embed_content()` for each text (or batch if the API supports it).
  Check the SDK docs — `embed_content` may accept a list of strings directly via the
  `content` parameter, or you may need to loop. Return shape `(len(texts), 768)`.
- Cast to `np.float32`.
- Handle rate limiting: if the API returns a rate limit error, do a simple `time.sleep(1)`
  and retry once. If it fails again, raise.

**SDK setup:**
```python
import google.generativeai as genai

# In __init__:
genai.configure(api_key=api_key)
```

### `src/embedders/__init__.py`

Update to export the new class:
```python
from src.embedders.ollama import OllamaEmbedder
from src.embedders.huggingface import HuggingFaceEmbedder
from src.embedders.google_text import GoogleTextEmbedder

__all__ = [
    "OllamaEmbedder",
    "HuggingFaceEmbedder",
    "GoogleTextEmbedder",
]
```

## Files to Modify

None beyond `src/embedders/__init__.py` (listed above).

## Files NOT to Touch

- `src/protocols.py` — no protocol changes in this task
- `src/retriever.py` — no changes needed, works with any Embedder
- `src/experiment.py` — no changes needed, accepts any Embedder list
- Any existing embedder files — leave them as-is

## Dependencies to Install

1. `google-generativeai` — for GoogleTextEmbedder

Install into the venv. Update `requirements.txt` with pinned versions.

**Use the install-package skill**: read `.claude/skills/install-package/SKILL.md`.

## Edge Cases

- **No API key set:** Raise `ValueError` with a clear message, not a cryptic SDK error.
- **Empty text list:** Return `np.empty((0, 768), dtype=np.float32)`.
- **Rate limiting:** Single retry with 1-second sleep. Don't add exponential backoff.
- **Network errors:** Let them propagate. The experiment runner already handles embedder
  failures gracefully.

## Tests

Create `tests/test_google_embedders.py`.

All tests must use **mocks** — do NOT call real Google APIs in tests. Mock at the SDK level.

### GoogleTextEmbedder tests:

1. **test_name_format** — `GoogleTextEmbedder().name == "google:text-embedding-005"`
2. **test_dimension** — `GoogleTextEmbedder().dimension == 768`
3. **test_embed_returns_correct_shape** — mock `genai.embed_content` to return a known
   vector. Call `embed(["hello", "world"])`. Assert shape is `(2, 768)` and dtype is
   `np.float32`.
4. **test_embed_empty_list** — `embed([])` returns shape `(0, 768)`.
5. **test_missing_api_key** — with no env var and no api_key param, constructor raises
   `ValueError`.
6. **test_embed_query_uses_retrieval_query_task_type** — mock `genai.embed_content`, call
   `embed_query(["test"])`, assert the mock was called with `task_type="retrieval_query"`.
7. **test_protocol_compliance** — `isinstance(GoogleTextEmbedder(...), Embedder)` is True.

### Mocking guidance:

- Mock `google.generativeai.embed_content` to return `{"embedding": [0.1] * 768}`
- Mock `google.generativeai.configure` to no-op
- Set `GOOGLE_API_KEY` env var in test fixtures via `monkeypatch`

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Research URLs included where research was done
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
