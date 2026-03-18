# task-016: Migrate Google embedder from google-generativeai to google-genai

## Summary
Migrate `GoogleTextEmbedder` from the deprecated `google-generativeai` SDK to the new
`google-genai` SDK (`from google import genai`). This fixes the deprecation flag from
task-006 and prepares the environment for task-017 (LLMScorer with Gemini adapter).
After migration, remove `google-generativeai` from requirements.txt.

## Requirements
1. `GoogleTextEmbedder` uses `from google import genai` (new SDK) instead of
   `import google.generativeai as genai` (old SDK).
2. The `embed()` and `embed_query()` methods produce the same output format:
   `np.ndarray` of shape `(len(texts), 768)` with dtype `float32`.
3. The `__init__` constructor accepts the same parameters: `model`, `task_type`, `api_key`.
4. The `name` and `dimension` properties work identically.
5. Rate-limit retry logic (single retry with 1-second sleep) is preserved.
6. All existing tests pass with mocked API calls (no network dependency).
7. `google-generativeai` is removed from `requirements.txt` after migration.

## Files to Modify
- `src/embedders/google_text.py` — **modify**. Replace `import google.generativeai as genai`
  with `from google import genai`. Update `__init__` to use `genai.Client(api_key=...)` instead
  of `genai.configure(api_key=...)`. Update `_embed_single` to use
  `client.models.embed_content(model=..., contents=..., config=...)` instead of
  `genai.embed_content(model=..., content=..., task_type=...)`.
- `requirements.txt` — **modify**. Remove the `google-generativeai==0.8.6` line and its
  exclusive dependencies: `google-ai-generativelanguage==0.6.15`,
  `google-api-python-client==2.192.0`, `google-auth-httplib2==0.3.0`,
  `googleapis-common-protos==1.73.0`, `grpcio==1.78.0`, `grpcio-status==1.71.2`,
  `proto-plus==1.27.1`, `protobuf==5.29.6`, `uritemplate==4.2.0`,
  `httplib2==0.31.2`. Only remove packages that are NOT also required by other
  installed packages — check with `pip show <pkg>` to see if other packages depend on them.
  When in doubt, leave the dependency in.

## Files to Read for Context
- `src/embedders/google_text.py` — the file being migrated
- New SDK docs: the API is `client.models.embed_content(model=..., contents=..., config=types.EmbedContentConfig(task_type=...))`

## New Dependencies
None — `google-genai==1.67.0` is already installed.

## API Migration Map

| Old SDK (`google-generativeai`) | New SDK (`google-genai`) |
|---|---|
| `import google.generativeai as genai` | `from google import genai` |
| `genai.configure(api_key=key)` | `client = genai.Client(api_key=key)` |
| `genai.embed_content(model=m, content=t, task_type=tt)` | `client.models.embed_content(model=m, contents=t, config=types.EmbedContentConfig(task_type=tt))` |
| `result["embedding"]` | `result.embeddings[0].values` |

Note: The new SDK uses a `Client` instance pattern instead of module-level configuration.
Import `from google.genai import types` for the config object.

The model name format may differ: old SDK used `"models/text-embedding-005"`, new SDK
may accept `"text-embedding-005"` or `"models/text-embedding-005"`. Check both and use
whichever works. The `gemini-embedding-001` model is the current recommended embedding
model — but keep `text-embedding-005` as default for now (it's what existing experiments
use).

## Edge Cases
- **No API key**: raise `ValueError` with the same message as before.
- **Empty text list**: return `np.empty((0, 768), dtype=np.float32)` — same as current.
- **Rate limit (429)**: single retry with 1-second sleep — same as current.
- **API error on retry**: raise the exception — same as current.

## Decisions Made
- **Keep text-embedding-005 as default**: changing the default model would break
  reproducibility of any cached embeddings. **Why:** backwards compatibility for experiment
  results.
- **Store client as instance variable**: `self._client = genai.Client(api_key=...)` instead
  of module-level `genai.configure()`. **Why:** the new SDK uses instance pattern; module-level
  config is not supported.
- **Remove old SDK from requirements.txt**: no reason to keep both. **Why:** they can
  conflict in the `google` namespace. Be careful only to remove deps exclusive to the old
  SDK.

## What NOT to Touch
- `src/embedders/__init__.py` — no changes needed
- `src/protocols.py` — Embedder protocol unchanged
- Any other embedder files
- `src/scorers/` — that's task-017

## Testing Approach
- Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-016-embedder-sdk-migration/tests/test_embedder_migration.py`
- Mock `genai.Client` and its `models.embed_content` method
- Test same scenarios as existing tests: basic embed, embed_query, empty list, rate limit retry
- Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-016-embedder-sdk-migration/tests/`
