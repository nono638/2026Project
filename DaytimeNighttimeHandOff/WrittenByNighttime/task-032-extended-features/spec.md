# task-032: Extended Feature Columns — Context Window, Readability, Embedding Spread, Query-Document Similarity

## Summary

Add 6 new columns to the experiment output: 2 pipeline metadata columns (LLM context window
size, context utilization ratio) and 4 document/query features (readability score, embedding
cluster spread, query-document embedding similarity, query-document lexical overlap). These
give the XGBoost meta-learner richer signal for predicting optimal RAG configuration — both
pre-query (document features) and post-query (query-document features).

**Why these 6:**
- **Context window**: Context rot / "lost in the middle" effect depends on how much of the
  context window is used. Tracking `llm_context_window` and `context_utilization_ratio`
  lets analysis correlate answer quality with context fullness.
- **Readability**: Document complexity affects RAG performance — harder documents may need
  different chunking or bigger models. Flesch-Kincaid is a well-validated, fast metric.
- **Embedding spread**: We already count topic clusters (doc_topic_count) but don't measure
  how tight/loose they are. Loose clusters = harder to retrieve from cleanly.
- **Query-doc similarity**: Cosine similarity between query embedding and mean document
  embedding — a cheap signal for whether the query is topically central to the document.
- **Query-doc lexical overlap**: Jaccard word overlap between query and document. Different
  signal from embedding similarity — captures exact keyword matching. Both are useful for
  the meta-learner to predict retrieval difficulty.

Research context: reference/research.md lists readability metrics (candidate #4), embedding
cluster variance (#3), and vocabulary diversity (#5) as document characterization features
identified from literature. Context window tracking was identified during daytime review of
pipeline metadata coverage.

## Requirements

1. New metadata function `get_llm_context_window(model, provider, host)` in `src/metadata.py`
   that returns the LLM's context window size (in tokens) or None if unknown.
2. For `provider="ollama"`, query the Ollama API via `ollama.Client(host).show(model)` to
   get the context window from model parameters. Cache results per model name.
3. For other providers, return None (no standard API for this).
4. New metadata column `llm_context_window` (int or None) in experiment output rows.
5. New metadata column `context_utilization_ratio` (float or None) = approximate tokens in
   context / context window size. Use `context_char_length / 4` as a rough char-to-token
   conversion. When `llm_context_window` is None, ratio is None.
6. New feature `doc_readability_score` (float) = Flesch-Kincaid grade level via `textstat`.
   Higher = harder to read. Typical range 0-18.
7. New feature `doc_embedding_spread` (float) = mean intra-cluster distance after KMeans
   clustering (same clustering already done in `_embedding_features()`). Measures how
   dispersed chunk embeddings are within their topic clusters.
8. New feature `query_doc_similarity` (float) = cosine similarity between query embedding
   and mean document embedding. Range [-1, 1], typically 0.1-0.8.
9. New feature `query_doc_lexical_overlap` (float) = Jaccard similarity between query word
   set and document word set (lowercased, punctuation stripped). Range [0, 1].
10. All existing tests continue to pass with these additions.
11. Update `required_cols` in `tests/test_e2e_smoke.py` to include all 6 new columns.

## Files to Modify

### `src/metadata.py` — Add context window lookup

Add after `build_dataset_metadata()`:

```python
# Module-level cache for context window lookups — avoids repeated API calls
# for the same model across different rows in the experiment loop.
_context_window_cache: dict[str, int | None] = {}


def get_llm_context_window(
    model: str, provider: str | None = None, host: str | None = None,
) -> int | None:
    """Query the LLM's context window size in tokens.

    For Ollama models, queries the Ollama API via client.show(). For other
    providers, returns None (no standard API for this).

    Results are cached per model name to avoid repeated API calls within
    a single experiment run.

    Args:
        model: Model name (e.g., "qwen3:4b").
        provider: LLM provider (e.g., "ollama"). If not "ollama", returns None.
        host: Ollama host URL, or None for localhost default.

    Returns:
        Context window size in tokens, or None if unknown.
    """
    if model in _context_window_cache:
        return _context_window_cache[model]

    ctx = None
    if provider == "ollama":
        ctx = _query_ollama_context_window(model, host)

    _context_window_cache[model] = ctx
    return ctx


def _query_ollama_context_window(model: str, host: str | None) -> int | None:
    """Query Ollama API for a model's context window size.

    Uses ollama.Client().show() which returns model metadata including
    parameters. The context window is in the 'num_ctx' model parameter.

    Args:
        model: Ollama model name.
        host: Ollama server URL, or None for default.

    Returns:
        Context window size in tokens, or None if unavailable.
    """
    try:
        from ollama import Client
        client = Client(host=host) if host else Client()
        info = client.show(model)
        # Ollama returns model info with parameter details.
        # The context window is typically in model_info or parameters.
        # Try multiple locations since Ollama API structure varies by version.
        if hasattr(info, 'model_info') and info.model_info:
            # Look for context_length or num_ctx in model_info dict
            for key in info.model_info:
                if 'context_length' in key.lower():
                    return int(info.model_info[key])
        # Fallback: check modelfile parameters
        if hasattr(info, 'parameters') and info.parameters:
            for line in info.parameters.split('\n'):
                if 'num_ctx' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return int(parts[-1])
        return None
    except Exception:
        # Ollama not running, model not found, network error, etc.
        return None


def build_llm_context_metadata(
    model: str,
    provider: str | None = None,
    host: str | None = None,
    context_char_length: int = 0,
) -> dict:
    """Build LLM context window metadata.

    Args:
        model: Model name.
        provider: LLM provider string.
        host: LLM host URL.
        context_char_length: Total character length of retrieved context.

    Returns:
        Dict with keys: llm_context_window, context_utilization_ratio.
    """
    ctx_window = get_llm_context_window(model, provider, host)

    if ctx_window is not None and ctx_window > 0:
        # Rough char-to-token conversion: ~4 chars per token for English.
        approx_tokens = context_char_length / 4
        ratio = approx_tokens / ctx_window
    else:
        ratio = None

    return {
        "llm_context_window": ctx_window,
        "context_utilization_ratio": round(ratio, 4) if ratio is not None else None,
    }
```

### `src/features.py` — Add 4 new features

**1. Add readability import and function** (after `_vocab_entropy`):

```python
def _readability_score(text: str) -> float:
    """Compute Flesch-Kincaid grade level for document text.

    Uses textstat library for accurate syllable counting and FK computation.
    Higher scores = harder to read. Typical range 0-18 where:
    - 5 = 5th grade level (very easy)
    - 12 = 12th grade (high school senior)
    - 16+ = college/graduate level

    Args:
        text: Full document text.

    Returns:
        Flesch-Kincaid grade level. Returns 0.0 for empty text.
    """
    if not text or not text.strip():
        return 0.0
    import textstat
    return textstat.flesch_kincaid_grade(text)
```

**2. Expand `_embedding_features()` to also return embedding spread.**

Change the return type from `tuple[float, float]` to `tuple[float, float, float]`:
- Still returns `(topic_count, semantic_coherence, embedding_spread)`
- Compute `embedding_spread` after KMeans: for each point, get distance to its assigned
  centroid, then average across all points. Use Euclidean distance (not cosine) since
  sklearn KMeans uses Euclidean internally.

```python
# After KMeans fit in _estimate_topic_count, compute intra-cluster spread
```

Actually, since `_estimate_topic_count` only returns an int, the spread computation should
happen in `_embedding_features()` directly. After getting `topic_count`, re-run a single
KMeans with `k=topic_count` (or reuse the clustering) and compute:

```python
# --- Embedding spread (intra-cluster variance) ---
from sklearn.cluster import KMeans as _KMeans
if topic_count >= 2 and n_chunks >= 3:
    km = _KMeans(n_clusters=topic_count, n_init="auto", random_state=42, max_iter=100)
    km.fit(embeddings)
    # Mean distance from each point to its assigned centroid
    distances = np.linalg.norm(embeddings - km.cluster_centers_[km.labels_], axis=1)
    embedding_spread = float(np.mean(distances))
else:
    # Single cluster or too few chunks — spread is just overall variance
    mean_emb = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - mean_emb, axis=1)
    embedding_spread = float(np.mean(distances))
```

**Important:** This duplicates KMeans computation. To avoid running KMeans twice, refactor
`_estimate_topic_count()` to return `(topic_count, km_model_or_None)` so
`_embedding_features()` can reuse the fitted model. Alternatively, fold the spread
computation into `_estimate_topic_count()` and return a tuple. The simplest refactor:

Change `_estimate_topic_count(embeddings, max_k=10) -> int` to
`_estimate_topic_count(embeddings, max_k=10) -> tuple[int, float]` where the second
value is the embedding spread. This avoids re-fitting.

Inside `_estimate_topic_count`, after finding `best_k`:
- If `best_k >= 2`: re-fit with `best_k` (or reuse the last fit if `best_k == upper`),
  compute mean distance to centroids.
- If `best_k == 1`: compute mean distance to global centroid.

**3. Add query-document similarity function:**

```python
def _query_doc_similarity(query: str, retriever: Retriever) -> float:
    """Cosine similarity between query embedding and mean document embedding.

    Embeds the query, computes mean of all chunk embeddings from the FAISS
    index, and returns their cosine similarity. Since chunk embeddings are
    L2-normalized, the mean needs re-normalization before dot product.

    Args:
        query: Query text.
        retriever: Retriever with populated FAISS index and embedder.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if no chunks.
    """
    n_chunks = retriever._index.ntotal
    if n_chunks == 0:
        return 0.0

    # Embed query
    query_emb = retriever._embedder.embed([query])
    faiss.normalize_L2(query_emb)
    query_vec = query_emb[0]

    # Mean of all chunk embeddings (reconstructed from FAISS index)
    embeddings = np.array(
        [retriever._index.reconstruct(i) for i in range(n_chunks)],
        dtype=np.float32,
    )
    mean_doc_emb = embeddings.mean(axis=0)
    # Re-normalize the mean vector for cosine similarity
    norm = np.linalg.norm(mean_doc_emb)
    if norm > 0:
        mean_doc_emb /= norm

    return float(np.dot(query_vec, mean_doc_emb))
```

**Note:** This requires `import faiss` at the top of features.py (currently not imported).
Add it next to the existing numpy import.

**4. Add query-document lexical overlap function:**

```python
def _query_doc_lexical_overlap(query: str, document: str) -> float:
    """Jaccard similarity between query and document word sets.

    Lowercased, punctuation stripped. Measures raw keyword overlap
    independent of semantics — complementary to embedding similarity.

    Args:
        query: Query text.
        document: Full document text.

    Returns:
        Jaccard similarity in [0, 1]. Returns 0.0 if both are empty.
    """
    q_words = set(query.lower().translate(_PUNCT_TABLE).split())
    d_words = set(document.lower().translate(_PUNCT_TABLE).split())
    if not q_words and not d_words:
        return 0.0
    intersection = q_words & d_words
    union = q_words | d_words
    return len(intersection) / len(union) if union else 0.0
```

**Note:** `_PUNCT_TABLE` doesn't exist in features.py — it's defined in retriever.py. Either
import it from retriever.py (`from src.retriever import _PUNCT_TABLE`) or define a local
copy. **Decision:** Define a local copy to avoid coupling features.py to retriever.py
internals. Add near the top of features.py:

```python
import string
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
```

**5. Wire all 4 new features into `extract_features()`:**

Update the function to call the new helpers and include results in the return dict:

```python
def extract_features(query, document, retriever, retrieved=None):
    ...
    # Existing features
    ner_density, ner_repetition = _ner_features(document)
    topic_count, semantic_coherence, embedding_spread = _embedding_features(retriever)
    # ^^^ Changed: now returns 3 values instead of 2
    doc_length = len(document.split())
    topic_density = topic_count / (doc_length / 1000) if doc_length > 0 else 0.0

    # New features
    readability = _readability_score(document)
    q_doc_sim = _query_doc_similarity(query, retriever)
    q_doc_overlap = _query_doc_lexical_overlap(query, document)

    return {
        # ... existing features unchanged ...
        "doc_ner_density": ner_density,
        "doc_ner_repetition": ner_repetition,
        "doc_topic_count": topic_count,
        "doc_topic_density": topic_density,
        "doc_semantic_coherence": semantic_coherence,
        # New document feature
        "doc_readability_score": readability,
        "doc_embedding_spread": embedding_spread,
        # New query-document features
        "query_doc_similarity": q_doc_sim,
        "query_doc_lexical_overlap": q_doc_overlap,
        # Retrieval-level (unchanged)
        "mean_retrieval_score": ...,
        "var_retrieval_score": ...,
    }
```

### `src/experiment.py` — Add context window metadata to row dict

In the run loop, after computing `build_context_metadata(retrieved)`, add the context
window metadata. The context_char_length is needed for the utilization ratio, so compute
it first:

```python
context_meta = build_context_metadata(retrieved)
# ... later in the row dict ...
**context_meta,
**build_llm_context_metadata(
    model,
    self._llm_provider,
    self._llm_host,
    context_meta["context_char_length"],
),
```

Import `build_llm_context_metadata` from `src.metadata` in the imports section.

**Placement in the row dict:** Put the two new columns right after `context_char_length`.

**Cache optimization:** `get_llm_context_window` already caches per model name, so calling
it inside the inner loop (per model) is fine — the API is only hit once per model.

### `tests/test_e2e_smoke.py` — Add new columns to required_cols

Add these to the `required_cols` list in `test_full_experiment_run`:

```python
# Extended features (task-032)
"doc_readability_score", "doc_embedding_spread",
"query_doc_similarity", "query_doc_lexical_overlap",
"llm_context_window", "context_utilization_ratio",
```

Note: `llm_context_window` and `context_utilization_ratio` may be None in tests (since
MockStrategy doesn't go through Ollama). That's fine — they just need to exist as columns.

### `tests/test_metadata.py` — Add tests for context window functions

Add `TestBuildLlmContextMetadata` class with tests:
- `build_llm_context_metadata()` with known context_char_length and mocked context window
- `build_llm_context_metadata()` when context window is None → ratio is None
- `get_llm_context_window()` caching behavior
- `_query_ollama_context_window()` when Ollama is not running → returns None

### `tests/test_integration.py` — Add feature checks

In `TestCrossComponentWiring.test_feature_extraction_with_retriever`, add assertions for
the 4 new feature keys:

```python
assert "doc_readability_score" in features
assert "doc_embedding_spread" in features
assert "query_doc_similarity" in features
assert "query_doc_lexical_overlap" in features
```

## New Dependencies

- `textstat==0.7.13` — Flesch-Kincaid readability scoring. Pure Python, depends on
  `pyphen==0.17.2` (hyphenation) and `nltk` (already installed, upgraded 3.9.2→3.9.3).
  **Already installed in venv and pinned in requirements.txt.**

## Edge Cases

- **Ollama not running** when querying context window → return None, don't crash.
  `_query_ollama_context_window()` wraps everything in try/except.
- **Model not found in Ollama** → same, return None.
- **context_utilization_ratio when context_window is None** → None (not 0 or infinity).
- **Empty document text for readability** → return 0.0.
- **Very short document (< 1 sentence)** → FK formula may give negative values. That's
  acceptable — it's a valid signal that the text is trivially simple.
- **Single chunk (1 embedding)** → embedding_spread = 0.0 (no variance possible).
  `_embedding_features` already handles n_chunks <= 1 → now returns `(1, 1.0, 0.0)`.
- **Empty chunks for query_doc_similarity** → return 0.0.
- **Query with no words in common with document** → lexical_overlap = 0.0.
- **Non-Ollama provider** → context_window is None, ratio is None. Feature columns still
  present (just None values).

## Decisions Made

- **textstat over manual FK implementation**: **Why:** textstat uses CMU Pronouncing
  Dictionary + fallback heuristics for ~95% syllable accuracy vs ~85% for simple vowel-
  group counting. Also gives us access to 10+ other readability indices if needed later.
  MIT license, pure Python, ~177KB.
- **Jaccard for lexical overlap, not TF-IDF cosine**: **Why:** Simpler, interpretable,
  no vocabulary fitting needed. The meta-learner can learn whatever threshold matters.
  TF-IDF would add complexity for marginal benefit when we already have embedding similarity.
- **Chars/4 for token approximation**: **Why:** Standard English approximation. Not
  perfect for all languages or tokenizers, but close enough for a utilization ratio that's
  used as a relative signal, not an absolute measure.
- **Euclidean distance for embedding spread (not cosine distance)**: **Why:** sklearn
  KMeans uses Euclidean distance internally. Using the same metric for spread measurement
  is consistent with how the clusters were formed.
- **Refactor `_estimate_topic_count` to return spread alongside count**: **Why:** Avoids
  running KMeans twice on the same embeddings. The spread is a natural byproduct of the
  clustering that's already happening.
- **Re-embed query for similarity rather than caching from retrieve()**: **Why:** The
  retriever doesn't expose the query embedding. Adding a cache/accessor to Retriever would
  be a larger change. One extra embed call per (query, embedder) is negligible (<1ms for
  HuggingFace, <50ms for Ollama).
- **Depends on task-031**: **Why:** task-031 modifies experiment.py's row dict (adds
  reranker columns, renames top_k). Building on top of that avoids merge conflicts.

## What NOT to Touch

- `src/retriever.py` — Don't add query embedding caching or context window lookups here.
  Retriever is retrieval-only.
- `src/protocols.py` — Don't add context_window to the LLM protocol. It's metadata, not
  a generation interface concern.
- `scripts/run_experiment_0.py` — Experiment 0 doesn't use the main Experiment runner for
  features. Its CSV has its own column set. Don't add these features there.
- `src/llms/*.py` — Don't modify LLM implementations. Context window lookup is in metadata.py.
- Existing feature computation — Don't change `_ner_features`, `_count_entities`,
  `_vocab_entropy`, or `_consecutive_cosine_mean`. Only modify `_embedding_features` and
  `_estimate_topic_count` (to add spread) and `extract_features` (to add new features).

## Testing Approach

### `tests/test_extended_features.py` — New unit tests (pre-written)

Tests for each new feature:
- Readability: returns float, non-negative for real text, 0.0 for empty text
- Embedding spread: non-negative, 0.0 for single chunk, > 0 for multi-topic documents
- Query-doc similarity: range check [-1, 1], higher for relevant query
- Query-doc lexical overlap: range [0, 1], 1.0 for identical text, 0.0 for no overlap

### `tests/test_metadata.py` — Context window metadata tests

- `build_llm_context_metadata()` with known values
- `build_llm_context_metadata()` with None context window
- `get_llm_context_window()` with non-Ollama provider returns None
- `_query_ollama_context_window()` when Ollama unavailable returns None

### Existing test updates

- `tests/test_e2e_smoke.py` — Add 6 new columns to required_cols
- `tests/test_integration.py` — Add feature key assertions

### Run commands

```bash
pytest tests/test_extended_features.py tests/test_metadata.py tests/test_integration.py tests/test_e2e_smoke.py -v
pytest tests/ -v  # Full suite
```
