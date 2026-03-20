# task-031: Reranker Pipeline Stage

## Summary

Add a Reranker Protocol and two cross-encoder implementations (MiniLM lightweight, BGE
deep) as an optional pipeline stage between retrieval and generation. The reranker
re-scores retrieved chunks using a cross-encoder model for more precise relevance ranking,
then truncates to a final top_k. This is a new experimental axis: reranker model, input
candidates, and output top_k are all variables that analysis can slice by.

**Why:** Embedding similarity alone provides coarse ranking. Cross-encoders jointly encode
query-passage pairs for deeper semantic matching. Literature shows 10-48% retrieval
precision improvement. But whether this helps for small corpora (single Wikipedia articles
with 8-15 chunks) is an empirical question — this feature enables that experiment.

Research sources:
- Cross-encoder reranking improves RAG accuracy: https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study
- ms-marco-MiniLM-L-6-v2 model card: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
- bge-reranker-v2-m3 model card: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Reranker benchmark comparison: https://research.aimultiple.com/rerankers/

## Requirements

1. New `Reranker` Protocol in `src/protocols.py` — `name` property + `rerank()` method
2. `MiniLMReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, CPU-fast)
3. `BGEReranker` using `BAAI/bge-reranker-v2-m3` (278M params, deeper)
4. Both return dicts with `text`, `score` (original retrieval score), `rerank_score`, `index`
5. `Experiment` constructor accepts optional `reranker` and `reranker_top_k` params
6. When reranker is set, the experiment retrieves `retrieval_top_k` candidates, reranks, and
   keeps `reranker_top_k` for the LLM. When no reranker, behavior is identical to today.
7. Rename existing `top_k` param to `retrieval_top_k` in `Experiment.__init__` (keep default=5)
8. New feature columns: `mean_rerank_score`, `var_rerank_score` (analogous to existing
   `mean_retrieval_score`, `var_retrieval_score`)
9. Replace `build_reranker_placeholder()` in `src/metadata.py` with `build_reranker_metadata()`
   that returns real values when a reranker is used, None when not
10. `--reranker` and `--reranker-top-k` CLI flags in `scripts/run_experiment.py`
11. All existing tests continue to pass (reranker=None is the default)

## Files to Create

### `src/rerankers/__init__.py`
Export `MiniLMReranker` and `BGEReranker`.

### `src/rerankers/minilm.py`
`MiniLMReranker` class:
- `__init__(self)` — loads `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
- `name` property returns `"minilm:ms-marco-MiniLM-L-6-v2"`
- `rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]`
  - Builds `(query, chunk["text"])` pairs
  - Calls `self._model.predict(pairs)` — returns raw logits
  - Applies sigmoid to get 0-1 scores (same pattern as `CrossEncoderFilter._sigmoid`)
  - Adds `rerank_score` to each chunk dict (preserve original `score` and `index`)
  - Sorts by `rerank_score` descending
  - If `top_k` is not None, truncates to `top_k` results
  - Returns the reranked list

### `src/rerankers/bge.py`
`BGEReranker` class — identical pattern, different model:
- Loads `CrossEncoder("BAAI/bge-reranker-v2-m3")`
- `name` returns `"bge:bge-reranker-v2-m3"`
- Same `rerank()` method

**Note:** Both models output raw logits. The sigmoid normalization is important because:
1. It makes scores comparable across models (both in 0-1 range)
2. It makes `mean_rerank_score` / `var_rerank_score` interpretable as features
3. The existing `CrossEncoderFilter` already does this (see `src/query_filters/cross_encoder.py:127-138`)

## Files to Modify

### `src/protocols.py` — Add Reranker Protocol
Add after the `Scorer` Protocol block (~line 86):
```python
@runtime_checkable
class Reranker(Protocol):
    """Interface for chunk reranking backends."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'minilm:ms-marco-MiniLM-L-6-v2')."""
        ...

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerank retrieved chunks by relevance to query.

        Args:
            query: The search query text.
            chunks: List of dicts with 'text', 'score', 'index' keys
                    (output from Retriever.retrieve()).
            top_k: If provided, return only the top_k highest-scoring chunks.
                   If None, return all chunks in reranked order.

        Returns:
            List of dicts with 'text', 'score' (original), 'rerank_score',
            'index' keys, sorted by rerank_score descending.
        """
        ...
```

### `src/experiment.py` — Integrate reranker into run loop
1. Rename `top_k` param to `retrieval_top_k` (default 5) in `__init__`
2. Add `reranker: Reranker | None = None` and `reranker_top_k: int | None = None` params
3. Add validation: if `reranker` is not None and `reranker_top_k` is None, raise `ValueError`
4. Add validation: if `reranker` is not None, check `isinstance(reranker, Reranker)`
5. Store `self._reranker` and `self._reranker_top_k`
6. In the run loop, after `retrieved = retriever.retrieve(query["text"])`:
   - If `self._reranker` is not None: `reranked = self._reranker.rerank(query["text"], retrieved, self._reranker_top_k)`
   - Use `reranked` (or `retrieved` if no reranker) for features and context metadata
   - Compute `mean_rerank_score` and `var_rerank_score` from `rerank_score` field
   - If no reranker, set both to None
7. Replace `build_reranker_placeholder()` call with `build_reranker_metadata()`
8. Update the `self._top_k` reference to `self._retrieval_top_k` everywhere

### `src/metadata.py` — Replace placeholder with real builder
Replace `build_reranker_placeholder()` with:
```python
def build_reranker_metadata(name: str | None = None, top_k: int | None = None) -> dict:
    """Build reranker metadata dict.

    Args:
        name: Reranker .name property string, or None if no reranker used.
        top_k: Reranker output top_k, or None.

    Returns:
        Dict with keys: reranker_model, reranker_top_k.
    """
    return {"reranker_model": name, "reranker_top_k": top_k}
```

### `src/features.py` — No changes needed
The `mean_rerank_score` and `var_rerank_score` are computed in `experiment.py` (not in
`extract_features`) because they depend on the reranker output, which is a pipeline-level
concern, not a feature extraction concern. `extract_features` receives `retrieved` (the
final chunks, whether reranked or not) and computes its existing features as before.

### `scripts/run_experiment.py` — Add CLI flags
1. Add `--reranker` flag: choices `minilm`, `bge`, `none` (default: `none`)
2. Add `--reranker-top-k` flag: int (default: None)
3. Add `--retrieval-top-k` flag: int (default: 5) — replaces the hardcoded 5
4. Validate: if `--reranker` is not `none`, `--reranker-top-k` is required
5. Build the reranker instance and pass to `Experiment()`

### `tests/test_integration.py` — Add import health check
Add `test_import_rerankers` to `TestImportHealth`:
```python
def test_import_rerankers(self):
    from src.rerankers import MiniLMReranker, BGEReranker
```

### `tests/test_metadata.py` — Update reranker test
Replace `TestBuildRerankerPlaceholder` with tests for `build_reranker_metadata()`.

## New Dependencies

None — `sentence-transformers` is already installed, which provides `CrossEncoder`.
The model weights (~25MB for MiniLM, ~1.1GB for BGE) download automatically on first use
via HuggingFace Hub (same pattern as HuggingFaceEmbedder and CrossEncoderFilter).

## Edge Cases

- **Empty chunk list to reranker**: return empty list (no crash)
- **top_k larger than input chunks**: return all chunks (don't pad)
- **top_k=0**: return empty list
- **Single chunk**: return it with rerank_score (no comparison needed but should still work)
- **reranker=None**: behavior identical to current code (all reranker metadata columns are None)
- **retrieval_top_k < reranker_top_k**: valid but wasteful — no validation needed, reranker
  just returns fewer than requested (same as empty case)

## Decisions Made

- **Sigmoid normalization**: Apply sigmoid to raw logits. **Why:** Makes scores comparable
  across models, matches existing CrossEncoderFilter pattern, produces interpretable 0-1
  features for the meta-learner.
- **Reranker as separate Protocol, not folded into Retriever**: **Why:** Cleaner separation
  of concerns, easier to toggle on/off, consistent with the project's Protocol pattern.
  User explicitly chose this.
- **Two explicit top_k params (`retrieval_top_k`, `reranker_top_k`)**: **Why:** User wants
  both as experimental variables. Auto-calculating would hide a knob.
- **rerank_score added to chunk dict, original score preserved**: **Why:** Both are useful
  features. The original retrieval score shows embedding confidence; the rerank score shows
  cross-encoder confidence. The meta-learner can use both.
- **ms-marco-MiniLM-L-6-v2 and bge-reranker-v2-m3**: **Why:** Most widely used open-source
  rerankers. 13x parameter gap gives genuine fast-vs-deep comparison. Both local, zero API
  cost. Both use sentence-transformers CrossEncoder (already in venv).
- **Rename `top_k` to `retrieval_top_k`**: **Why:** With two top_k values in the pipeline,
  the generic name is ambiguous. Explicit naming prevents confusion.

## What NOT to Touch

- `src/retriever.py` — Retriever stays retrieval-only. No reranking logic here.
- `scripts/run_experiment_0.py` — Experiment 0 is scorer validation; reranking is not
  relevant there. Its placeholder metadata columns (None) remain correct.
- `src/query_filters/cross_encoder.py` — This is a query filter, not a chunk reranker.
  Different purpose, different interface. Don't merge or refactor.
- Strategy implementations — strategies call `retriever.retrieve()` internally for their
  own needs (e.g., SelfRAG re-retrieves). The reranker operates on the *experiment-level*
  retrieval for features/context, not inside strategies.

## Testing Approach

### `tests/test_reranker.py` — Unit tests for both implementations
- Protocol compliance: both satisfy Reranker protocol
- `.name` format is correct
- `rerank()` with normal input returns dicts with all required keys
- `rerank()` preserves original `score` and `index` fields
- `rerank()` output sorted by `rerank_score` descending
- `rerank()` with `top_k` truncates correctly
- `rerank()` with empty input returns empty list
- `rerank()` with `top_k=0` returns empty list
- `rerank()` with `top_k > len(chunks)` returns all chunks
- `rerank_score` values are between 0 and 1 (sigmoid applied)

### `tests/test_experiment_timing.py` / `tests/test_e2e_smoke.py` — Update existing
- Verify existing tests still pass with `reranker=None` (default)
- The `retrieval_top_k` rename must not break any existing test that passes `top_k=`

### Integration in `tests/test_integration.py`
- Import health for `src.rerankers`

### Metadata tests in `tests/test_metadata.py`
- `build_reranker_metadata()` with values
- `build_reranker_metadata()` with None (no reranker)

Run: `pytest tests/test_reranker.py tests/test_metadata.py tests/test_integration.py tests/test_e2e_smoke.py tests/test_experiment_timing.py -v`
Full suite: `pytest tests/ -v`
