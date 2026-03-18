# task-019: Hybrid retrieval (dense + BM25 with RRF)

## Summary

Upgrade the `Retriever` class to support hybrid retrieval by default: dense vector search
(existing FAISS) combined with sparse keyword search (BM25Okapi) via Reciprocal Rank Fusion
(RRF). This is the production standard for RAG systems — dense-only retrieval misses exact
keyword matches (entity names, error codes, acronyms) while sparse-only misses semantic
similarity. Hybrid improves NDCG 26-31% over dense-only (Blended RAG, IBM 2024).

RRF avoids the weight-tuning problem by using rank positions instead of raw scores. BM25
scores are unbounded (50, 100+) while cosine similarity is 0-1, making direct score
combination fragile. RRF sidesteps this entirely — one less parameter to explain or hold
constant.

For our experiments, this is held constant (always hybrid, always RRF k=60). The
infrastructure supports dense-only and sparse-only modes for future users.

## Requirements

1. `Retriever.__init__` accepts a new `mode` parameter: `Literal["hybrid", "dense", "sparse"]`,
   default `"hybrid"`.
2. In `"hybrid"` mode, `retrieve()` runs both FAISS and BM25, fuses results with RRF (k=60),
   returns top_k results sorted by RRF score.
3. In `"dense"` mode, `retrieve()` behaves exactly as it does now (FAISS only). This is the
   backwards-compatible path.
4. In `"sparse"` mode, `retrieve()` uses BM25 only. No FAISS search. The embedder is still
   required at init (for the FAISS index) but unused at retrieval time.
5. BM25 index is built alongside the FAISS index at init time from the same chunks.
6. The return type of `retrieve()` is unchanged: `list[dict]` with `"text"`, `"score"`,
   `"index"` keys. In hybrid mode, `"score"` is the RRF fused score.
7. `Experiment.__init__` accepts a new `retrieval_mode` parameter (same type, default
   `"hybrid"`) and passes it through when constructing `Retriever` instances.
8. Empty chunk list still handled gracefully (return `[]`).
9. All existing tests continue to pass unchanged.

## Files to Modify

- `src/retriever.py` — Add BM25 index construction in `__init__`, add `_retrieve_sparse()`
  and `_fuse_rrf()` private methods, modify `retrieve()` to dispatch by mode. Add
  `_tokenize()` static method for BM25 preprocessing.
- `src/experiment.py` — Add `retrieval_mode` parameter to `Experiment.__init__`, pass it
  through to `Retriever(chunks, embedder, top_k, mode=self._retrieval_mode)` on line 121.

## New Dependencies

- `rank-bm25==0.2.2` — Pure Python BM25 implementation. Used by LangChain and LlamaIndex
  internally. Provides `BM25Okapi`, `BM25L`, `BM25Plus`. We use `BM25Okapi` (industry
  default: Elasticsearch, Lucene, LangChain all default to Okapi BM25).
  - PyPI: https://pypi.org/project/rank-bm25/
  - GitHub: https://github.com/dorianbrown/rank_bm25

## Edge Cases

- **Empty chunks**: Return `[]` immediately, don't build BM25 index. Already handled for
  FAISS — add same guard for BM25.
- **Single chunk**: Both FAISS and BM25 return it. RRF gives it rank 1 from both sources.
  Works correctly.
- **top_k > len(chunks)**: Return all chunks. Both FAISS and BM25 handle this naturally.
  RRF union can have at most `len(chunks)` unique results.
- **Query has no BM25 matches** (all BM25 scores = 0): Those chunks get no sparse rank.
  Only dense results contribute to RRF. Effectively degrades to dense-only for that query.
  This is correct behavior.
- **Chunks with only stopwords or empty strings**: BM25 tokenization produces empty token
  lists. `rank_bm25` handles this — the chunk gets a score of 0. It may still appear in
  dense results.

## Decisions Made

- **BM25Okapi, not BM25+ or BM25L**: Okapi BM25 is the universal default across
  Elasticsearch, Lucene, rank_bm25, LangChain, and LlamaIndex. BM25+ (Lv & Zhai, 2011)
  fixes an edge case with very short documents but isn't widely adopted. Using the
  standard maximizes reproducibility.
- **RRF with k=60, not weighted linear combination**: RRF uses rank positions instead of
  raw scores, avoiding the need to normalize incompatible score scales (BM25 unbounded,
  cosine 0-1). k=60 is the default in Elasticsearch 8.8+, Azure AI Search, OpenSearch,
  and Chroma. No weight parameter to tune = no weight to hold constant in experiments.
  - Elasticsearch RRF docs: https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf
- **Simple tokenization** (lowercase + strip punctuation + split whitespace): Reproducible
  and deterministic. Not worth adding a tokenizer dependency (spaCy, NLTK) for BM25 input.
  Production systems often use more sophisticated tokenization, but for research
  reproducibility and minimal dependencies, simple is better.
- **Over-retrieve from both sources**: Retrieve `min(len(chunks), 100)` candidates from
  each source before fusing. FAISS IndexFlatIP brute-forces all vectors anyway (no speed
  penalty for larger k). BM25 `get_scores()` scores all documents. The fused top-k will
  have better recall than retrieving only top-k from each source.
- **`mode` parameter on Retriever, not separate classes**: Keeps the codebase simple. One
  class, one cache key, one interface. Strategies don't know or care about retrieval mode.
- **Default is "hybrid"**: This is the production standard. Dense-only is available via
  `mode="dense"` for backwards compatibility or explicit testing.

## What NOT to Touch

- **Strategies** (`src/strategies/*.py`): They call `retriever.retrieve()` and get the same
  return type. No changes needed.
- **Features** (`src/features.py`): Uses `retriever.retrieve()` scores. With hybrid, scores
  become RRF values instead of cosine similarity. This is fine — features are internally
  consistent within an experiment since retrieval mode is held constant. Do not rename the
  feature columns.
- **Protocols** (`src/protocols.py`): Retriever is a concrete class, not a Protocol. No
  protocol changes needed.
- **Scorer** (`src/scorers/*.py`): Completely independent of retrieval.
- **ExperimentResult analysis methods**: Work on the DataFrame — agnostic to how retrieval
  happened.

## Implementation Details

### RRF Formula

```python
def _fuse_rrf(self, dense_results: list[dict], sparse_results: list[dict],
              top_k: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion of dense and sparse results.

    RRF score for document d = sum over retrievers R of: 1 / (k + rank_R(d))
    where rank_R(d) is the 1-based rank of d in retriever R's results.
    Documents not returned by a retriever get no contribution from that retriever.
    """
```

### Tokenization

```python
@staticmethod
def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, strip punctuation, split whitespace.

    Simple and deterministic — no external tokenizer dependency.
    """
    # Strip punctuation, lowercase, split on whitespace, filter empty
```

Use `str.translate` with `str.maketrans` for punctuation stripping (fastest pure Python
approach). Import `string.punctuation` for the character set.

### Retriever.__init__ changes

```python
def __init__(self, chunks: list[str], embedder: Embedder, top_k: int = 5,
             mode: Literal["hybrid", "dense", "sparse"] = "hybrid") -> None:
```

- Always build the FAISS index (even in sparse mode — keeps the code simple, and the
  embedder is required anyway for the Experiment runner's cache key).
- Always build the BM25 index (even in dense mode — negligible cost, simplifies mode
  switching).
- Store `self._mode` for dispatch in `retrieve()`.

### Experiment.__init__ changes

```python
def __init__(self, ..., retrieval_mode: str = "hybrid") -> None:
```

Pass `mode=self._retrieval_mode` to every `Retriever(...)` construction. Add protocol
validation: assert `retrieval_mode in ("hybrid", "dense", "sparse")`.

## Testing Approach

- Pre-written tests in `tests/test_hybrid_retrieval.py` (see tests/ directory in this task)
- Tests use mock embedder (deterministic vectors) and known text chunks to verify:
  - Hybrid mode returns results combining both dense and sparse signals
  - Dense-only mode returns same results as current behavior
  - Sparse-only mode returns BM25 results only
  - RRF fusion correctly combines rankings (chunk ranked high in one but low in other)
  - Empty chunks return `[]`
  - top_k is respected
  - Experiment runner passes through retrieval_mode
- Run with: `pytest tests/test_hybrid_retrieval.py -v`
