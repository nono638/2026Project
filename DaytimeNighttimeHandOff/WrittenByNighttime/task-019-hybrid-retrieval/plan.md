# Plan: task-019 — Hybrid retrieval (dense + BM25 + RRF)

## Files to Modify
- `src/retriever.py` — add BM25 index, mode parameter, RRF fusion
- `src/experiment.py` — add retrieval_mode parameter, pass through to Retriever

## Approach
1. Add `mode` parameter to Retriever.__init__
2. Build BM25 index alongside FAISS at init time
3. Add `_retrieve_dense()`, `_retrieve_sparse()`, `_retrieve_hybrid()`, `_fuse_rrf()` methods
4. Add `_tokenize()` static method for BM25 preprocessing
5. Update Experiment to accept and pass through `retrieval_mode`
6. Run tests

## Ambiguities
- None — spec is very detailed with implementation code examples.
