# Plan: task-031 — Reranker Pipeline Stage

## Files to Create
1. `src/rerankers/__init__.py` — exports MiniLMReranker, BGEReranker
2. `src/rerankers/minilm.py` — MiniLMReranker using ms-marco-MiniLM-L-6-v2
3. `src/rerankers/bge.py` — BGEReranker using bge-reranker-v2-m3

## Files to Modify
1. `src/protocols.py` — add Reranker Protocol after Scorer block
2. `src/experiment.py` — rename top_k→retrieval_top_k, add reranker/reranker_top_k params, integrate reranker into run loop, compute mean/var rerank scores
3. `src/metadata.py` — replace build_reranker_placeholder() with build_reranker_metadata()
4. `scripts/run_experiment.py` — add --reranker, --reranker-top-k, --retrieval-top-k CLI flags
5. `tests/test_integration.py` — add test_import_rerankers
6. `tests/test_metadata.py` — update TestBuildRerankerPlaceholder to test new function

## Approach
- Create reranker implementations first (standalone)
- Add Protocol to protocols.py
- Modify experiment.py (rename + integrate)
- Update metadata.py
- Update CLI script
- Update existing tests for import health and metadata
- Run pre-written tests then full suite

## Ambiguities
- None identified; spec is detailed and explicit.
