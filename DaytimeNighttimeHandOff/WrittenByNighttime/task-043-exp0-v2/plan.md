# Plan: task-043 — Experiment 0 v2

## Files to Modify

### 1. `scripts/experiment_utils.py`
- Add `_RerankedRetriever` class (~20 lines): wraps Retriever + reranker, `.retrieve()` chains retrieval → reranking, `.chunks` proxied from underlying retriever
- Update `generate_answer()`: add `reranker=None` and `reranker_top_k=None` params. When reranker is provided, wrap Retriever in `_RerankedRetriever` before passing to strategy

### 2. `scripts/run_experiment_0.py`
- **Delete** local `compute_f1()`, `exact_match()` — import from `experiment_utils`
- **Keep** `compute_bertscores()` (only used by Exp 0)
- **Remove** `generate_answers()` function — replace with calls to `experiment_utils.generate_answer()`
- **Add** CLI flags: `--reranker`, `--reranker-top-k`, `--retrieval-top-k`, `--difficulty`
- **Change** `--n` default to 150, `--output-dir` default to "results/experiment_0_v2"
- **Add** difficulty/question_type filtering before sampling
- **Add** `_build_reranker(name)` helper
- **Fix** scorer context: pass `context_sent_to_llm` instead of `doc_text` to scorer
- **Add** `answer_quality` column computation post-scoring (good/questionable/poor)
- **Add** new columns: difficulty, question_type, failure_stage, etc.
- **Update** report to include answer_quality distribution and failure_stage breakdown

### 3. `scripts/generate_gallery.py`
- Update `main()` to detect `results/experiment_0_v2/` alongside `results/experiment_0/`
- When v2 exists: generate both v1 and v2 sections on the Experiment 0 page
- v2 section adds two new charts: answer_quality distribution, failure_stage breakdown

## Ambiguities
- Spec says "retrieve 10, keep 3" but the retriever's current default top_k is 5. The `_RerankedRetriever` will pass `top_k=None` to the underlying retriever (let retrieval_top_k be set via CLI), and `reranker_top_k` to the reranker.
- answer_quality computation requires Sonnet column name `anthropic_claude_sonnet_4_20250514_quality` — will check for this exact column name.
