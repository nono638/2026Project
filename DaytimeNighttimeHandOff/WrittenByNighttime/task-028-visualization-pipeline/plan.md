# Plan: task-028 — Automated Visualization Pipeline

## Files to create
- `scripts/generate_visuals.py` — main script (~400 lines)

## Files to modify
- None (standalone script)

## Approach
1. Create `scripts/generate_visuals.py` with these functions:
   - `generate_explainer(answers_df, scores_df, out_dir)` — RAG pipeline diagram
   - `generate_exp0_heatmap(scores_df, exp0_dir)` — inter-judge correlation
   - `generate_exp0_scatter(scores_df, exp0_dir)` — judge vs gold BERTScore
   - `generate_exp0_distributions(scores_df, exp0_dir)` — box plots
   - `generate_exp0_bertscore_hist(scores_df, exp0_dir)` — BERTScore histogram
   - `generate_html(visuals_dir)` — HTML gallery page
   - `main()` with argparse for `--experiment N`

2. All functions handle missing data gracefully (skip + log warning).
3. Constants from spec: DPI=300, IBM colorblind palette, JUDGE_DISPLAY_NAMES mapping.
4. Style: seaborn-v0_8-whitegrid (confirmed available in matplotlib 3.10.8).

## CSV structure (confirmed)
- `raw_scores.csv`: example_id, question, gold_answer, rag_answer, gold_exact_match,
  gold_f1, {judge}_faithfulness/relevance/conciseness/quality, gold_bertscore
- `raw_answers.csv`: example_id, question, gold_answer, rag_answer, doc_text

## Ambiguities
- None significant. Spec is detailed with code snippets for each visual.
