# Plan: task-030 — Interactive Plotly Dashboard for Experiment 0

## Approach

Create `scripts/generate_experiment0_dashboard.py` that:
1. Reads raw_scores.csv and raw_answers.csv
2. Enriches with HotpotQA metadata (difficulty, question_type) if possible
3. Auto-detects valid judges (skips all-NaN columns)
4. Generates 18 visualizations as Plotly figures + HTML widgets
5. Combines into single self-contained HTML file

## Files Created
- `scripts/generate_experiment0_dashboard.py` — main script

## Key Functions
- `generate_dashboard()` — main entry point
- `enrich_with_hotpotqa_metadata()` — add HotpotQA difficulty/type columns
- `get_valid_judges()` — detect judge columns, skip all-NaN
- `JUDGE_DISPLAY_NAMES` — display name mapping dict
- Individual chart functions for each of the 18 visualizations

## Ambiguities
- The spec mentions `gold_bertscore` column — it exists in the CSV.
- The spec says "quality" metric for scatter plots — will use the `*_quality` columns.
- For pipeline walkthrough, doc_text comes from raw_answers.csv, other data from raw_scores.csv.
