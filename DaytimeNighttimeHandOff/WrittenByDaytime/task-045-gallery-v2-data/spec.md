# task-045: Regenerate Gallery with Experiment 0v2 Data

## What

Update the Experiment 0 v2 dashboard generator (`_generate_experiment_0_v2` in
`scripts/generate_gallery.py`) to handle the gemini-3.1-pro-preview partial data
gracefully, then regenerate the full gallery so `site/` reflects v2 results.

## Why

Experiment 0v2 completed with 150 scored questions, but gemini-3.1-pro-preview only
has 11/150 scores (rate-limited). The dashboard generator may break or show misleading
charts if it doesn't account for this. We need the gallery to render cleanly with the
data we have, and clearly note which judge has incomplete data.

## Files to Modify

- `scripts/generate_gallery.py` — update `_generate_experiment_0_v2()` only

## Files NOT to Touch

- `scripts/run_experiment_0.py` — do not modify
- `scripts/generate_experiment0_dashboard.py` — do not modify
- `results/` — do not modify any data files
- `src/` — no source changes

## Detailed Requirements

### 1. Handle partial judge data in v2 dashboard

In `_generate_experiment_0_v2()`, when building charts from `raw_scores.csv`:

- **Detect judges with incomplete data**: count non-null values per judge quality column.
  If a judge has < 50% non-null scores, mark it as "partial" in all charts.
- **Correlation heatmap**: exclude judges with < 50% data from the correlation matrix
  (correlations from 11 data points are meaningless). Add a note below the heatmap:
  "gemini-3.1-pro-preview excluded (11/150 scores due to API rate limit)."
- **Bar charts (mean scores)**: include partial judges but with a different color or
  hatching, and an asterisk + footnote: "* partial data (11/150 scored)".
- **Gold correlation chart**: exclude partial judges (same reasoning as correlation matrix).

### 2. Add v2 findings summary

At the top of the v2 dashboard page, add a findings summary section (HTML, not a chart):

```
Key Findings (v2 — 150 medium+hard HotpotQA questions, 7 LLM judges):
- Best judge by BERTScore correlation: Claude Haiku (r=0.640)
- Best free judge: Gemini 2.5 Pro (r=0.518)
- Pipeline accuracy: 74% exact match, 0.917 mean BERTScore
- Answer quality: 49% good, 47% poor, 5% questionable
- Failure stages: 74% none, 13% retrieval, 13% generation
- Note: gemini-3.1-pro-preview scored only 11/150 (API rate limit) — excluded from correlations
```

### 3. Regenerate gallery

After code changes, run `python scripts/generate_gallery.py` and verify:
- `site/experiment_0_v2.html` renders without errors
- Charts display correctly with partial-data handling
- `site/experiment_0.html` (v1) still renders correctly
- `site/index.html` shows both v1 and v2 experiment cards

## Edge Cases

- If ALL judge columns are null for a row, skip that row (don't crash)
- If gemini-3.1-pro-preview later gets full 150/150 data (re-run), the partial-data
  logic should not trigger (threshold is < 50% non-null)
- The v1 dashboard should be completely unaffected by these changes

## What NOT to Do

- Don't remove gemini-3.1-pro-preview from the data — just handle it gracefully
- Don't modify the experiment runner or scoring code
- Don't add new Python dependencies
- Don't change the v1 dashboard generation logic
