# Plan: task-045 — Regenerate Gallery with Experiment 0v2 Data

## Files to Modify
- `scripts/generate_gallery.py` — update `_generate_experiment_0_v2()` only

## Approach

### 1. Detect partial judges
- After loading CSV, scan `*_quality` columns for non-null counts
- Mark judges with < 50% non-null as "partial"
- Store partial info (judge name, count) for use in charts

### 2. Add findings summary at top
- Insert HTML section at the top of the v2 dashboard content with hardcoded findings

### 3. Update correlation/gold charts
- Before building standard figures via `build_experiment0_figures`, filter out partial judge columns
- Or pass exclusion info through the data (drop partial columns before analysis)

### 4. Update bar charts
- Include partial judges but with distinct color + asterisk footnote

### 5. Regenerate gallery
- Run gallery generator, verify output

## Ambiguity
- The spec says to update `_generate_experiment_0_v2()` only, but `build_experiment0_figures()`
  from the external dashboard script may include partial judges. Simplest approach: drop partial
  judge columns from the DataFrame before passing to `build_experiment0_figures()`, then add them
  back for the bar charts.
