# Plan: task-049 — Gallery website overhaul

## Files to modify
- `scripts/generate_gallery.py` — all changes in this one file

## Approach

### Change 1: _NAV_ITEMS (line 284)
Change `("exp0", "Exp 0: Scorer Validation", "experiment_0.html")` to
`("exp0v3", "Exp 0: Scorer Validation", "experiment_0_v3.html")`

### Change 2: _generate_index() (line 346)
Replace the current hero + cards layout with:
- New hero from spec's exact prose (RAGBench + v3 centerpiece)
- Keep key findings grid as-is
- Filter out Exp 0 v2/v3 cards from experiments_info (only show Exp 1, 2, Methodology)

### Change 3: nav_active updates
Change `nav_active="exp0"` to `nav_active="exp0v3"` in both `_generate_experiment_0_v2()` (line 979) and `_generate_experiment_0()` (line 1450)

### Change 4: New _generate_experiment_0_v3() function
Place after `_generate_experiment_0_v2()` (around line 982). This function:
- Includes the "Road to v3" narrative section (exact prose from spec)
- Key findings card with v3-specific stats
- Download link to raw_scores_v3.csv
- Reuses chart logic from v2 generator (copy the chart-building code since it's self-contained)
- Uses `nav_active="exp0v3"`

### Change 5: main() updates (line 1968+)
- v3 block: call `_generate_experiment_0_v3()` instead of `_generate_experiment_0_v2()`
- Remove v2/v3 from experiments_info (they had broken card links with "0v2" and "0v3")
- Keep the v1 card as "num: 0" pointing to experiment_0.html (it remains)

## Ambiguities
- The spec says to remove separate v1/v2/v3 cards from the home page. I'll filter them out in the index generator by only passing Exp 1, 2 cards + methodology.
