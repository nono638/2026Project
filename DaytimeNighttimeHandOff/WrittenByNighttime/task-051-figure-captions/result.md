# Result: task-051 — Figure captions for Exp 0 v1/v2/v3 dashboards

**Status:** done
**Completed:** 2026-04-29T13:45:00

## Commits
- (filled in after commit) — night: task-051 figure captions

## Test Results
- Command run: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/`
- Outcome: 633 passed, 15 failed
- Failures: all 15 are pre-existing bert_score / exp12 failures (bert_score module install issue) — unrelated to this task
- Manual verification: `python scripts/generate_gallery.py` runs cleanly; `grep -c 'class="caption"'` shows v3=17, v1(gallery)=15, v2=3 captions present

## Decisions Made
- Captions live in a single `CAPTIONS` dict keyed by chart title in `scripts/generate_experiment0_dashboard.py`. Reused via `_caption_html(title)` helper from both the standalone v1 dashboard and the gallery's v1/v2/v3 generators.
- The four required captions (answer-length, judge-vs-gold scatter, inter-judge heatmap, BERTScore histogram) use the spec's verbatim text. Other charts use the same what/takeaway pattern.
- For v3, added a "Summary" card above the first chart with the actual finding (Claude Haiku r=0.450 wins, 6-judge ranking, n=500). Numbers read from `results/experiment_0_v3/report.md`.
- Restructured the v2/v3 try/except in `generate_gallery.py` to import helpers (`_fig_to_html`, `_caption_html`) separately from the figure-build call. Previously a single try/except meant a build failure swallowed the helper import, leaving captions blank. Now helpers always import; only `figures = build_experiment0_figures(...)` is in the inner try.

## Flags for Morning Review
- `build_experiment0_figures()` raises `TypeError: unsupported operand type(s) for /: 'str' and 'int'` on `results/experiment_0_v2/raw_scores.csv`. Pre-existing — likely a column-dtype issue in the v2 CSV (some numeric column read as string). Fix is out of scope for this task; v2 page now logs the warning instead of silently dropping captions. The v2 page still renders the 3 v2-specific charts with captions; only the standard scorer-comparison charts are missing for v2.
- v2 caption count is 3 because of the above. v1=15 and v3=17 are correct.
- Standalone v1 dashboard (`scripts/generate_experiment0_dashboard.py`) was also updated with captions; not regenerated in this task since the gallery is the user-facing path.

## Attempted Approaches
N/A — straightforward implementation. One iteration was needed when the initial v2/v3 caption injection didn't render — root-caused to the import-and-build try/except eating the helper import on a pre-existing v2 build failure. Fixed by separating helper imports.
