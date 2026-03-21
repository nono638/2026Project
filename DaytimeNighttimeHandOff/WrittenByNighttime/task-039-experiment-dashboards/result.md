# Result: task-039 — Experiment 1 & 2 Interactive Dashboards + Gallery Integration
**Status:** done
**Completed:** 2026-03-21T09:48:41

## Commits
- `<pending>` — night: task-039 experiment dashboards + gallery integration

## Test Results
- Command run: `.venv/Scripts/python -m pytest tests/test_experiment1_dashboard.py tests/test_experiment2_dashboard.py -v`
- Outcome: 22 passed, 0 failed
- Failures: none
- Regression suite: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v` — 549 passed, 0 failed

## Decisions Made
- Pre-written tests had `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent` which resolves to `DaytimeNighttimeHandOff/` not project root when run from WrittenByDaytime location. Fixed in copies placed in `tests/` to use `.parent.parent` (correct for tests/ directory depth).
- Used `"size"` instead of `"len"` in pandas `.agg()` — `"len"` is not a recognized aggregation function in newer pandas.
- Added `try/except` for `pd.errors.EmptyDataError` when reading CSV — `pd.DataFrame().to_csv()` writes a file with no columns that `pd.read_csv()` can't parse.
- Spec says `build_experiment1_figures(csv_path: Path)` takes Path (not DataFrame like Exp 0). Followed spec.

## Flags for Morning Review
- Pre-written test `PROJECT_ROOT` calculation is wrong for WrittenByDaytime location (4 parents deep = DaytimeNighttimeHandOff/, needs 5). Tests in `tests/` directory work correctly with the fix to 2 parents.
