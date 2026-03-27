# Result: task-049 — Gallery website overhaul

**Status:** done
**Completed:** 2026-03-27T00:27:27

## Commits
- `<pending>` — night: task-049 gallery v3 overhaul

## Test Results
- Command run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-049-gallery-v3-overhaul/tests/test_gallery_v3_overhaul.py -v`
- Outcome: 20 passed, 0 failed
- Failures: none
- Regression tests: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v` — 640 passed, 2 pre-existing failures (test_exp12_dry_run scorer columns — unrelated). The 1 new failure in test_gallery.py was fixed by updating the test to expect `experiment_0_v3.html` instead of `experiment_0.html`.

## Decisions Made
- Used exact prose from spec for both the hero section and "Road to v3" narrative — no creative license taken.
- Filtered Exp 0 variants from the card grid using `str(num).startswith("0")` check, so all Exp 0 entries (0, "0v2", "0v3") are excluded from cards. They're covered by the hero.
- Updated existing `tests/test_gallery.py::test_index_has_experiment_links` to expect `experiment_0_v3.html` since the home page now links to v3 instead of v1.
- Removed `experiments_info.append()` calls for v2 and v3 in `main()` since they no longer need cards (and their "0v2"/"0v3" num values generated broken links).

## Flags for Morning Review
- The existing `tests/test_gallery.py` assertion was updated — verify this is acceptable (it now checks for `experiment_0_v3.html` instead of `experiment_0.html` on the index page).
- The v3 chart code is largely copied from v2's chart code. If DRY refactoring is desired in the future, the shared chart logic could be extracted into a helper function.

## Attempted Approaches (if skipped/blocked)
N/A — completed on first approach.
