# task-041: Auto-regenerate gallery after experiment runs

## Summary

Wire the findings gallery generation into each experiment script so that when an
experiment completes, its gallery page (and the index) automatically regenerate.
Currently, a user must manually run `python scripts/generate_gallery.py` after each
experiment. This task adds a `--no-gallery` opt-out flag to each experiment script and
calls `generate_gallery.main()` at the end of each run by default.

## Requirements

1. After `run_experiment_0.py` finishes (after the final print block), call
   `generate_gallery.main(experiments=[0])` to regenerate the Experiment 0 page and
   the gallery index.
2. After `run_experiment_1.py` finishes (after the final print block), call
   `generate_gallery.main(experiments=[1])` to regenerate the Experiment 1 page and
   the gallery index.
3. After `run_experiment_2.py` finishes (after the final print block), call
   `generate_gallery.main(experiments=[2])` to regenerate the Experiment 2 page and
   the gallery index.
4. Each experiment script gains a `--no-gallery` CLI flag (action `store_true`,
   default `False`). When set, skip gallery regeneration entirely.
5. Gallery regeneration failures must NOT crash the experiment script. Wrap the call
   in a try/except that logs a warning and continues. The experiment data is already
   saved to CSV at this point — a gallery build error must not lose it.
6. Print a clear message before and after gallery regeneration:
   `"Regenerating gallery..."` and `"Gallery updated in site/"` (or the warning on
   failure).

## Files to Modify

- `scripts/run_experiment_0.py`
  - Add `--no-gallery` argument to the argparse parser
  - At the end of `main()`, after the "Experiment 0 complete" print block (line ~662),
    add the gallery regeneration call
  - Add import: `from scripts.generate_gallery import main as generate_gallery`
    (or use a lazy import inside the function to avoid circular issues)

- `scripts/run_experiment_1.py`
  - Same pattern: add `--no-gallery` argument
  - At the end of `main()`, after the "Experiment 1 complete" print block (line ~599),
    add the gallery regeneration call
  - Same import

- `scripts/run_experiment_2.py`
  - Same pattern: add `--no-gallery` argument
  - At the end of `main()`, after the "Experiment 2 complete" print block (line ~618),
    add the gallery regeneration call
  - Same import

## New Dependencies

None — `generate_gallery.py` and all its dependencies (plotly, pandas) are already
installed.

## Edge Cases

- **Gallery generation fails** (e.g., plotly import error on RunPod, missing data
  file): Log warning, print "Gallery regeneration failed: {error}", continue without
  crashing. The experiment CSV is already saved.
- **`--no-gallery` is set**: Skip gallery regeneration silently (just log at debug
  level).
- **Experiment runs on RunPod without site/ directory**: `generate_gallery.main()`
  already creates the output dir with `mkdir(parents=True)`, so this is fine.
- **Partial experiment results (cost limit hit)**: Gallery should still regenerate —
  it works with whatever CSV data exists.
- **Import failure** (generate_gallery not importable): Use a lazy import inside the
  gallery regen block so the experiment script still works even if generate_gallery.py
  has an issue. Catch ImportError in the try/except.

## Decisions Made

- **Lazy import over top-level import**: Use a lazy import inside the gallery regen
  block rather than a top-level `from scripts.generate_gallery import main`. **Why:**
  Avoids potential import issues (plotly not installed, circular imports) from breaking
  the experiment script itself. The experiment data must always be saved regardless.
- **Per-experiment regeneration, not full rebuild**: Call
  `generate_gallery.main(experiments=[N])` with only the relevant experiment number.
  **Why:** Faster — no need to rebuild all 3 pages when only one experiment just ran.
  The index always regenerates (it's built from whatever experiments are discovered).
- **`--no-gallery` as opt-out, not opt-in**: Default is to regenerate. **Why:** The
  whole point is automatic generation — users who don't want it can opt out, but the
  default should be the convenient path.
- **Warning, not error, on failure**: **Why:** The experiment data is the primary
  artifact. The gallery is a convenience. Never lose data over a gallery rendering
  issue.

## What NOT to Touch

- `scripts/generate_gallery.py` — it already has the right `main()` API. No changes
  needed.
- `scripts/generate_experiment0_dashboard.py`, `generate_experiment1_dashboard.py`,
  `generate_experiment2_dashboard.py` — these are called by generate_gallery.py, not
  by the experiment scripts directly.
- The experiment logic itself (scoring, CSV writing, report generation) — only add
  code AFTER the existing completion block.

## Testing Approach

- Test that each experiment script's `main()` calls gallery generation by default
  (mock `generate_gallery.main` and assert it was called with the right experiment
  number).
- Test that `--no-gallery` flag prevents the call.
- Test that gallery generation failure is caught and logged, not re-raised.
- Tests use `unittest.mock.patch` to mock the gallery import.
- Run with: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-041-auto-gallery-regen/tests/`
