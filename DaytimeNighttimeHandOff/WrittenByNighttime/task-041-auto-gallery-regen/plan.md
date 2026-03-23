# Plan: task-041 — Auto-regenerate gallery after experiment runs

## Approach

For each of the three experiment scripts (`run_experiment_0.py`, `run_experiment_1.py`,
`run_experiment_2.py`):

1. **Add `--no-gallery` flag** to `parse_args()` — `action="store_true"`, default `False`.
2. **Add gallery regeneration block** at the end of `main()`, after the "Experiment N complete"
   print block (before `if __name__`). The block:
   - Checks `args.no_gallery` — if True, skip silently.
   - Uses a **lazy import** inside a try/except to import `generate_gallery.main`.
   - Prints "Regenerating gallery..." before the call.
   - Calls `generate_gallery.main(experiments=[N])` where N is 0, 1, or 2.
   - Prints "Gallery updated in site/" on success.
   - Catches any Exception (including ImportError) and prints a warning instead of crashing.

## Files to Modify

- `scripts/run_experiment_0.py` — lines ~196 (parse_args) and ~662 (end of main)
- `scripts/run_experiment_1.py` — lines ~155 (parse_args) and ~599 (end of main)
- `scripts/run_experiment_2.py` — lines ~183 (parse_args) and ~618 (end of main)

## No New Files

No new files needed. No changes to `generate_gallery.py`.

## Ambiguities

None — spec is clear. Lazy import + try/except, `--no-gallery` opt-out, per-experiment
regeneration.
