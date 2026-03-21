# task-035: Findings Gallery — Static Site from Experiment Results

## Summary

Build a static HTML site generator that turns experiment results into a browsable
findings gallery. Uses the same Plotly pattern as `generate_experiment0_dashboard.py`
(self-contained HTML, no server needed). Produces an index page linking to per-experiment
dashboards, with a landing page showing key cross-experiment takeaways.

This is the "Explorer" audience deliverable — someone curious about RAG browses
pre-computed results without running anything. Built with Exp 0 data now, structured
so Exp 1 & 2 dashboards slot in when those results exist.

## Requirements

1. A new script `scripts/generate_gallery.py` that produces a complete static site in `site/` directory
2. **Index page** (`site/index.html`): project title, one-paragraph summary, links to each experiment dashboard, key findings highlights section
3. **Experiment 0 dashboard** (`site/experiment_0.html`): generated from `results/experiment_0/raw_scores.csv`. Reuse the visualization logic from `generate_experiment0_dashboard.py` — import or adapt, don't duplicate
4. **Experiment 1 placeholder** (`site/experiment_1.html`): "Coming soon" page with planned experiment description (5 strategies × 6 models). Generated only if `results/experiment_1/` doesn't exist. If it does exist, generate a real dashboard.
5. **Experiment 2 placeholder** (`site/experiment_2.html`): Same pattern — placeholder or real dashboard depending on data existence.
6. All pages share a consistent nav bar (Home, Exp 0, Exp 1, Exp 2) and visual style
7. All HTML is self-contained — inline CSS and JS, no external CDN dependencies. Plotly.js is embedded via `plotly.io.to_html(full_html=False)` wrapped in a page template
8. A shared CSS theme: clean, readable, dark nav bar, light content area, IBM colorblind-safe palette (same as existing dashboard)
9. Script is runnable as: `python scripts/generate_gallery.py` (no args needed, sensible defaults)
10. Optional `--output` flag for output directory (default: `site/`)
11. Optional `--experiments` flag to generate only specific experiments (e.g., `--experiments 0,1`)

## Files to Modify

### `scripts/generate_gallery.py` — Create new
Main gallery generator script. Structure:
- `_build_page_template(title, nav_active, content_html)` → wraps content in shared HTML shell with nav + CSS
- `_generate_index(experiments_info)` → builds index page with findings highlights
- `_generate_experiment_0(csv_path)` → builds Exp 0 dashboard (adapt from existing script)
- `_generate_placeholder(experiment_num, description)` → builds placeholder pages
- `main()` → discovers available results, generates all pages

### `scripts/generate_experiment0_dashboard.py` — Minor refactor
Extract the visualization-building functions so they can be imported by the gallery
generator. Currently this script is monolithic — the chart-building functions are fine
as-is, but `main()` builds and writes the full HTML file directly. Refactor:
- Keep all existing chart functions unchanged
- Extract a `build_experiment0_figures(df)` function that returns a list of `(title, plotly_figure)` tuples
- Keep the existing `main()` working as before (backward compat) — it calls `build_experiment0_figures()` then wraps in HTML
- The gallery generator imports `build_experiment0_figures()` and wraps in its own template

### `site/.gitkeep` — Create new
Empty file so the output directory is tracked (actual generated HTML is .gitignored).

### `.gitignore` — Modify
Add `site/*.html` so generated files aren't committed (the generator is the source of truth).

## New Dependencies

None — Plotly is already installed. All HTML generation is string templates.

## Edge Cases

1. **`results/experiment_0/` doesn't exist**: skip Exp 0 dashboard, show placeholder on index. Print warning to stdout.
2. **`raw_scores.csv` exists but is empty**: skip that experiment's dashboard, show "No data" on its page.
3. **`site/` directory doesn't exist**: create it.
4. **Running from project root vs scripts dir**: use `Path(__file__).resolve().parent.parent` for project root (same pattern as other scripts).
5. **Plotly not installed**: fail with clear error message (it's already installed, but be explicit).

## Decisions Made

- **Single generator script, not a framework**: `generate_gallery.py` produces static HTML directly. **Why:** no build toolchain, no Node.js, no dependencies beyond what we have. Matches the project's pure-Python approach. Can revisit with a real framework later if needed.
- **Inline CSS/JS, no CDN**: self-contained HTML files that work offline. **Why:** the gallery might be viewed from a file:// URL during demo, and CDN dependencies are a point of failure. Plotly.js inlined via plotly.io makes this free.
- **Import from existing dashboard script, don't duplicate**: the Exp 0 charts already work well. Extract and import rather than copy-paste. **Why:** DRY — changes to chart logic propagate to both standalone dashboard and gallery.
- **Placeholder pages generated, not hand-written**: even "Coming soon" pages go through the generator so they get the shared nav/style. **Why:** consistency, and they auto-upgrade to real dashboards when data appears.
- **`.gitignore` the output, track the generator**: `site/*.html` is generated artifact. **Why:** the script is the source of truth. Avoids stale HTML in git.
- **IBM colorblind-safe palette carried over**: same `_COLORS` as existing dashboard. **Why:** accessibility, visual consistency.

## What NOT to Touch

- **`scripts/generate_visuals.py`** — the old matplotlib pipeline. Leave it as-is; the gallery uses Plotly.
- **`results/experiment_0/`** — read-only. Don't modify experiment data.
- **`scripts/run_experiment.py`** — experiment runner is separate from visualization.
- **Existing test files** — this task doesn't change any source modules, only adds a script.

## Testing Approach

Pre-written tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-035-findings-gallery/tests/`.
Tests go to `tests/test_gallery.py`.

Tests cover:
- Page template renders valid HTML with nav and title
- Index page contains links to experiment pages
- Experiment 0 dashboard generates from CSV data
- Placeholder pages have "Coming soon" content
- Missing data directory handled gracefully
- Output directory creation
- `build_experiment0_figures()` returns list of (title, figure) tuples

Run with: `pytest tests/test_gallery.py -v`
