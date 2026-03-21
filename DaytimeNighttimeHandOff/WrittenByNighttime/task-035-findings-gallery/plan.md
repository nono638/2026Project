# Plan: task-035 — Findings Gallery

## Approach

### `scripts/generate_experiment0_dashboard.py` — Minor refactor
- Extract `build_experiment0_figures(df)` that returns list of (title, plotly_figure) tuples
- Keep all existing chart functions unchanged
- Keep existing `main()` working (backward compat)

### `scripts/generate_gallery.py` — Create new
- `_build_page_template(title, nav_active, content_html)` — shared HTML shell with nav + CSS
- `_generate_index(experiments_info)` — landing page with experiment links and findings highlights
- `_generate_experiment_0(csv_path)` — builds Exp 0 dashboard page (imports from existing script)
- `_generate_placeholder(experiment_num, description)` — "Coming soon" pages
- `main(results_dir, output_dir)` — discovers available results, generates all pages

### Other changes
- `site/.gitkeep` — create
- `.gitignore` — add `site/*.html`

## Ambiguities
- The existing dashboard uses CDN Plotly JS, but spec says self-contained (no CDN). Will use plotly.io.to_html(full_html=False) for charts and include Plotly JS inline in the page template.
- build_experiment0_figures needs to work with the existing chart generators which have various signatures (some need judges, answers_df). Will create a simpler version that generates the subset of charts that only need the scores_df.
