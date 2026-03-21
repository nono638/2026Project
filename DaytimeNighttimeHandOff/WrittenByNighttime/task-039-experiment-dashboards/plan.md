# Plan: task-039 — Experiment 1 & 2 Dashboards + Gallery Integration

## Files to Create
1. `scripts/generate_experiment1_dashboard.py` — 12 charts for strategy × model analysis
2. `scripts/generate_experiment2_dashboard.py` — 12 charts for chunking × model analysis

## Files to Modify
1. `scripts/generate_gallery.py` — Replace placeholder logic with real dashboard generation

## Approach
- Follow the pattern from `generate_experiment0_dashboard.py`:
  - `build_experimentN_figures(csv_path)` returns `list[tuple[str, go.Figure]]`
  - `generate_dashboard(csv_path, output_path)` writes self-contained HTML
  - `_fig_to_html(fig)` helper for chart embedding
- IBM colorblind palette, Viridis for heatmaps
- Handle missing columns and partial data gracefully
- Model ordering by parameter count, not alphabetically

## Ambiguities
- Spec says `build_experiment1_figures(csv_path: Path)` takes a Path but Exp 0's takes a DataFrame. Will follow spec and take Path (read CSV inside).
- Exp 0 gallery integration passes DataFrame to `build_experiment0_figures(df)` — will adapt gallery to match the Path-based signature for Exp 1/2.
