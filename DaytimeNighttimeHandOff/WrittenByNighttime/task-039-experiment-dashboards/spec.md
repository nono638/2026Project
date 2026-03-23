# Task 039: Experiment 1 & 2 Interactive Dashboards

## What to Build

Two Plotly dashboard generators — one for Experiment 1 (strategy × model) and one for
Experiment 2 (chunking × model) — plus gallery integration so placeholder pages become
real dashboards when data exists.

**Pattern:** Follow `scripts/generate_experiment0_dashboard.py` exactly. Each script:
- Reads CSV from `results/experiment_N/raw_scores.csv`
- Generates interactive Plotly charts
- Writes a self-contained HTML file
- Exports a `build_experimentN_figures()` function for the gallery to call

## Files to Create

### 1. `scripts/generate_experiment1_dashboard.py`

Reads `results/experiment_1/raw_scores.csv`. Expects these columns (from `run_experiment_1.py`):
- `strategy` — one of: naive, self_rag, multi_query, corrective, adaptive
- `model` — one of: qwen3:0.6b, qwen3:1.7b, qwen3:4b, qwen3:8b, gemma3:1b, gemma3:4b
- `quality` — mean of faithfulness/relevance/conciseness (0-1 float)
- `faithfulness`, `relevance`, `conciseness` — individual scorer dimensions (0-1)
- `gold_f1`, `gold_exact_match`, `gold_bertscore` — extrinsic metrics
- `strategy_latency_ms`, `total_latency_ms` — timing
- `question`, `gold_answer`, `rag_answer` — text content

**Charts to generate (in this order):**

1. **Summary card** — HTML stats: total configs, total rows, best config (by mean quality),
   worst config, overall mean quality. Same pattern as `_chart_summary_card()` in Exp 0.

2. **Quality heatmap** — Plotly `Heatmap`: x=model, y=strategy, z=mean quality. Annotate
   cells with values. Order x-axis by model size (0.6b → 8b, with gemma interleaved by size).
   Order y-axis alphabetically. Use Viridis colorscale.

3. **Latency heatmap** — Same layout, z=mean strategy_latency_ms. Use Plasma colorscale.
   Annotate with values in seconds (divide by 1000, 1 decimal).

4. **Quality vs model size line plot** — One line per strategy. x=model size in B params
   (0.6, 1.0, 1.7, 4.0, 4.0, 8.0), y=mean quality. Add error bars (std). Use this mapping:
   ```python
   MODEL_SIZES = {
       "qwen3:0.6b": 0.6, "gemma3:1b": 1.0, "qwen3:1.7b": 1.7,
       "gemma3:4b": 4.0, "qwen3:4b": 4.0, "qwen3:8b": 8.0,
   }
   ```
   Note: gemma3:4b and qwen3:4b both at 4.0 — use different markers to distinguish.

5. **Latency vs model size line plot** — Same as above but y=mean strategy_latency_ms.
   Log scale on y-axis recommended.

6. **Strategy beats size analysis** — Grouped bar chart. For each non-naive strategy, show
   cases where (strategy + small_model) beats (naive + larger_model) by mean quality.
   x=strategy, y=count of "beats" cases. Color by margin (delta quality). This is the
   project's core research question — make it visually prominent.

7. **Per-metric breakdown** — Grouped bar chart: x=config (strategy+model), y=score,
   grouped by metric (faithfulness, relevance, conciseness). Only show top-10 and bottom-5
   configs by quality to avoid visual clutter (30 configs × 3 metrics = too many bars).

8. **Score distributions by strategy** — Violin plots: one violin per strategy, y=quality.
   Shows spread and density of scores across all models and queries.

9. **Score distributions by model** — Violin plots: one violin per model (ordered by size),
   y=quality. Shows how model size affects score spread.

10. **Gold metrics by config** — Heatmap: x=model, y=strategy, z=mean gold_f1. Skip if
    gold_f1 column is all NaN.

11. **Quality vs latency scatter (Pareto)** — Scatter: x=mean strategy_latency_ms (log),
    y=mean quality. One point per config (strategy+model). Label each point. Draw Pareto
    frontier line connecting non-dominated configs (higher quality AND lower latency).

12. **Per-query detail table** — Interactive HTML table (Plotly `Table`) showing worst-10
    and best-10 individual answers by quality. Columns: strategy, model, question (truncated
    50 chars), quality, gold_f1. Helps identify where RAG breaks down.

**Key functions:**
- `build_experiment1_figures(csv_path: Path) -> list[tuple[str, go.Figure]]` — returns
  (title, figure) pairs for gallery reuse
- `generate_dashboard(csv_path: Path, output_path: Path) -> None` — writes full HTML
- `_fig_to_html(fig: go.Figure) -> str` — reuse from Exp 0 or copy the pattern

**Output:** `visuals/experiment_1.html`

### 2. `scripts/generate_experiment2_dashboard.py`

Reads `results/experiment_2/raw_scores.csv`. Same structure as Exp 1 but:
- `chunker` column instead of `strategy` — one of: recursive, fixed, sentence, semantic
- `model` column — only Qwen3: qwen3:0.6b, qwen3:1.7b, qwen3:4b, qwen3:8b
- `chunk_size`, `chunk_overlap` — chunker parameters (vary by chunker type)

**Charts — same structure as Exp 1 but adapted:**

1. **Summary card** — same pattern
2. **Quality heatmap** — x=model, y=chunker, z=mean quality
3. **Latency heatmap** — x=model, y=chunker, z=mean strategy_latency_ms
4. **Quality vs model size line plot** — one line per chunker
5. **Latency vs model size line plot** — one line per chunker
6. **Chunking impact analysis** — Instead of "beats size": bar chart showing quality
   delta between best and worst chunker at each model size. Research question: does
   chunking matter more for small models or large models?
7. **Per-metric breakdown** — top/bottom configs (16 total is manageable — show all)
8. **Score distributions by chunker** — violin plots
9. **Score distributions by model** — violin plots
10. **Gold metrics by config** — heatmap
11. **Quality vs latency scatter (Pareto)** — same pattern
12. **Chunk count analysis** — Scatter: x=mean num_chunks, y=mean quality, per config.
    Shows whether more chunks help or hurt.

**Key functions:**
- `build_experiment2_figures(csv_path: Path) -> list[tuple[str, go.Figure]]`
- `generate_dashboard(csv_path: Path, output_path: Path) -> None`

**Output:** `visuals/experiment_2.html`

## Files to Modify

### 3. `scripts/generate_gallery.py`

Replace placeholder generation for Exp 1 and 2 with real dashboard pages when data exists.

**Changes:**
- Import `build_experiment1_figures` from `generate_experiment1_dashboard`
- Import `build_experiment2_figures` from `generate_experiment2_dashboard`
- In the section that checks for experiment data (around line 484-495), replace the
  placeholder logic with real dashboard generation — same pattern as
  `_generate_experiment_0()` uses `build_experiment0_figures()`
- Keep placeholder as fallback when CSV doesn't exist or is empty

**Do NOT change:**
- Navigation structure (_NAV_ITEMS)
- Page template (_build_page_template)
- Index page (_generate_index)
- Experiment 0 page (_generate_experiment_0)
- CSS styling

## Design Decisions

- **Two separate scripts, not one parameterized script.** Exp 1 and 2 have different
  research questions and different signature charts (strategy-beats-size vs chunking-impact).
  Parameterizing would add complexity for little reuse. Copy-paste the structural pattern.

- **IBM colorblind-safe palette.** Use the same palette as generate_visuals.py:
  ```python
  IBM_COLORS = [
      "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000",
      "#000000", "#AAAAAA",
  ]
  ```

- **Viridis for heatmaps, IBM for line/bar charts.** Consistent with Exp 0 dashboard.

- **Handle missing data gracefully.** If a column is missing or all NaN, skip that chart
  with a log warning. Do not crash. Partial results (from cost limit) should still produce
  useful charts.

- **Model size ordering.** Always order models by parameter count, not alphabetically:
  0.6b → 1b → 1.7b → 4b (gemma) → 4b (qwen) → 8b.

## What NOT to Touch

- `scripts/generate_experiment0_dashboard.py` — do not modify
- `scripts/generate_visuals.py` — do not modify
- `src/` — no changes to library code
- `results/` — do not create test data here

## Testing

Tests go in `tests/test_experiment1_dashboard.py` and `tests/test_experiment2_dashboard.py`.

Test with synthetic DataFrames — do NOT depend on real experiment data.

See pre-written tests in the tests/ directory of this spec.
