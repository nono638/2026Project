# task-028: Automated Visualization Pipeline

## Summary

Create `scripts/generate_visuals.py` that reads experiment CSVs from `results/` and
generates publication-quality PNGs + a minimal HTML gallery page. The pipeline is fully
automated: run the script after any experiment and it produces all visuals. Currently
only Experiment 0 data exists, but the script structure must accommodate future
experiments (1, 2) and cross-experiment comparisons without refactoring.

## Requirements

1. Running `python scripts/generate_visuals.py` produces all visuals for all experiments
   that have data in `results/experiment_N/`.
2. The explainer diagram (`visuals/explainer_rag_pipeline.png`) shows one concrete example
   row flowing through the RAG pipeline — question, retrieved context, LLM, generated
   answer, then evaluation (gold answer, BERTScore, judge scores). A cold reader should
   understand what RAG does from this one figure.
3. Experiment 0 produces 4 visuals:
   - `judge_correlation_heatmap.png` — inter-judge Pearson r matrix
   - `judge_vs_gold_scatter.png` — one subplot per judge, x=BERTScore, y=quality
   - `score_distributions.png` — box plots of each judge's quality scores
   - `bertscore_distribution.png` — histogram of gold BERTScore values
4. All PNGs are 300 DPI, suitable for academic publication.
5. `visuals/index.html` is a self-contained static HTML page embedding all PNGs with
   section headers and one-line captions.
6. The script handles missing data gracefully: if an experiment directory or column
   doesn't exist, skip that visual and log a warning — don't crash.
7. The script accepts `--experiment N` to regenerate visuals for one experiment only.
   Without the flag, it regenerates everything.

## Files to Create

### `scripts/generate_visuals.py` — main script (~350-450 lines)

Structure:

```python
"""Automated visualization pipeline for experiment results.

Reads CSVs from results/experiment_N/ and generates PNGs + HTML gallery.

Usage:
    python scripts/generate_visuals.py                 # all experiments
    python scripts/generate_visuals.py --experiment 0  # one experiment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALS_DIR = PROJECT_ROOT / "visuals"

# --- Constants ---
DPI = 300
FIG_WIDTH = 8  # inches, standard for single-column academic figures
TITLE_SIZE = 14
LABEL_SIZE = 11
STYLE = "seaborn-v0_8-whitegrid"

# Map raw column prefixes to short display names for chart labels.
# Column prefixes in raw_scores.csv are like "google_gemini_2_5_flash_lite".
JUDGE_DISPLAY_NAMES = {
    "google_gemini_2_5_flash_lite": "Flash-Lite",
    "google_gemini_2_5_flash": "Flash",
    "google_gemini_2_5_pro": "Gemini Pro",
    "anthropic_claude_haiku_4_5_20251001": "Haiku",
    "anthropic_claude_sonnet_4_20250514": "Sonnet",
}

# Colorblind-friendly palette (IBM Design Library).
# Order matches JUDGE_DISPLAY_NAMES iteration order.
JUDGE_COLORS = {
    "Flash-Lite": "#648FFF",
    "Flash": "#785EF0",
    "Gemini Pro": "#DC267F",
    "Haiku": "#FE6100",
    "Sonnet": "#FFB000",
}
```

### Output directory structure

```
visuals/
  explainer_rag_pipeline.png
  experiment_0/
    judge_correlation_heatmap.png
    judge_vs_gold_scatter.png
    score_distributions.png
    bertscore_distribution.png
  index.html
```

Create `visuals/` and `visuals/experiment_0/` directories if they don't exist.

## Detailed Visual Specifications

### 1. Explainer Diagram (`explainer_rag_pipeline.png`)

**Purpose:** Orient a cold reader on what RAG is and what this project measures.

**Data source:** Join `raw_answers.csv` (has `doc_text`) with `raw_scores.csv` (has scores)
on `example_id`. Pick the row whose `gold_bertscore` is closest to the dataset median —
this gives an interesting example (not trivially perfect, not broken).

**Layout:** Single figure, ~10 x 12 inches, top-to-bottom vertical flow. Use
`FancyBboxPatch` for boxes and `annotate` with `arrowprops` for arrows.

```
┌─────────────────────────────────────────────┐
│           RAG Pipeline: One Example         │  <- title
├─────────────────────────────────────────────┤
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  QUESTION                             │  │  <- light blue (#E3F2FD)
│  │  "The Office of the Federal..."       │  │     textwrap.fill(question, 70)
│  └───────────────────────────────────────┘  │
│                    │                        │
│                    ▼                        │
│  ┌───────────────────────────────────────┐  │
│  │  RETRIEVED CONTEXT (top chunks)       │  │  <- light green (#E8F5E9)
│  │  "## United States Government Manual  │  │     first 300 chars of doc_text
│  │   The United States Government..."    │  │     + "..." if truncated
│  └───────────────────────────────────────┘  │
│                    │                        │
│                    ▼                        │
│  ┌──────────────────────┐                   │
│  │  LLM: Qwen3 4B       │                  │  <- light gray (#F5F5F5)
│  │  (via Ollama)         │                  │
│  └──────────────────────┘                   │
│                    │                        │
│                    ▼                        │
│  ┌───────────────────────────────────────┐  │
│  │  GENERATED ANSWER                     │  │  <- light yellow (#FFF8E1)
│  │  "The National Archives and Records   │  │     full rag_answer
│  │   Administration (NARA)"              │  │
│  └───────────────────────────────────────┘  │
│                    │                        │
│                    ▼                        │
│  ┌───────────────────────────────────────┐  │
│  │  EVALUATION                           │  │  <- light orange (#FFF3E0)
│  │                                       │  │
│  │  Gold answer: "National Archives..."  │  │
│  │  BERTScore:   0.952                   │  │
│  │  Sonnet:      5.0 / 5.0              │  │     show top 2 judges
│  │  Haiku:       5.0 / 5.0              │  │     that have data
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Implementation notes:**
- Use `fig, ax = plt.subplots(figsize=(10, 12))`
- `ax.set_xlim(0, 10)`, `ax.set_ylim(0, 12)`, `ax.axis("off")`
- Each box: `ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
  facecolor=color, edgecolor="#333333", linewidth=1.5))`
- Text inside boxes: `ax.text(x, y, text, fontsize=10, family="monospace",
  verticalalignment="top", wrap=True)`
- Arrows between boxes: `ax.annotate("", xy=(5, y_end), xytext=(5, y_start),
  arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))`
- Use `textwrap.fill()` to wrap long text to ~70 chars
- Truncate doc_text to first 300 characters + "..." — the point is to show that
  context exists, not to display it all

### 2. Judge Correlation Heatmap (`experiment_0/judge_correlation_heatmap.png`)

**Purpose:** Show inter-judge agreement. Are cheap judges redundant with expensive ones?

**Data:** From `raw_scores.csv`, extract `*_quality` columns. Compute pairwise Pearson r.
Only include judges that have >= 10 non-null scores.

**Layout:** ~8 x 6 inches. Square heatmap with `ax.imshow()`. Annotate each cell with
the r value. Use diverging colormap `RdYlGn` centered at 0 (red=negative, green=positive).
Display names on axes (not raw column names). Title: "Inter-Judge Correlation (Pearson r)".

**Implementation:**
```python
# Build correlation matrix from quality columns
quality_cols = {name: f"{prefix}_quality" for prefix, name in JUDGE_DISPLAY_NAMES.items()
                if f"{prefix}_quality" in df.columns}
# Filter to judges with >= 10 valid scores
valid = {name: col for name, col in quality_cols.items() if df[col].notna().sum() >= 10}
sub = df[list(valid.values())].rename(columns={v: k for k, v in valid.items()})
corr = sub.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
# Annotate cells with r values
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=LABEL_SIZE)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
fig.colorbar(im, label="Pearson r")
ax.set_title("Inter-Judge Correlation (Pearson r)", fontsize=TITLE_SIZE)
```

### 3. Judge vs Gold Scatter (`experiment_0/judge_vs_gold_scatter.png`)

**Purpose:** Show which judges actually track semantic correctness (BERTScore).

**Data:** For each judge with >= 10 valid scores, scatter x=gold_bertscore, y=judge_quality.
Annotate each subplot with Pearson r.

**Layout:** Subplots in a single row (1 x N judges), ~3 inches wide per subplot.
If only 2 judges have data, figure is 6x4. If 5, it's 15x4 — that's fine, will scale.
Consistent axes: x from 0.7 to 1.0 (BERTScore range for this data), y from 1 to 5.

```python
fig, axes = plt.subplots(1, n_judges, figsize=(3 * n_judges, 4), sharey=True)
for ax, (name, col) in zip(axes, valid_judges.items()):
    valid = df[[col, "gold_bertscore"]].dropna()
    ax.scatter(valid["gold_bertscore"], valid[col], alpha=0.5, color=JUDGE_COLORS[name], s=40)
    r = valid[col].corr(valid["gold_bertscore"])
    ax.set_title(f"{name}\nr = {r:.3f}", fontsize=LABEL_SIZE)
    ax.set_xlabel("BERTScore")
    if ax == axes[0]:
        ax.set_ylabel("Judge Quality Score")
    ax.set_ylim(0.5, 5.5)
```

### 4. Score Distributions (`experiment_0/score_distributions.png`)

**Purpose:** Reveal whether a judge is discriminating (spread across 1-5) or
rubber-stamping (everything is 5).

**Data:** Quality columns for judges with >= 10 scores.

**Layout:** ~8 x 5 inches. Horizontal box plots, one per judge, ordered by median.
Color-coded using JUDGE_COLORS. Title: "Judge Score Distributions (Quality)".

```python
fig, ax = plt.subplots(figsize=(8, 5))
data = [df[col].dropna().values for col in valid_cols]
bp = ax.boxplot(data, vert=False, labels=display_names, patch_artist=True)
for patch, name in zip(bp["boxes"], display_names):
    patch.set_facecolor(JUDGE_COLORS[name])
    patch.set_alpha(0.7)
ax.set_xlabel("Quality Score (1-5)")
ax.set_title("Judge Score Distributions", fontsize=TITLE_SIZE)
```

### 5. BERTScore Distribution (`experiment_0/bertscore_distribution.png`)

**Purpose:** Show the baseline quality of RAG answers before judging. Tells the reader
whether the RAG system is producing mostly-good or mostly-bad answers.

**Data:** `gold_bertscore` column.

**Layout:** ~8 x 4 inches. Histogram with 20 bins. Vertical dashed line at median.
Annotate with mean, median, and count. Color: muted blue. Title: "RAG Answer Quality
(BERTScore F1 vs Gold)".

```python
fig, ax = plt.subplots(figsize=(8, 4))
scores = df["gold_bertscore"].dropna()
ax.hist(scores, bins=20, color="#648FFF", alpha=0.7, edgecolor="white")
ax.axvline(scores.median(), color="#333333", linestyle="--", label=f"Median: {scores.median():.3f}")
ax.set_xlabel("BERTScore F1")
ax.set_ylabel("Count")
ax.set_title("RAG Answer Quality (BERTScore F1 vs Gold)", fontsize=TITLE_SIZE)
ax.legend()
```

### 6. HTML Gallery (`visuals/index.html`)

**Purpose:** Browsable page that embeds all PNGs with context.

**Implementation:** Generate the HTML as a Python string in `generate_html()`.
Do NOT use a template engine — just f-strings.

**Structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGBench — Experiment Results</title>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px;
               margin: 0 auto; padding: 2rem; background: #fafafa; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 0.5rem; }
        h2 { margin-top: 2.5rem; color: #444; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd;
              border-radius: 4px; margin: 1rem 0; }
        .caption { color: #666; font-size: 0.9rem; margin-top: -0.5rem; }
        .missing { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <h1>RAGBench — Experiment Results</h1>
    <p>Auto-generated visualizations. Re-run <code>python scripts/generate_visuals.py</code>
       to regenerate.</p>

    <h2>How RAG Works</h2>
    <img src="explainer_rag_pipeline.png" alt="RAG Pipeline Explainer">
    <p class="caption">One example from HotpotQA showing each stage of the pipeline.</p>

    <h2>Experiment 0: Scorer Validation</h2>
    <!-- embed each exp0 image with caption -->
    <!-- if image doesn't exist, show <p class="missing">Not yet generated.</p> -->
</body>
</html>
```

The HTML generator should scan `visuals/` for existing PNGs and only reference files
that actually exist. Use relative paths (images are in same directory tree).

## New Dependencies

None — matplotlib and pandas are already installed.

## Edge Cases

- **Judge has all NaN scores** (e.g., Gemini Pro): Skip that judge in all visuals.
  Log a warning: "Skipping {judge}: no valid scores."
- **Experiment directory doesn't exist**: Skip. Log: "No data for experiment {n}, skipping."
- **raw_answers.csv missing** (scores-only re-run): Skip the explainer diagram.
  Log a warning but don't crash.
- **gold_bertscore column missing**: Skip the BERTScore histogram and use gold_f1 as
  fallback in scatter plots. If neither exists, skip gold-correlation visuals entirely.
- **Only 1 judge has data**: Skip the correlation heatmap (needs >= 2). Generate the
  other visuals normally.
- **All BERTScores are identical**: The histogram will look like one bar. That's fine —
  don't add special handling.

## Decisions Made

- **Pure matplotlib, no seaborn**: Avoids a new dependency. matplotlib's built-in styles
  and `imshow` + manual annotation produce equivalent results for our chart types.
- **IBM colorblind-friendly palette**: 5 distinct colors that work for color-vision
  deficiency. Source: https://davidmathlogic.com/colorblind/
- **300 DPI for all PNGs**: Standard for academic publication. File sizes for charts
  are small (~100-300KB) so this doesn't bloat the web page.
- **Median-BERTScore row for explainer**: Avoids cherry-picking a perfect example.
  Shows realistic pipeline behavior including imperfect retrieval.
- **Truncate doc_text to 300 chars in explainer**: The point is to show context exists
  and is relevant, not to display the full document. 300 chars is ~4 lines.
- **>= 10 valid scores threshold**: Judges with fewer data points produce unreliable
  correlations. 10 is a reasonable minimum for Pearson r.
- **Horizontal box plots**: Easier to read judge names on the y-axis than rotated
  x-axis labels.
- **HTML with inline CSS, no framework**: Minimal, zero dependencies, works offline.
  The gallery is a presentation layer, not an app.
- **`matplotlib.use("Agg")` at top of script**: Ensures the script works on headless
  servers (RunPod, CI) without a display. Must be called before any pyplot import.
- **Output to `visuals/` at project root**: Separate from `results/` (which is raw data).
  Visuals are derived artifacts.

## What NOT to Touch

- `scripts/run_experiment_0.py` — don't modify the experiment script
- `results/experiment_0/*.csv` — read-only, never write to results
- `src/` — this is a standalone visualization script, not a framework change
- `tests/` (the main test directory) — task tests go in the spec's tests/ dir

## Testing Approach

Tests use pytest with a temporary output directory. They verify:
- Each generator function creates the expected PNG file
- The HTML file is created and references existing images
- Missing data is handled without crashes
- The display name mapping covers all column patterns in the data

Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-028-visualization-pipeline/tests/`

Tests import individual functions from `scripts.generate_visuals` and call them with
the real CSVs from `results/experiment_0/`. This is intentional — these are integration
tests that verify the actual output, not mocked unit tests. The real data is small (50 rows)
and already committed.
