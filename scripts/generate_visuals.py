"""Automated visualization pipeline for experiment results.

Reads CSVs from results/experiment_N/ and generates PNGs + HTML gallery.

Usage:
    python scripts/generate_visuals.py                 # all experiments
    python scripts/generate_visuals.py --experiment 0  # one experiment
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on headless servers (RunPod, CI)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
# See: https://davidmathlogic.com/colorblind/
JUDGE_COLORS = {
    "Flash-Lite": "#648FFF",
    "Flash": "#785EF0",
    "Gemini Pro": "#DC267F",
    "Haiku": "#FE6100",
    "Sonnet": "#FFB000",
}

# Minimum number of non-null scores for a judge to be included in visuals.
# Fewer than 10 data points produce unreliable Pearson r correlations.
MIN_VALID_SCORES = 10


def _get_valid_judges(df: pd.DataFrame) -> dict[str, str]:
    """Find judges with enough valid quality scores.

    Args:
        df: DataFrame with *_quality columns.

    Returns:
        Dict mapping display name to quality column name, for judges
        with >= MIN_VALID_SCORES non-null values.
    """
    valid: dict[str, str] = {}
    for prefix, display_name in JUDGE_DISPLAY_NAMES.items():
        col = f"{prefix}_quality"
        if col in df.columns and df[col].notna().sum() >= MIN_VALID_SCORES:
            valid[display_name] = col
        elif col in df.columns:
            logger.warning("Skipping %s: only %d valid scores (need %d)",
                           display_name, df[col].notna().sum(), MIN_VALID_SCORES)
    return valid


def generate_explainer(
    answers_df: Optional[pd.DataFrame],
    scores_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate the RAG pipeline explainer diagram.

    Shows one concrete example row flowing through the pipeline: question,
    retrieved context, LLM, generated answer, then evaluation scores.
    Picks the row whose gold_bertscore is closest to median for a realistic
    (not cherry-picked) example.

    Args:
        answers_df: DataFrame from raw_answers.csv (with doc_text). If None,
                    the explainer is skipped (raw_answers.csv unavailable).
        scores_df: DataFrame from raw_scores.csv.
        out_dir: Directory to write explainer_rag_pipeline.png into.
    """
    if answers_df is None:
        logger.warning("Skipping explainer: no answers data (raw_answers.csv missing)")
        return

    # Join answers with scores to get both doc_text and evaluation metrics
    merged = answers_df.merge(scores_df, on="example_id", suffixes=("", "_score"))

    # Pick the row whose gold_bertscore is closest to median — avoids cherry-picking
    if "gold_bertscore" not in merged.columns or merged["gold_bertscore"].isna().all():
        logger.warning("Skipping explainer: no gold_bertscore data")
        return

    median_bs = merged["gold_bertscore"].median()
    idx = (merged["gold_bertscore"] - median_bs).abs().idxmin()
    row = merged.loc[idx]

    # Extract fields for the diagram
    question = textwrap.fill(str(row.get("question", "")), 65)
    doc_text_raw = str(row.get("doc_text", ""))
    # Truncate to 300 chars — the point is to show context exists, not display it all
    doc_text = textwrap.fill(doc_text_raw[:300] + ("..." if len(doc_text_raw) > 300 else ""), 65)
    rag_answer = textwrap.fill(str(row.get("rag_answer", "")), 65)
    gold_answer = str(row.get("gold_answer", "N/A"))
    bertscore = row.get("gold_bertscore", float("nan"))

    # Find top 2 judges with valid scores for this row
    judge_lines = []
    for prefix, display_name in JUDGE_DISPLAY_NAMES.items():
        col = f"{prefix}_quality"
        if col in row.index and pd.notna(row[col]):
            judge_lines.append(f"{display_name}: {row[col]:.1f} / 5.0")
        if len(judge_lines) >= 2:
            break

    # Build the figure — vertical flow diagram using FancyBboxPatch
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        # Title
        ax.text(5, 11.6, "RAG Pipeline: One Example", fontsize=16,
                ha="center", va="top", fontweight="bold")

        # Box specs: (label, text, y_top, height, bg_color)
        boxes = [
            ("QUESTION", question, 10.8, None, "#E3F2FD"),
            ("RETRIEVED CONTEXT (top chunks)", doc_text, None, None, "#E8F5E9"),
            ("LLM: Qwen3 4B (via Ollama)", "", None, 0.7, "#F5F5F5"),
            ("GENERATED ANSWER", rag_answer, None, None, "#FFF8E1"),
        ]

        # Evaluation box content
        eval_lines = [f"Gold answer: {gold_answer}"]
        if not np.isnan(bertscore):
            eval_lines.append(f"BERTScore:   {bertscore:.3f}")
        eval_lines.extend(judge_lines)
        eval_text = "\n".join(eval_lines)

        y_cursor = 10.8
        box_positions = []  # Track (y_center) for arrows

        for label, content, y_top, fixed_height, color in boxes:
            if y_top is not None:
                y_cursor = y_top

            # Estimate height from content lines
            lines = content.count("\n") + 1 if content else 0
            height = fixed_height if fixed_height else max(0.7, 0.35 * (lines + 1) + 0.3)
            y_bottom = y_cursor - height

            # Draw box
            patch = FancyBboxPatch(
                (0.5, y_bottom), 9, height,
                boxstyle="round,pad=0.2",
                facecolor=color, edgecolor="#333333", linewidth=1.5,
            )
            ax.add_patch(patch)

            # Label
            ax.text(0.8, y_cursor - 0.15, label, fontsize=9, fontweight="bold",
                    va="top", color="#333333")

            # Content text
            if content:
                ax.text(0.8, y_cursor - 0.45, content, fontsize=8,
                        va="top", family="monospace", color="#444444",
                        linespacing=1.3)

            box_positions.append((y_cursor, y_bottom))
            y_cursor = y_bottom - 0.3  # Gap between boxes

        # Evaluation box (special — has multiple lines)
        eval_height = max(1.2, 0.35 * (eval_text.count("\n") + 2) + 0.3)
        eval_y_bottom = y_cursor - eval_height
        patch = FancyBboxPatch(
            (0.5, eval_y_bottom), 9, eval_height,
            boxstyle="round,pad=0.2",
            facecolor="#FFF3E0", edgecolor="#333333", linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(0.8, y_cursor - 0.15, "EVALUATION", fontsize=9,
                fontweight="bold", va="top", color="#333333")
        ax.text(0.8, y_cursor - 0.45, eval_text, fontsize=8,
                va="top", family="monospace", color="#444444", linespacing=1.3)
        box_positions.append((y_cursor, eval_y_bottom))

        # Draw arrows between consecutive boxes
        for i in range(len(box_positions) - 1):
            _, y_bot = box_positions[i]
            y_top_next, _ = box_positions[i + 1]
            ax.annotate(
                "", xy=(5, y_top_next + 0.05), xytext=(5, y_bot - 0.05),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "explainer_rag_pipeline.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Generated explainer_rag_pipeline.png")


def generate_exp0_heatmap(df: pd.DataFrame, exp0_dir: Path) -> None:
    """Generate inter-judge correlation heatmap (Pearson r).

    Needs at least 2 judges with >= MIN_VALID_SCORES to produce a meaningful
    correlation matrix. Skips silently if insufficient data.

    Args:
        df: DataFrame from raw_scores.csv.
        exp0_dir: Directory for experiment_0 visuals.
    """
    valid = _get_valid_judges(df)
    if len(valid) < 2:
        logger.warning("Skipping heatmap: fewer than 2 judges with enough data")
        return

    # Build correlation matrix from quality columns
    sub = df[list(valid.values())].rename(
        columns={col: name for name, col in valid.items()}
    )
    corr = sub.corr()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Diverging colormap centered at 0 (red=negative, green=positive)
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)

        # Annotate each cell with the r value
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=LABEL_SIZE)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, label="Pearson r")
        ax.set_title("Inter-Judge Correlation (Pearson r)", fontsize=TITLE_SIZE)

        exp0_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(exp0_dir / "judge_correlation_heatmap.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Generated judge_correlation_heatmap.png")


def generate_exp0_scatter(df: pd.DataFrame, exp0_dir: Path) -> None:
    """Generate judge-vs-gold scatter plots (one subplot per judge).

    Shows which judges actually track semantic correctness (BERTScore).
    Falls back to gold_f1 if gold_bertscore is missing.

    Args:
        df: DataFrame from raw_scores.csv.
        exp0_dir: Directory for experiment_0 visuals.
    """
    # Determine the gold reference column — prefer BERTScore, fall back to F1
    if "gold_bertscore" in df.columns and df["gold_bertscore"].notna().any():
        gold_col = "gold_bertscore"
        gold_label = "BERTScore"
    elif "gold_f1" in df.columns and df["gold_f1"].notna().any():
        gold_col = "gold_f1"
        gold_label = "Gold F1"
    else:
        logger.warning("Skipping scatter: no gold_bertscore or gold_f1 column")
        return

    valid = _get_valid_judges(df)
    if not valid:
        logger.warning("Skipping scatter: no judges with enough data")
        return

    n_judges = len(valid)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n_judges, figsize=(3 * n_judges, 4), sharey=True)
        if n_judges == 1:
            axes = [axes]

        for ax, (name, col) in zip(axes, valid.items()):
            pair = df[[col, gold_col]].dropna()
            color = JUDGE_COLORS.get(name, "#648FFF")
            ax.scatter(pair[gold_col], pair[col], alpha=0.5, color=color, s=40)
            r = pair[col].corr(pair[gold_col])
            ax.set_title(f"{name}\nr = {r:.3f}", fontsize=LABEL_SIZE)
            ax.set_xlabel(gold_label)
            if ax is axes[0]:
                ax.set_ylabel("Judge Quality Score")
            ax.set_ylim(0.5, 5.5)

        fig.suptitle("Judge Quality vs Gold Reference", fontsize=TITLE_SIZE, y=1.02)
        exp0_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(exp0_dir / "judge_vs_gold_scatter.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Generated judge_vs_gold_scatter.png")


def generate_exp0_distributions(df: pd.DataFrame, exp0_dir: Path) -> None:
    """Generate horizontal box plots of judge score distributions.

    Reveals whether a judge is discriminating (spread across 1-5) or
    rubber-stamping (everything is 5).

    Args:
        df: DataFrame from raw_scores.csv.
        exp0_dir: Directory for experiment_0 visuals.
    """
    valid = _get_valid_judges(df)
    if not valid:
        logger.warning("Skipping distributions: no judges with enough data")
        return

    # Sort by median score for ordered display
    medians = {name: df[col].dropna().median() for name, col in valid.items()}
    sorted_judges = sorted(valid.items(), key=lambda x: medians[x[0]])

    display_names = [name for name, _ in sorted_judges]
    data = [df[col].dropna().values for _, col in sorted_judges]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(3, len(valid) * 0.8 + 1)))
        bp = ax.boxplot(data, vert=False, labels=display_names, patch_artist=True)

        for patch, name in zip(bp["boxes"], display_names):
            color = JUDGE_COLORS.get(name, "#648FFF")
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Quality Score (1-5)")
        ax.set_title("Judge Score Distributions", fontsize=TITLE_SIZE)

        exp0_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(exp0_dir / "score_distributions.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Generated score_distributions.png")


def generate_exp0_bertscore_hist(df: pd.DataFrame, exp0_dir: Path) -> None:
    """Generate BERTScore distribution histogram.

    Shows the baseline quality of RAG answers before judging — tells the
    reader whether the system is producing mostly-good or mostly-bad answers.

    Args:
        df: DataFrame from raw_scores.csv.
        exp0_dir: Directory for experiment_0 visuals.
    """
    if "gold_bertscore" not in df.columns or df["gold_bertscore"].isna().all():
        logger.warning("Skipping BERTScore histogram: no gold_bertscore column")
        return

    scores = df["gold_bertscore"].dropna()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=20, color="#648FFF", alpha=0.7, edgecolor="white")
        ax.axvline(scores.median(), color="#333333", linestyle="--",
                   label=f"Median: {scores.median():.3f}")
        # Annotate with summary stats
        ax.text(0.98, 0.95,
                f"n = {len(scores)}\nmean = {scores.mean():.3f}\nmedian = {scores.median():.3f}",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("BERTScore F1")
        ax.set_ylabel("Count")
        ax.set_title("RAG Answer Quality (BERTScore F1 vs Gold)", fontsize=TITLE_SIZE)
        ax.legend()

        exp0_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(exp0_dir / "bertscore_distribution.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Generated bertscore_distribution.png")


def generate_html(visuals_dir: Path) -> None:
    """Generate a self-contained HTML gallery embedding all existing PNGs.

    Scans visuals_dir for PNGs and builds an index page with section headers
    and captions. Missing images get a placeholder message instead of broken
    links. Uses inline CSS — no framework, works offline.

    Args:
        visuals_dir: Root directory for visual outputs (e.g., visuals/).
    """
    # Define the expected images with captions
    explainer = visuals_dir / "explainer_rag_pipeline.png"
    exp0_images = [
        ("judge_correlation_heatmap.png",
         "Pearson r between each pair of LLM judges on quality scores."),
        ("judge_vs_gold_scatter.png",
         "Each judge's quality score vs BERTScore (semantic similarity to gold answer)."),
        ("score_distributions.png",
         "Distribution of quality scores per judge. Wider spread = more discriminating."),
        ("bertscore_distribution.png",
         "Distribution of BERTScore F1 across all 50 examples."),
    ]

    # Build HTML sections
    explainer_section = ""
    if explainer.exists():
        explainer_section = (
            '    <img src="explainer_rag_pipeline.png" '
            'alt="RAG Pipeline Explainer">\n'
            '    <p class="caption">One example from HotpotQA showing each '
            "stage of the pipeline.</p>"
        )
    else:
        explainer_section = '    <p class="missing">Not yet generated.</p>'

    exp0_section_parts = []
    exp0_dir = visuals_dir / "experiment_0"
    for filename, caption in exp0_images:
        if (exp0_dir / filename).exists():
            exp0_section_parts.append(
                f'    <img src="experiment_0/{filename}" alt="{filename}">\n'
                f'    <p class="caption">{caption}</p>'
            )
        else:
            exp0_section_parts.append(
                f'    <p class="missing">{filename}: Not yet generated.</p>'
            )
    exp0_html = "\n".join(exp0_section_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGBench — Experiment Results</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px;
               margin: 0 auto; padding: 2rem; background: #fafafa; color: #222; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        h2 {{ margin-top: 2.5rem; color: #444; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd;
              border-radius: 4px; margin: 1rem 0; }}
        .caption {{ color: #666; font-size: 0.9rem; margin-top: -0.5rem; }}
        .missing {{ color: #999; font-style: italic; }}
    </style>
</head>
<body>
    <h1>RAGBench — Experiment Results</h1>
    <p>Auto-generated visualizations. Re-run <code>python scripts/generate_visuals.py</code>
       to regenerate.</p>

    <h2>How RAG Works</h2>
{explainer_section}

    <h2>Experiment 0: Scorer Validation</h2>
{exp0_html}
</body>
</html>
"""
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (visuals_dir / "index.html").write_text(html, encoding="utf-8")
    logger.info("Generated index.html")


def run_experiment_0(results_dir: Path, visuals_dir: Path) -> None:
    """Generate all Experiment 0 visuals.

    Args:
        results_dir: Path to results/ directory.
        visuals_dir: Path to visuals/ output directory.
    """
    exp0_data = results_dir / "experiment_0"
    if not exp0_data.exists():
        logger.warning("No data for experiment 0, skipping")
        return

    scores_path = exp0_data / "raw_scores.csv"
    answers_path = exp0_data / "raw_answers.csv"

    if not scores_path.exists():
        logger.warning("No raw_scores.csv for experiment 0, skipping")
        return

    scores_df = pd.read_csv(scores_path)

    answers_df = None
    if answers_path.exists():
        answers_df = pd.read_csv(answers_path)

    exp0_out = visuals_dir / "experiment_0"

    # Generate each visual independently — one failure shouldn't block others
    generate_explainer(answers_df, scores_df, visuals_dir)
    generate_exp0_heatmap(scores_df, exp0_out)
    generate_exp0_scatter(scores_df, exp0_out)
    generate_exp0_distributions(scores_df, exp0_out)
    generate_exp0_bertscore_hist(scores_df, exp0_out)


def main() -> None:
    """Entry point: parse args and generate visuals for requested experiments."""
    parser = argparse.ArgumentParser(
        description="Generate experiment visualizations and HTML gallery.",
    )
    parser.add_argument(
        "--experiment", type=int, default=None,
        help="Generate visuals for one experiment only (e.g., --experiment 0). "
             "Without this flag, all experiments are processed.",
    )
    args = parser.parse_args()

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment is not None:
        # Single experiment mode
        if args.experiment == 0:
            run_experiment_0(RESULTS_DIR, VISUALS_DIR)
        else:
            logger.warning("Experiment %d not yet supported", args.experiment)
    else:
        # All experiments mode — run each that has data
        run_experiment_0(RESULTS_DIR, VISUALS_DIR)
        # Future: run_experiment_1, run_experiment_2, cross-experiment comparisons

    # Always regenerate the HTML gallery last (references generated PNGs)
    generate_html(VISUALS_DIR)

    logger.info("Done. Visuals written to %s", VISUALS_DIR)


if __name__ == "__main__":
    main()
