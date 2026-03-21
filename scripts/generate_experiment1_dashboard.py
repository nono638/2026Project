#!/usr/bin/env python3
"""Generate interactive Plotly dashboard for Experiment 1 results.

Experiment 1: Strategy x Model Size — 5 RAG strategies x 6 models.
Produces a single self-contained HTML file with 12 visualizations.

Pattern follows generate_experiment0_dashboard.py: build_experiment1_figures()
returns (title, figure) pairs for gallery reuse, generate_dashboard() writes
a full standalone HTML page.

Usage:
    python scripts/generate_experiment1_dashboard.py
    python scripts/generate_experiment1_dashboard.py --csv results/experiment_1/raw_scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IBM Design colorblind-safe palette
IBM_COLORS = [
    "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000",
    "#000000", "#AAAAAA",
]

# Model sizes in billions of parameters — used for x-axis ordering
MODEL_SIZES: dict[str, float] = {
    "qwen3:0.6b": 0.6,
    "gemma3:1b": 1.0,
    "qwen3:1.7b": 1.7,
    "gemma3:4b": 4.0,
    "qwen3:4b": 4.0,
    "qwen3:8b": 8.0,
}

# Canonical model order by parameter count
MODEL_ORDER = ["qwen3:0.6b", "gemma3:1b", "qwen3:1.7b", "gemma3:4b", "qwen3:4b", "qwen3:8b"]

STRATEGY_ORDER = ["adaptive", "corrective", "multi_query", "naive", "self_rag"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Check if a column exists and has at least one non-NaN value."""
    return col in df.columns and df[col].notna().any()


def _order_models(models: list[str]) -> list[str]:
    """Sort model names by parameter count, preserving unknown models at end."""
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    return known + unknown


def _order_strategies(strategies: list[str]) -> list[str]:
    """Sort strategies alphabetically (matching STRATEGY_ORDER)."""
    return sorted(strategies)


def _fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div (no full page wrapper).

    Args:
        fig: Plotly figure.

    Returns:
        HTML string with the chart div.
    """
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_summary_card(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Summary statistics card as a Plotly Table.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    config_means = df.groupby(["strategy", "model"])["quality"].mean()
    best_idx = config_means.idxmax()
    worst_idx = config_means.idxmin()

    stats = {
        "Metric": [
            "Total Configurations",
            "Total Rows",
            "Best Config (mean quality)",
            "Worst Config (mean quality)",
            "Overall Mean Quality",
        ],
        "Value": [
            str(len(config_means)),
            str(len(df)),
            f"{best_idx[0]} + {best_idx[1]} ({config_means[best_idx]:.3f})",
            f"{worst_idx[0]} + {worst_idx[1]} ({config_means[worst_idx]:.3f})",
            f"{df['quality'].mean():.3f}",
        ],
    }

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(stats.keys()), fill_color="#648FFF",
                    font=dict(color="white", size=14), align="left"),
        cells=dict(values=list(stats.values()), fill_color="white",
                   font=dict(size=13), align="left", height=30),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220)
    return ("Summary", fig)


def _chart_quality_heatmap(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality heatmap: x=model, y=strategy, z=mean quality.

    Uses Viridis colorscale for perceptual uniformity and accessibility.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    pivot = df.pivot_table(values="quality", index="strategy", columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    strategies = _order_strategies(list(pivot.index))
    pivot = pivot.reindex(index=strategies, columns=models)

    annotations = []
    for i, strat in enumerate(strategies):
        for j, model in enumerate(models):
            val = pivot.loc[strat, model] if not pd.isna(pivot.loc[strat, model]) else None
            if val is not None:
                annotations.append(dict(
                    x=model, y=strat, text=f"{val:.3f}",
                    showarrow=False, font=dict(size=12),
                ))

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=models, y=strategies,
        colorscale="Viridis", colorbar=dict(title="Quality"),
        text=[[f"{v:.3f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Mean Quality by Strategy and Model",
        xaxis_title="Model", yaxis_title="Strategy",
        height=400, margin=dict(l=120),
    )
    return ("Quality Heatmap", fig)


def _chart_latency_heatmap(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Latency heatmap: x=model, y=strategy, z=mean strategy_latency_ms.

    Uses Plasma colorscale to visually distinguish from quality heatmap.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        logger.warning("strategy_latency_ms column missing — skipping latency heatmap")
        return ("Latency Heatmap", go.Figure())

    pivot = df.pivot_table(values="strategy_latency_ms", index="strategy",
                           columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    strategies = _order_strategies(list(pivot.index))
    pivot = pivot.reindex(index=strategies, columns=models)

    # Convert to seconds for display
    display_vals = pivot.values / 1000.0

    fig = go.Figure(data=go.Heatmap(
        z=display_vals, x=models, y=strategies,
        colorscale="Plasma", colorbar=dict(title="Latency (s)"),
        text=[[f"{v:.1f}s" if not pd.isna(v) else "" for v in row] for row in display_vals],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Mean Strategy Latency by Strategy and Model",
        xaxis_title="Model", yaxis_title="Strategy",
        height=400, margin=dict(l=120),
    )
    return ("Latency Heatmap", fig)


def _chart_quality_vs_model_size(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality vs model size: one line per strategy with error bars.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    fig = go.Figure()
    strategies = _order_strategies(df["strategy"].unique().tolist())

    # Markers: different symbols to distinguish overlapping points (e.g. gemma3:4b and qwen3:4b)
    markers = ["circle", "square", "diamond", "cross", "triangle-up", "star", "hexagon"]

    for i, strat in enumerate(strategies):
        sdf = df[df["strategy"] == strat]
        # Map models to sizes and compute stats
        sdf = sdf.copy()
        sdf["model_size"] = sdf["model"].map(MODEL_SIZES)
        sdf = sdf.dropna(subset=["model_size"])
        stats = sdf.groupby(["model", "model_size"])["quality"].agg(["mean", "std"]).reset_index()
        stats = stats.sort_values("model_size")

        fig.add_trace(go.Scatter(
            x=stats["model_size"], y=stats["mean"],
            error_y=dict(type="data", array=stats["std"].fillna(0), visible=True),
            mode="lines+markers",
            name=strat,
            marker=dict(color=IBM_COLORS[i % len(IBM_COLORS)],
                        symbol=markers[i % len(markers)], size=9),
            line=dict(color=IBM_COLORS[i % len(IBM_COLORS)]),
            text=stats["model"],
            hovertemplate="<b>%{text}</b><br>Size: %{x}B<br>Quality: %{y:.3f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="Quality vs Model Size by Strategy",
        xaxis_title="Model Size (B params)", yaxis_title="Mean Quality",
        height=500, legend_title="Strategy",
    )
    return ("Quality vs Model Size", fig)


def _chart_latency_vs_model_size(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Latency vs model size: one line per strategy, log y-axis.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        return ("Latency vs Model Size", go.Figure())

    fig = go.Figure()
    strategies = _order_strategies(df["strategy"].unique().tolist())
    markers = ["circle", "square", "diamond", "cross", "triangle-up", "star", "hexagon"]

    for i, strat in enumerate(strategies):
        sdf = df[df["strategy"] == strat].copy()
        sdf["model_size"] = sdf["model"].map(MODEL_SIZES)
        sdf = sdf.dropna(subset=["model_size"])
        stats = sdf.groupby(["model", "model_size"])["strategy_latency_ms"].agg(["mean", "std"]).reset_index()
        stats = stats.sort_values("model_size")

        fig.add_trace(go.Scatter(
            x=stats["model_size"], y=stats["mean"],
            error_y=dict(type="data", array=stats["std"].fillna(0), visible=True),
            mode="lines+markers",
            name=strat,
            marker=dict(color=IBM_COLORS[i % len(IBM_COLORS)],
                        symbol=markers[i % len(markers)], size=9),
            line=dict(color=IBM_COLORS[i % len(IBM_COLORS)]),
            text=stats["model"],
            hovertemplate="<b>%{text}</b><br>Size: %{x}B<br>Latency: %{y:.0f}ms<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="Strategy Latency vs Model Size",
        xaxis_title="Model Size (B params)", yaxis_title="Mean Latency (ms)",
        yaxis_type="log",
        height=500, legend_title="Strategy",
    )
    return ("Latency vs Model Size", fig)


def _chart_strategy_beats_size(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Strategy beats size analysis: cases where strategy+small beats naive+large.

    This is the project's core research question — can a clever RAG strategy
    on a small model outperform naive RAG on a larger model?

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    config_means = df.groupby(["strategy", "model"])["quality"].mean().reset_index()
    config_means["model_size"] = config_means["model"].map(MODEL_SIZES)

    naive_configs = config_means[config_means["strategy"] == "naive"]
    non_naive = config_means[config_means["strategy"] != "naive"]

    results = []
    for _, row in non_naive.iterrows():
        # Find naive configs with larger models
        bigger_naive = naive_configs[naive_configs["model_size"] > row["model_size"]]
        for _, naive_row in bigger_naive.iterrows():
            if row["quality"] > naive_row["quality"]:
                results.append({
                    "strategy": row["strategy"],
                    "small_model": row["model"],
                    "large_naive_model": naive_row["model"],
                    "delta": row["quality"] - naive_row["quality"],
                })

    if not results:
        # No beats found — return empty figure with message
        fig = go.Figure()
        fig.add_annotation(text="No cases found where strategy+small beats naive+large",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        fig.update_layout(height=300)
        return ("Strategy Beats Size", fig)

    rdf = pd.DataFrame(results)
    beat_counts = rdf.groupby("strategy").agg(
        count=("delta", "size"),
        mean_delta=("delta", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    fig = go.Figure(data=go.Bar(
        x=beat_counts["strategy"], y=beat_counts["count"],
        marker_color=[IBM_COLORS[i % len(IBM_COLORS)] for i in range(len(beat_counts))],
        text=[f"+{d:.3f}" for d in beat_counts["mean_delta"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Beats: %{y} cases<br>Mean delta: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Strategy Beats Size: Cases Where Strategy+Small > Naive+Large",
        xaxis_title="Strategy", yaxis_title="Number of 'Beats' Cases",
        height=400,
    )
    return ("Strategy Beats Size", fig)


def _chart_per_metric_breakdown(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-metric breakdown for top-10 and bottom-5 configs.

    Shows faithfulness, relevance, conciseness separately to reveal
    which dimension drives quality differences.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    metrics = ["faithfulness", "relevance", "conciseness"]
    available = [m for m in metrics if _safe_col(df, m)]
    if not available:
        return ("Per-Metric Breakdown", go.Figure())

    config_means = df.groupby(["strategy", "model"])[["quality"] + available].mean().reset_index()
    config_means["config"] = config_means["strategy"] + " + " + config_means["model"]
    config_means = config_means.sort_values("quality", ascending=False)

    # Top 10 and bottom 5
    top = config_means.head(10)
    bottom = config_means.tail(5)
    selected = pd.concat([top, bottom]).drop_duplicates(subset=["config"])

    fig = go.Figure()
    for i, metric in enumerate(available):
        fig.add_trace(go.Bar(
            x=selected["config"], y=selected[metric],
            name=metric.capitalize(),
            marker_color=IBM_COLORS[i % len(IBM_COLORS)],
        ))

    fig.update_layout(
        barmode="group",
        title="Per-Metric Breakdown (Top 10 + Bottom 5 Configs)",
        xaxis_title="Configuration", yaxis_title="Score",
        xaxis_tickangle=-45, height=500,
        margin=dict(b=150),
    )
    return ("Per-Metric Breakdown", fig)


def _chart_score_distributions_by_strategy(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Violin plots of quality distribution per strategy.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    strategies = _order_strategies(df["strategy"].unique().tolist())
    fig = go.Figure()
    for i, strat in enumerate(strategies):
        fig.add_trace(go.Violin(
            y=df[df["strategy"] == strat]["quality"],
            name=strat, box_visible=True, meanline_visible=True,
            marker_color=IBM_COLORS[i % len(IBM_COLORS)],
        ))
    fig.update_layout(
        title="Score Distributions by Strategy",
        yaxis_title="Quality", height=450,
        showlegend=False,
    )
    return ("Score Distributions by Strategy", fig)


def _chart_score_distributions_by_model(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Violin plots of quality distribution per model, ordered by size.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    models = _order_models(df["model"].unique().tolist())
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Violin(
            y=df[df["model"] == model]["quality"],
            name=model, box_visible=True, meanline_visible=True,
            marker_color=IBM_COLORS[i % len(IBM_COLORS)],
        ))
    fig.update_layout(
        title="Score Distributions by Model",
        yaxis_title="Quality", height=450,
        showlegend=False,
    )
    return ("Score Distributions by Model", fig)


def _chart_gold_metrics_heatmap(df: pd.DataFrame) -> tuple[str, go.Figure] | None:
    """Gold F1 heatmap: x=model, y=strategy.

    Skipped if gold_f1 is missing or all NaN.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure) or None if gold_f1 unavailable.
    """
    if not _safe_col(df, "gold_f1"):
        logger.warning("gold_f1 column missing or all NaN — skipping gold metrics heatmap")
        return None

    pivot = df.pivot_table(values="gold_f1", index="strategy", columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    strategies = _order_strategies(list(pivot.index))
    pivot = pivot.reindex(index=strategies, columns=models)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=models, y=strategies,
        colorscale="Viridis", colorbar=dict(title="Gold F1"),
        text=[[f"{v:.3f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Gold F1 by Strategy and Model",
        xaxis_title="Model", yaxis_title="Strategy",
        height=400, margin=dict(l=120),
    )
    return ("Gold Metrics Heatmap", fig)


def _chart_pareto_frontier(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality vs latency scatter with Pareto frontier.

    Non-dominated configs (higher quality AND lower latency) are connected
    by a frontier line to show the efficiency boundary.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        return ("Quality vs Latency (Pareto)", go.Figure())

    config_stats = df.groupby(["strategy", "model"]).agg(
        quality=("quality", "mean"),
        latency=("strategy_latency_ms", "mean"),
    ).reset_index()
    config_stats["config"] = config_stats["strategy"] + " + " + config_stats["model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=config_stats["latency"], y=config_stats["quality"],
        mode="markers+text", text=config_stats["config"],
        textposition="top center", textfont=dict(size=8),
        marker=dict(size=10, color=IBM_COLORS[0]),
        hovertemplate="<b>%{text}</b><br>Latency: %{x:.0f}ms<br>Quality: %{y:.3f}<extra></extra>",
    ))

    # Compute and draw Pareto frontier
    # Pareto-optimal: no other config has both higher quality AND lower latency
    pareto = []
    for _, row in config_stats.iterrows():
        dominated = False
        for _, other in config_stats.iterrows():
            if other["quality"] >= row["quality"] and other["latency"] <= row["latency"]:
                if other["quality"] > row["quality"] or other["latency"] < row["latency"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(row)

    if pareto:
        pareto_df = pd.DataFrame(pareto).sort_values("latency")
        fig.add_trace(go.Scatter(
            x=pareto_df["latency"], y=pareto_df["quality"],
            mode="lines", name="Pareto Frontier",
            line=dict(color="#DC267F", dash="dash", width=2),
        ))

    fig.update_layout(
        title="Quality vs Latency — Pareto Frontier",
        xaxis_title="Mean Strategy Latency (ms)", yaxis_title="Mean Quality",
        xaxis_type="log", height=500,
    )
    return ("Quality vs Latency (Pareto)", fig)


def _chart_per_query_detail(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Interactive table showing worst-10 and best-10 individual answers.

    Helps identify where RAG breaks down at the per-query level.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    sorted_df = df.sort_values("quality")
    worst = sorted_df.head(10)
    best = sorted_df.tail(10)
    selected = pd.concat([worst, best])

    question_col = selected["question"].str[:50] if _safe_col(df, "question") else [""] * len(selected)
    gold_f1_col = selected["gold_f1"].round(3) if _safe_col(df, "gold_f1") else ["N/A"] * len(selected)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Strategy", "Model", "Question", "Quality", "Gold F1"],
            fill_color="#648FFF", font=dict(color="white", size=12), align="left",
        ),
        cells=dict(
            values=[
                selected["strategy"],
                selected["model"],
                question_col,
                selected["quality"].round(3),
                gold_f1_col,
            ],
            fill_color=[["#ffe0e0"] * 10 + ["#e0ffe0"] * 10],
            font=dict(size=11), align="left", height=25,
        ),
    )])
    fig.update_layout(
        title="Per-Query Detail: Worst 10 + Best 10",
        height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    return ("Per-Query Detail", fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_experiment1_figures(
    csv_path: Path | str,
) -> list[tuple[str, go.Figure]]:
    """Build all Experiment 1 chart figures from a raw scores CSV.

    Reads the CSV and generates interactive Plotly charts. Each chart is
    returned as a (title, figure) tuple for flexible embedding — the gallery
    uses these to build composite pages.

    Args:
        csv_path: Path to ``results/experiment_1/raw_scores.csv``.

    Returns:
        List of ``(title, figure)`` tuples. Empty list if CSV is empty.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.warning("Empty or unparseable CSV at %s — returning empty figures", csv_path)
        return []

    if df.empty or "quality" not in df.columns:
        logger.warning("Empty or invalid CSV at %s — returning empty figures", csv_path)
        return []

    if "strategy" not in df.columns or "model" not in df.columns:
        logger.warning("Missing strategy or model column — returning empty figures")
        return []

    figures: list[tuple[str, go.Figure]] = []

    # 1. Summary card
    figures.append(_chart_summary_card(df))
    # 2. Quality heatmap
    figures.append(_chart_quality_heatmap(df))
    # 3. Latency heatmap
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_latency_heatmap(df))
    # 4. Quality vs model size
    figures.append(_chart_quality_vs_model_size(df))
    # 5. Latency vs model size
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_latency_vs_model_size(df))
    # 6. Strategy beats size
    figures.append(_chart_strategy_beats_size(df))
    # 7. Per-metric breakdown
    figures.append(_chart_per_metric_breakdown(df))
    # 8. Score distributions by strategy
    figures.append(_chart_score_distributions_by_strategy(df))
    # 9. Score distributions by model
    figures.append(_chart_score_distributions_by_model(df))
    # 10. Gold metrics heatmap
    gold_result = _chart_gold_metrics_heatmap(df)
    if gold_result is not None:
        figures.append(gold_result)
    # 11. Pareto frontier
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_pareto_frontier(df))
    # 12. Per-query detail
    figures.append(_chart_per_query_detail(df))

    return figures


def generate_dashboard(
    csv_path: Path | str,
    output_path: Path | str,
) -> None:
    """Generate a self-contained HTML dashboard for Experiment 1.

    Args:
        csv_path: Path to raw_scores.csv.
        output_path: Where to write the HTML file.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures = build_experiment1_figures(csv_path)

    # Build standalone HTML page
    parts = [
        '<!DOCTYPE html><html lang="en"><head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '<title>Experiment 1: Strategy x Model Size — RAGBench</title>',
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        '<style>body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }'
        ' .chart { margin: 30px 0; }</style>',
        '</head><body>',
        '<h1>Experiment 1: Strategy x Model Size</h1>',
        '<p>5 RAG strategies x 6 models. Interactive charts — hover, click legend, drag to zoom.</p>',
    ]

    if not figures:
        parts.append('<p><em>No data available or CSV is empty.</em></p>')
    else:
        for title, fig in figures:
            chart_html = _fig_to_html(fig)
            parts.append(f'<div class="chart"><h2>{title}</h2>{chart_html}</div>')

    parts.append('</body></html>')

    output_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Experiment 1 dashboard")
    parser.add_argument("--csv", type=str,
                        default="results/experiment_1/raw_scores.csv",
                        help="Path to raw_scores.csv")
    parser.add_argument("--output", type=str,
                        default="visuals/experiment_1.html",
                        help="Output HTML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_dashboard(args.csv, args.output)
