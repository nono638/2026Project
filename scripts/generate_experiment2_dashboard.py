#!/usr/bin/env python3
"""Generate interactive Plotly dashboard for Experiment 2 results.

Experiment 2: Chunking x Model Size — 4 chunking strategies x 4 Qwen3 models.
Produces a single self-contained HTML file with 12 visualizations.

Pattern follows generate_experiment0_dashboard.py: build_experiment2_figures()
returns (title, figure) pairs for gallery reuse, generate_dashboard() writes
a full standalone HTML page.

Usage:
    python scripts/generate_experiment2_dashboard.py
    python scripts/generate_experiment2_dashboard.py --csv results/experiment_2/raw_scores.csv
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

# Model sizes in billions — Exp 2 uses only Qwen3
MODEL_SIZES: dict[str, float] = {
    # task-054: Qwen 3.5 small replaces Qwen3.
    "qwen3.5:0.8b": 0.8,
    "qwen3.5:2b": 2.0,
    "qwen3.5:4b": 4.0,
    "qwen3.5:9b": 9.0,
}

MODEL_ORDER = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]

CHUNKER_ORDER = ["fixed", "recursive", "semantic", "sentence"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Check if a column exists and has at least one non-NaN value."""
    return col in df.columns and df[col].notna().any()


# Bootstrap settings — duplicated from generate_experiment1_dashboard.py
# rather than imported because both modules are run-once render scripts
# and a shared utils module would only marginally reduce coupling. The
# fixed seed makes CIs stable across re-renders (CIs are an analysis
# artefact, not a measurement, so a moving seed would silently shift
# error bars between regenerations).
_BOOT_N = 2000
_BOOT_RNG = np.random.default_rng(20260520)


def _bootstrap_mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) 95% bootstrap CI on the mean.

    Percentile method. Returns ``(nan, nan, nan)`` for inputs with
    fewer than 2 finite values so callers can render a point marker
    instead of an error bar.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size == 1 else float("nan"),
                float("nan"), float("nan"))
    idx = _BOOT_RNG.integers(0, v.size, size=(_BOOT_N, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return (float(v.mean()),
            float(np.quantile(boot_means, lo)),
            float(np.quantile(boot_means, hi)))


def _ci_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Per-group mean + 95% bootstrap CI as a tidy DataFrame.

    Columns: group_col, mean, ci_lo, ci_hi, err_minus, err_plus, n.
    err_minus / err_plus are half-widths for direct use as Plotly
    ``error_y`` arrays.
    """
    rows: list[dict] = []
    for key, sub in df.groupby(group_col, dropna=True):
        mean, lo, hi = _bootstrap_mean_ci(sub[value_col].to_numpy())
        rows.append({
            group_col: key,
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "err_minus": (mean - lo) if np.isfinite(lo) else 0.0,
            "err_plus":  (hi - mean) if np.isfinite(hi) else 0.0,
            "n": int(sub[value_col].notna().sum()),
        })
    return pd.DataFrame(rows)


def _order_models(models: list[str]) -> list[str]:
    """Sort model names by parameter count."""
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    return known + unknown


def _order_chunkers(chunkers: list[str]) -> list[str]:
    """Sort chunkers alphabetically."""
    return sorted(chunkers)


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
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    config_means = df.groupby(["chunker", "model"])["consensus_quality"].mean()
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
            f"{df['consensus_quality'].mean():.3f}",
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


def _chart_chunker_ranking_ci(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-chunker mean quality with 95% bootstrap CI error bars.

    Headline visual for &ldquo;does chunker choice actually matter?&rdquo;
    Ordered by mean (descending) so the visual rank is immediate. If
    error bars overlap, the chunker means are not significantly
    different at the 95% level.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    stats = _ci_table(df, "chunker", "consensus_quality")
    stats = stats.sort_values("mean", ascending=False).reset_index(drop=True)
    colors = [IBM_COLORS[i % len(IBM_COLORS)] for i in range(len(stats))]
    fig = go.Figure(data=go.Bar(
        x=stats["chunker"], y=stats["mean"],
        error_y=dict(
            type="data",
            array=stats["err_plus"],
            arrayminus=stats["err_minus"],
            visible=True, thickness=1.6, width=8,
        ),
        marker_color=colors,
        text=[f"{m:.3f}" for m in stats["mean"]],
        textposition="outside",
        customdata=np.column_stack([stats["ci_lo"], stats["ci_hi"], stats["n"]]),
        hovertemplate=(
            "<b>%{x}</b><br>Mean quality: %{y:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
            "n = %{customdata[2]}<extra></extra>"
        ),
    ))
    ymin = max(0.0, stats["ci_lo"].min() - 0.15)
    ymax = min(5.0, stats["ci_hi"].max() + 0.20)
    fig.update_layout(
        title="Per-Chunker Ranking with 95% Bootstrap CI",
        xaxis_title="Chunker",
        yaxis_title="Mean Quality (1–5 scale)",
        yaxis=dict(range=[ymin, ymax]),
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return ("Per-Chunker Ranking (with CIs)", fig)


def _chart_model_ranking_ci(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-model mean quality with 95% bootstrap CI error bars.

    Models ordered by parameter count so the size-vs-quality trend
    reads left-to-right. Companion to ``_chart_chunker_ranking_ci``.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    stats = _ci_table(df, "model", "consensus_quality")
    ordered = _order_models(stats["model"].tolist())
    stats = stats.set_index("model").loc[ordered].reset_index()

    fig = go.Figure(data=go.Bar(
        x=stats["model"], y=stats["mean"],
        error_y=dict(
            type="data",
            array=stats["err_plus"],
            arrayminus=stats["err_minus"],
            visible=True, thickness=1.6, width=8,
        ),
        marker_color=IBM_COLORS[0],
        text=[f"{m:.3f}" for m in stats["mean"]],
        textposition="outside",
        customdata=np.column_stack([stats["ci_lo"], stats["ci_hi"], stats["n"]]),
        hovertemplate=(
            "<b>%{x}</b><br>Mean quality: %{y:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
            "n = %{customdata[2]}<extra></extra>"
        ),
    ))
    ymin = max(0.0, stats["ci_lo"].min() - 0.15)
    ymax = min(5.0, stats["ci_hi"].max() + 0.20)
    fig.update_layout(
        title="Per-Model Ranking with 95% Bootstrap CI (ordered by size)",
        xaxis_title="Model (ordered by parameter count)",
        yaxis_title="Mean Quality (1–5 scale)",
        yaxis=dict(range=[ymin, ymax]),
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return ("Per-Model Ranking (with CIs)", fig)


def _chart_quality_heatmap(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality heatmap: x=model, y=chunker, z=mean quality.

    Uses Viridis colorscale for perceptual uniformity and accessibility.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    pivot = df.pivot_table(values="consensus_quality", index="chunker", columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    chunkers = _order_chunkers(list(pivot.index))
    pivot = pivot.reindex(index=chunkers, columns=models)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=models, y=chunkers,
        colorscale="Viridis", colorbar=dict(title="Quality"),
        text=[[f"{v:.3f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Mean Quality by Chunker and Model",
        xaxis_title="Model", yaxis_title="Chunker",
        height=350, margin=dict(l=120),
    )
    return ("Quality Heatmap", fig)


def _chart_latency_heatmap(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Latency heatmap: x=model, y=chunker, z=mean strategy_latency_ms.

    Uses Plasma colorscale to visually distinguish from quality heatmap.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        logger.warning("strategy_latency_ms missing — skipping latency heatmap")
        return ("Latency Heatmap", go.Figure())

    pivot = df.pivot_table(values="strategy_latency_ms", index="chunker",
                           columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    chunkers = _order_chunkers(list(pivot.index))
    pivot = pivot.reindex(index=chunkers, columns=models)

    display_vals = pivot.values / 1000.0

    fig = go.Figure(data=go.Heatmap(
        z=display_vals, x=models, y=chunkers,
        colorscale="Plasma", colorbar=dict(title="Latency (s)"),
        text=[[f"{v:.1f}s" if not pd.isna(v) else "" for v in row] for row in display_vals],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Mean Strategy Latency by Chunker and Model",
        xaxis_title="Model", yaxis_title="Chunker",
        height=350, margin=dict(l=120),
    )
    return ("Latency Heatmap", fig)


def _chart_quality_vs_model_size(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality vs model size: one line per chunker with 95% bootstrap CI.

    Error bars are 95% percentile bootstrap CIs on the mean for each
    (chunker, model) cell &mdash; not per-row standard deviations. See
    the matching docstring in generate_experiment1_dashboard.py for the
    rationale (std-of-individuals overstates the uncertainty in a
    well-sampled mean; CI is what the reader is actually asking about).

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    fig = go.Figure()
    chunkers = _order_chunkers(df["chunker"].unique().tolist())
    markers = ["circle", "square", "diamond", "cross", "triangle-up", "star"]

    for i, chunker in enumerate(chunkers):
        cdf = df[df["chunker"] == chunker].copy()
        cdf["model_size"] = cdf["model"].map(MODEL_SIZES)
        cdf = cdf.dropna(subset=["model_size"])
        rows: list[dict] = []
        for (model, size), cell in cdf.groupby(["model", "model_size"]):
            mean, lo, hi = _bootstrap_mean_ci(cell["consensus_quality"].to_numpy())
            rows.append({
                "model": model, "model_size": size, "mean": mean,
                "err_minus": (mean - lo) if np.isfinite(lo) else 0.0,
                "err_plus":  (hi - mean) if np.isfinite(hi) else 0.0,
                "n": int(cell["consensus_quality"].notna().sum()),
            })
        stats = pd.DataFrame(rows).sort_values("model_size")

        fig.add_trace(go.Scatter(
            x=stats["model_size"], y=stats["mean"],
            error_y=dict(
                type="data",
                array=stats["err_plus"],
                arrayminus=stats["err_minus"],
                visible=True, thickness=1.4, width=4,
            ),
            mode="lines+markers", name=chunker,
            marker=dict(color=IBM_COLORS[i % len(IBM_COLORS)],
                        symbol=markers[i % len(markers)], size=9),
            line=dict(color=IBM_COLORS[i % len(IBM_COLORS)]),
            text=stats["model"],
            customdata=stats["n"],
            hovertemplate=(
                "<b>%{text}</b><br>Size: %{x}B<br>"
                "Mean quality: %{y:.3f}<br>n = %{customdata}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        title="Quality vs Model Size by Chunker (error bars = 95% bootstrap CI on the mean)",
        xaxis_title="Model Size (B params)", yaxis_title="Mean Quality",
        height=500, legend_title="Chunker",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return ("Quality vs Model Size", fig)


def _chart_latency_vs_model_size(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Latency vs model size: one line per chunker, log y-axis.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        return ("Latency vs Model Size", go.Figure())

    fig = go.Figure()
    chunkers = _order_chunkers(df["chunker"].unique().tolist())
    markers = ["circle", "square", "diamond", "cross", "triangle-up", "star"]

    for i, chunker in enumerate(chunkers):
        cdf = df[df["chunker"] == chunker].copy()
        cdf["model_size"] = cdf["model"].map(MODEL_SIZES)
        cdf = cdf.dropna(subset=["model_size"])
        stats = cdf.groupby(["model", "model_size"])["strategy_latency_ms"].agg(["mean", "std"]).reset_index()
        stats = stats.sort_values("model_size")

        fig.add_trace(go.Scatter(
            x=stats["model_size"], y=stats["mean"],
            error_y=dict(type="data", array=stats["std"].fillna(0), visible=True),
            mode="lines+markers", name=chunker,
            marker=dict(color=IBM_COLORS[i % len(IBM_COLORS)],
                        symbol=markers[i % len(markers)], size=9),
            line=dict(color=IBM_COLORS[i % len(IBM_COLORS)]),
            text=stats["model"],
            hovertemplate="<b>%{text}</b><br>Size: %{x}B<br>Latency: %{y:.0f}ms<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="Strategy Latency vs Model Size by Chunker",
        xaxis_title="Model Size (B params)", yaxis_title="Mean Latency (ms)",
        yaxis_type="log", height=500, legend_title="Chunker",
    )
    return ("Latency vs Model Size", fig)


def _chart_chunking_impact(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Chunking impact analysis: quality delta between best and worst chunker per model.

    Research question: does chunking matter more for small or large models?

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    config_means = df.groupby(["chunker", "model"])["consensus_quality"].mean().reset_index()
    config_means["model_size"] = config_means["model"].map(MODEL_SIZES)

    models = _order_models(config_means["model"].unique().tolist())
    deltas = []
    best_chunkers = []
    worst_chunkers = []

    for model in models:
        mdf = config_means[config_means["model"] == model]
        if len(mdf) < 2:
            deltas.append(0)
            best_chunkers.append("N/A")
            worst_chunkers.append("N/A")
            continue
        best_row = mdf.loc[mdf["consensus_quality"].idxmax()]
        worst_row = mdf.loc[mdf["consensus_quality"].idxmin()]
        deltas.append(best_row["consensus_quality"] - worst_row["consensus_quality"])
        best_chunkers.append(best_row["chunker"])
        worst_chunkers.append(worst_row["chunker"])

    fig = go.Figure(data=go.Bar(
        x=models, y=deltas,
        marker_color=[IBM_COLORS[i % len(IBM_COLORS)] for i in range(len(models))],
        text=[f"Best: {b}<br>Worst: {w}" for b, w in zip(best_chunkers, worst_chunkers)],
        hovertemplate="<b>%{x}</b><br>Quality Delta: %{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Chunking Impact: Quality Delta (Best − Worst Chunker) per Model",
        xaxis_title="Model", yaxis_title="Quality Delta",
        height=400,
    )
    return ("Chunking Impact Analysis", fig)


def _chart_per_metric_breakdown(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-metric breakdown for all configs (16 is manageable).

    Shows faithfulness, relevance, conciseness separately. Per-judge
    columns (e.g. ``anthropic_claude_haiku_4_5_20251001_faithfulness``,
    ``openai_gpt_5_4_mini_faithfulness``) are averaged into one
    per-metric series before grouping, matching the multi-judge schema
    rolled out in task-052. Before the fix the chart rendered empty
    because it looked for bare metric columns that no longer exist.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    metrics = ["faithfulness", "relevance", "conciseness"]
    df = df.copy()
    available: list[str] = []
    for m in metrics:
        cols = [
            c for c in df.columns
            if c.endswith(f"_{m}")
            and c not in (f"answer_{m}", f"consensus_{m}")
        ]
        if not cols:
            if _safe_col(df, m):
                df[f"_avg_{m}"] = df[m]
                available.append(m)
            continue
        df[f"_avg_{m}"] = df[cols].mean(axis=1, skipna=True)
        available.append(m)

    if not available:
        fig = go.Figure()
        fig.add_annotation(
            text=("No per-metric judge columns found in the CSV — "
                  "this chart needs faithfulness / relevance / "
                  "conciseness per judge."),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        return ("Per-Metric Breakdown", fig)

    avg_cols = [f"_avg_{m}" for m in available]
    config_means = (
        df.groupby(["chunker", "model"])[["consensus_quality"] + avg_cols]
        .mean()
        .reset_index()
    )
    config_means["config"] = config_means["chunker"] + " + " + config_means["model"]
    config_means = config_means.sort_values("consensus_quality", ascending=False)

    selected_keys = list(zip(config_means["chunker"], config_means["model"]))
    config_list = config_means["config"].tolist()

    fig = go.Figure()
    for i, metric in enumerate(available):
        means: list[float] = []
        err_minus: list[float] = []
        err_plus: list[float] = []
        for chunker, mdl in selected_keys:
            cell = df[(df["chunker"] == chunker) & (df["model"] == mdl)]
            mean, lo, hi = _bootstrap_mean_ci(cell[f"_avg_{metric}"].to_numpy())
            means.append(mean)
            err_minus.append((mean - lo) if np.isfinite(lo) else 0.0)
            err_plus.append((hi - mean) if np.isfinite(hi) else 0.0)
        fig.add_trace(go.Bar(
            x=config_list, y=means,
            error_y=dict(
                type="data",
                array=err_plus, arrayminus=err_minus,
                visible=True, thickness=1.2, width=3,
            ),
            name=metric.capitalize(),
            marker_color=IBM_COLORS[i % len(IBM_COLORS)],
        ))

    fig.update_layout(
        barmode="group",
        title="Per-Metric Breakdown — all configs (error bars = 95% CI on judge-averaged score)",
        xaxis_title="Configuration",
        yaxis_title="Score (1–5)",
        xaxis_tickangle=-45, height=520,
        margin=dict(b=150),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return ("Per-Metric Breakdown", fig)


def _chart_score_distributions_by_chunker(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Violin plots of quality distribution per chunker.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    chunkers = _order_chunkers(df["chunker"].unique().tolist())
    fig = go.Figure()
    for i, chunker in enumerate(chunkers):
        fig.add_trace(go.Violin(
            y=df[df["chunker"] == chunker]["consensus_quality"],
            name=chunker, box_visible=True, meanline_visible=True,
            marker_color=IBM_COLORS[i % len(IBM_COLORS)],
        ))
    fig.update_layout(
        title="Score Distributions by Chunker",
        yaxis_title="Quality", height=450,
        showlegend=False,
    )
    return ("Score Distributions by Chunker", fig)


def _chart_score_distributions_by_model(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Violin plots of quality distribution per model, ordered by size.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    models = _order_models(df["model"].unique().tolist())
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Violin(
            y=df[df["model"] == model]["consensus_quality"],
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
    """Gold F1 heatmap: x=model, y=chunker.

    Skipped if gold_f1 is missing or all NaN.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure) or None if gold_f1 unavailable.
    """
    if not _safe_col(df, "gold_f1"):
        logger.warning("gold_f1 missing or all NaN — skipping gold metrics heatmap")
        return None

    pivot = df.pivot_table(values="gold_f1", index="chunker", columns="model", aggfunc="mean")
    models = _order_models([c for c in pivot.columns])
    chunkers = _order_chunkers(list(pivot.index))
    pivot = pivot.reindex(index=chunkers, columns=models)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=models, y=chunkers,
        colorscale="Viridis", colorbar=dict(title="Gold F1"),
        text=[[f"{v:.3f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Gold F1 by Chunker and Model",
        xaxis_title="Model", yaxis_title="Chunker",
        height=350, margin=dict(l=120),
    )
    return ("Gold Metrics Heatmap", fig)


def _chart_pareto_frontier(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Quality vs latency scatter with Pareto frontier.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "strategy_latency_ms"):
        return ("Quality vs Latency (Pareto)", go.Figure())

    config_stats = df.groupby(["chunker", "model"]).agg(
        consensus_quality=("consensus_quality", "mean"),
        latency=("strategy_latency_ms", "mean"),
    ).reset_index()
    config_stats["config"] = config_stats["chunker"] + " + " + config_stats["model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=config_stats["latency"], y=config_stats["consensus_quality"],
        mode="markers+text", text=config_stats["config"],
        textposition="top center", textfont=dict(size=8),
        marker=dict(size=10, color=IBM_COLORS[0]),
        hovertemplate="<b>%{text}</b><br>Latency: %{x:.0f}ms<br>Quality: %{y:.3f}<extra></extra>",
    ))

    # Pareto frontier: non-dominated configs (higher quality AND lower latency)
    pareto = []
    for _, row in config_stats.iterrows():
        dominated = False
        for _, other in config_stats.iterrows():
            if other["consensus_quality"] >= row["consensus_quality"] and other["latency"] <= row["latency"]:
                if other["consensus_quality"] > row["consensus_quality"] or other["latency"] < row["latency"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(row)

    if pareto:
        pareto_df = pd.DataFrame(pareto).sort_values("latency")
        fig.add_trace(go.Scatter(
            x=pareto_df["latency"], y=pareto_df["consensus_quality"],
            mode="lines", name="Pareto Frontier",
            line=dict(color="#DC267F", dash="dash", width=2),
        ))

    fig.update_layout(
        title="Quality vs Latency — Pareto Frontier",
        xaxis_title="Mean Strategy Latency (ms)", yaxis_title="Mean Quality",
        xaxis_type="log", height=500,
    )
    return ("Quality vs Latency (Pareto)", fig)


def _chart_chunk_count_analysis(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Chunk count analysis: scatter of mean num_chunks vs mean quality per config.

    Shows whether more chunks help or hurt quality.

    Args:
        df: Experiment 2 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    if not _safe_col(df, "num_chunks"):
        logger.warning("num_chunks column missing — skipping chunk count analysis")
        return ("Chunk Count Analysis", go.Figure())

    config_stats = df.groupby(["chunker", "model"]).agg(
        consensus_quality=("consensus_quality", "mean"),
        num_chunks=("num_chunks", "mean"),
    ).reset_index()
    config_stats["config"] = config_stats["chunker"] + " + " + config_stats["model"]

    # Color by chunker for visual grouping
    chunkers = _order_chunkers(config_stats["chunker"].unique().tolist())
    color_map = {c: IBM_COLORS[i % len(IBM_COLORS)] for i, c in enumerate(chunkers)}

    fig = go.Figure()
    for chunker in chunkers:
        cdf = config_stats[config_stats["chunker"] == chunker]
        fig.add_trace(go.Scatter(
            x=cdf["num_chunks"], y=cdf["consensus_quality"],
            mode="markers+text", text=cdf["model"],
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=12, color=color_map[chunker]),
            name=chunker,
            hovertemplate="<b>%{text}</b><br>Chunks: %{x:.1f}<br>Quality: %{y:.3f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="Chunk Count vs Quality",
        xaxis_title="Mean Number of Chunks", yaxis_title="Mean Quality",
        height=500, legend_title="Chunker",
    )
    return ("Chunk Count Analysis", fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_experiment2_figures(
    csv_path: Path | str,
) -> list[tuple[str, go.Figure]]:
    """Build all Experiment 2 chart figures from a raw scores CSV.

    Reads the CSV and generates interactive Plotly charts. Each chart is
    returned as a (title, figure) tuple for flexible embedding — the gallery
    uses these to build composite pages.

    Args:
        csv_path: Path to ``results/experiment_2/raw_scores.csv``.

    Returns:
        List of ``(title, figure)`` tuples. Empty list if CSV is empty.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.warning("Empty or unparseable CSV at %s — returning empty figures", csv_path)
        return []

    if df.empty or "consensus_quality" not in df.columns:
        logger.warning("Empty or invalid CSV at %s — returning empty figures", csv_path)
        return []

    if "chunker" not in df.columns or "model" not in df.columns:
        logger.warning("Missing chunker or model column — returning empty figures")
        return []

    figures: list[tuple[str, go.Figure]] = []

    # 1. Summary card
    figures.append(_chart_summary_card(df))
    # 2. CI-ranked headline charts — answer "is the chunker effect real?"
    # and "does model size matter here?" before any heatmaps or
    # drill-downs. Error-bar overlap = means not distinguishable at 95%.
    figures.append(_chart_chunker_ranking_ci(df))
    figures.append(_chart_model_ranking_ci(df))
    # 3. Quality heatmap
    figures.append(_chart_quality_heatmap(df))
    # 3. Latency heatmap
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_latency_heatmap(df))
    # 4. Quality vs model size
    figures.append(_chart_quality_vs_model_size(df))
    # 5. Latency vs model size
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_latency_vs_model_size(df))
    # 6. Chunking impact analysis
    figures.append(_chart_chunking_impact(df))
    # 7. Per-metric breakdown
    figures.append(_chart_per_metric_breakdown(df))
    # 8. Score distributions by chunker
    figures.append(_chart_score_distributions_by_chunker(df))
    # 9. Score distributions by model
    figures.append(_chart_score_distributions_by_model(df))
    # 10. Gold metrics heatmap
    gold_result = _chart_gold_metrics_heatmap(df)
    if gold_result is not None:
        figures.append(gold_result)
    # 11. Pareto frontier
    if _safe_col(df, "strategy_latency_ms"):
        figures.append(_chart_pareto_frontier(df))
    # 12. Chunk count analysis
    figures.append(_chart_chunk_count_analysis(df))

    return figures


def generate_dashboard(
    csv_path: Path | str,
    output_path: Path | str,
) -> None:
    """Generate a self-contained HTML dashboard for Experiment 2.

    Args:
        csv_path: Path to raw_scores.csv.
        output_path: Where to write the HTML file.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures = build_experiment2_figures(csv_path)

    parts = [
        '<!DOCTYPE html><html lang="en"><head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '<title>Experiment 2: Chunking x Model Size — RAGBench</title>',
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        '<style>body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }'
        ' .chart { margin: 30px 0; }</style>',
        '</head><body>',
        '<h1>Experiment 2: Chunking x Model Size</h1>',
        '<p>4 chunking strategies x 4 Qwen3 models. Interactive charts — hover, click legend, drag to zoom.</p>',
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
    parser = argparse.ArgumentParser(description="Generate Experiment 2 dashboard")
    parser.add_argument("--csv", type=str,
                        default="results/experiment_2/raw_scores.csv",
                        help="Path to raw_scores.csv")
    parser.add_argument("--output", type=str,
                        default="visuals/experiment_2.html",
                        help="Output HTML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_dashboard(args.csv, args.output)
