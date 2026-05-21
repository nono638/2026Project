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

# Model sizes in billions of parameters — used for x-axis ordering.
# task-054: updated to current open weights (2026-05-06). For MoE entries,
# the value is effective active params at inference.
MODEL_SIZES: dict[str, float] = {
    "qwen3.5:0.8b": 0.8,
    "qwen3.5:2b": 2.0,
    "gemma4:e2b": 2.3,
    "qwen3.6:35b-a3b": 3.0,
    "gemma4:26b": 3.8,
    "qwen3.5:4b": 4.0,
    "gemma4:e4b": 4.5,
    "qwen3.5:9b": 9.0,
    "qwen3.6:27b": 27.0,
    "gemma4:31b": 31.0,
}

# Canonical model order by (effective) parameter count
MODEL_ORDER = [
    "qwen3.5:0.8b", "qwen3.5:2b", "gemma4:e2b", "qwen3.6:35b-a3b",
    "gemma4:26b", "qwen3.5:4b", "gemma4:e4b", "qwen3.5:9b",
    "qwen3.6:27b", "gemma4:31b",
]

STRATEGY_ORDER = ["adaptive", "corrective", "multi_query", "naive", "self_rag"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Check if a column exists and has at least one non-NaN value."""
    return col in df.columns and df[col].notna().any()


# Bootstrap settings used across every chart in this dashboard. 2000
# resamples is enough to stabilise the 2.5 / 97.5 percentiles for the
# 200-questions-per-cell sample sizes (cell SE on a 1-5 scale is well
# below 0.1, so the percentile estimate stabilises by ~1000 resamples).
# A fixed seed makes the rendered CIs identical across re-runs — the
# CIs are an analysis artefact, not a measurement, and a moving seed
# would silently shift the error bars between gallery regenerations.
_BOOT_N = 2000
_BOOT_RNG = np.random.default_rng(20260520)


def _bootstrap_mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI on the mean of ``values``.

    Uses the percentile method (no BCa correction) — good enough for the
    well-behaved continuous quality scores in the 1-5 range. Returns
    ``(nan, nan, nan)`` for inputs with fewer than 2 finite values so
    callers can render a single-point marker instead of an error bar.

    Args:
        values: 1D array-like of metric values, NaNs already dropped.
        level: Confidence level. Default 95%.

    Returns:
        Three floats: (point estimate, lower bound, upper bound).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size == 1 else float("nan"),
                float("nan"), float("nan"))
    # Resample row-indices with replacement, take the mean of each
    # resample, then take percentiles. vectorised in numpy.
    idx = _BOOT_RNG.integers(0, v.size, size=(_BOOT_N, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return (float(v.mean()),
            float(np.quantile(boot_means, lo)),
            float(np.quantile(boot_means, hi)))


def _ci_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return a DataFrame with mean + bootstrap CI for each group.

    Columns: ``group_col``, ``mean``, ``ci_lo``, ``ci_hi``,
    ``err_minus``, ``err_plus``, ``n``. The two ``err_*`` columns are
    half-widths suitable for direct use as Plotly's
    ``error_y=dict(type="data", arrayminus=..., array=...)``.
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
    config_means = df.groupby(["strategy", "model"])["consensus_quality"].mean()
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


def _chart_strategy_ranking_ci(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-strategy mean quality with 95% bootstrap CI error bars.

    The headline visual for &ldquo;are these strategies actually
    distinguishable?&rdquo; A vertical bar chart with error bars whose
    *overlap* answers that question directly: if a pair of CIs overlap,
    the means are not significantly different at the 95% level.

    Strategies are ordered by mean (descending) so the eye can scan
    left-to-right from best to worst. The reader walks away knowing,
    e.g., that Naive / Multi-Query / Corrective overlap (a 3-way tie)
    and that Self-RAG / Adaptive sit well below them.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    stats = _ci_table(df, "strategy", "consensus_quality")
    stats = stats.sort_values("mean", ascending=False).reset_index(drop=True)

    # Colour each bar from the IBM palette in rank order so the visual
    # rank also tracks colour gradient.
    colors = [IBM_COLORS[i % len(IBM_COLORS)] for i in range(len(stats))]
    fig = go.Figure(data=go.Bar(
        x=stats["strategy"], y=stats["mean"],
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
    # Tight y-axis from a fraction below the lowest CI to a fraction
    # above the highest so the error-bar overlap is easy to see.
    ymin = max(0.0, stats["ci_lo"].min() - 0.15)
    ymax = min(5.0, stats["ci_hi"].max() + 0.20)
    fig.update_layout(
        title="Per-Strategy Ranking with 95% Bootstrap CI",
        xaxis_title="Strategy",
        yaxis_title="Mean Quality (1–5 scale)",
        yaxis=dict(range=[ymin, ymax]),
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return ("Per-Strategy Ranking (with CIs)", fig)


def _chart_model_ranking_ci(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Per-model mean quality with 95% bootstrap CI error bars.

    Companion to ``_chart_strategy_ranking_ci``. Models ordered by
    parameter count (not by mean) so the size-vs-quality trend is
    visible and the eye can see where adjacent sizes are
    indistinguishable (e.g.&nbsp;qwen3.5:4b vs qwen3.5:9b in Exp 1).

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    stats = _ci_table(df, "model", "consensus_quality")
    # Keep size order so the chart reads as a scale curve.
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
    """Quality heatmap: x=model, y=strategy, z=mean quality.

    Uses Viridis colorscale for perceptual uniformity and accessibility.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    pivot = df.pivot_table(values="consensus_quality", index="strategy", columns="model", aggfunc="mean")
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
    """Quality vs model size: one line per strategy with 95% bootstrap CI.

    The error bars are 95% percentile bootstrap CIs on the *mean* for
    each (strategy, model) cell — not standard deviations of individual
    scores. Pre-CI versions of this chart showed per-row std, which is
    the spread of single-question quality (around 1.0 on a 1-5 scale)
    rather than the uncertainty in the cell mean (typically 0.05-0.15
    at n=200). Per-row std is huge and visually noisy; the CI tells
    you whether two cells are distinguishable, which is the actual
    question the reader is bringing to this chart.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    fig = go.Figure()
    strategies = _order_strategies(df["strategy"].unique().tolist())

    # Markers: different symbols to distinguish overlapping points (e.g. gemma4:e4b and qwen3.5:4b)
    markers = ["circle", "square", "diamond", "cross", "triangle-up", "star", "hexagon"]

    for i, strat in enumerate(strategies):
        sdf = df[df["strategy"] == strat].copy()
        sdf["model_size"] = sdf["model"].map(MODEL_SIZES)
        sdf = sdf.dropna(subset=["model_size"])
        # Per-cell bootstrap CI: groupby (model, model_size), compute
        # CI on the consensus_quality vector inside each cell.
        rows: list[dict] = []
        for (model, size), cell in sdf.groupby(["model", "model_size"]):
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
                visible=True,
                thickness=1.4,
                width=4,
            ),
            mode="lines+markers",
            name=strat,
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
        title="Quality vs Model Size by Strategy (error bars = 95% bootstrap CI on the mean)",
        xaxis_title="Model Size (B params)", yaxis_title="Mean Quality",
        height=500, legend_title="Strategy",
        plot_bgcolor="white", paper_bgcolor="white",
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
    config_means = df.groupby(["strategy", "model"])["consensus_quality"].mean().reset_index()
    config_means["model_size"] = config_means["model"].map(MODEL_SIZES)

    naive_configs = config_means[config_means["strategy"] == "naive"]
    non_naive = config_means[config_means["strategy"] != "naive"]

    results = []
    for _, row in non_naive.iterrows():
        # Find naive configs with larger models
        bigger_naive = naive_configs[naive_configs["model_size"] > row["model_size"]]
        for _, naive_row in bigger_naive.iterrows():
            if row["consensus_quality"] > naive_row["consensus_quality"]:
                results.append({
                    "strategy": row["strategy"],
                    "small_model": row["model"],
                    "large_naive_model": naive_row["model"],
                    "delta": row["consensus_quality"] - naive_row["consensus_quality"],
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
    which dimension drives quality differences. Per-judge columns
    (e.g. ``anthropic_claude_haiku_4_5_20251001_faithfulness``,
    ``openai_gpt_5_4_mini_faithfulness``) are averaged into a single
    per-metric series before grouping; this preserves the original
    "what aspect of quality breaks down" comparison while working with
    the multi-judge schema rolled out in task-052. Before the fix the
    chart rendered empty because it looked for bare metric columns
    that no longer exist in the panel-style CSV.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    metrics = ["faithfulness", "relevance", "conciseness"]
    # Discover per-judge columns of the form ``<safe_judge>_<metric>``
    # and average them into a single per-metric column for plotting.
    # Skip metrics with no columns at all so the legend doesn't carry a
    # phantom "Faithfulness" trace at zero.
    df = df.copy()
    available: list[str] = []
    for m in metrics:
        # Per-judge suffixes — exclude the consensus_* / answer_* names.
        cols = [
            c for c in df.columns
            if c.endswith(f"_{m}")
            and c not in (f"answer_{m}", f"consensus_{m}")
        ]
        if not cols:
            # Older single-judge CSVs may still expose the bare column.
            if _safe_col(df, m):
                df[f"_avg_{m}"] = df[m]
                available.append(m)
            continue
        df[f"_avg_{m}"] = df[cols].mean(axis=1, skipna=True)
        available.append(m)

    if not available:
        # Annotate the empty chart so it doesn't look like a render bug.
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
        df.groupby(["strategy", "model"])[["consensus_quality"] + avg_cols]
        .mean()
        .reset_index()
    )
    config_means["config"] = config_means["strategy"] + " + " + config_means["model"]
    config_means = config_means.sort_values("consensus_quality", ascending=False)

    # Top 10 and bottom 5 — same selection rule as before.
    top = config_means.head(10)
    bottom = config_means.tail(5)
    selected = pd.concat([top, bottom]).drop_duplicates(subset=["config"])

    # Per-cell bootstrap CI for each (strategy, model, metric). Pre-CI
    # versions of this chart drew bare means and so couldn't show
    # whether the metric gaps within a single config were
    # distinguishable from sampling noise.
    selected_keys = list(zip(selected["strategy"], selected["model"]))
    config_list = selected["config"].tolist()

    fig = go.Figure()
    for i, metric in enumerate(available):
        means: list[float] = []
        err_minus: list[float] = []
        err_plus: list[float] = []
        for strat, mdl in selected_keys:
            cell = df[(df["strategy"] == strat) & (df["model"] == mdl)]
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
        title="Per-Metric Breakdown — top 10 + bottom 5 configs (error bars = 95% CI on judge-averaged score)",
        xaxis_title="Configuration",
        yaxis_title="Score (1–5)",
        xaxis_tickangle=-45, height=520,
        margin=dict(b=150),
        plot_bgcolor="white", paper_bgcolor="white",
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
            y=df[df["strategy"] == strat]["consensus_quality"],
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
        consensus_quality=("consensus_quality", "mean"),
        latency=("strategy_latency_ms", "mean"),
    ).reset_index()
    config_stats["config"] = config_stats["strategy"] + " + " + config_stats["model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=config_stats["latency"], y=config_stats["consensus_quality"],
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


def _chart_per_query_detail(df: pd.DataFrame) -> tuple[str, go.Figure]:
    """Interactive table showing worst-10 and best-10 individual answers.

    Helps identify where RAG breaks down at the per-query level.

    Args:
        df: Experiment 1 raw scores DataFrame.

    Returns:
        Tuple of (title, Plotly figure).
    """
    sorted_df = df.sort_values("consensus_quality")
    worst = sorted_df.head(10)
    best = sorted_df.tail(10)
    selected = pd.concat([worst, best])

    def _clip(series: pd.Series, n: int) -> list[str]:
        """Truncate strings to ``n`` chars with an ellipsis marker.

        Plotly Table cells don't wrap horizontally usefully — long
        prose either pushes column widths off-screen or gets cut off
        in browser-dependent ways. A hard clip with a visible ``…``
        keeps the table scannable while preserving enough text for a
        reader to understand the row at a glance.
        """
        return [
            (str(v)[:n] + "…") if isinstance(v, str) and len(v) > n else (str(v) if v is not None else "")
            for v in series
        ]

    question_col = _clip(selected["question"], 70) if _safe_col(df, "question") else [""] * len(selected)
    gold_answer_col = _clip(selected["gold_answer"], 80) if _safe_col(df, "gold_answer") else [""] * len(selected)
    rag_answer_col  = _clip(selected["rag_answer"],  140) if _safe_col(df, "rag_answer") else [""] * len(selected)
    gold_f1_col = (
        selected["gold_f1"].round(3).astype(str).tolist()
        if _safe_col(df, "gold_f1") else ["N/A"] * len(selected)
    )

    fig = go.Figure(data=[go.Table(
        columnwidth=[60, 80, 220, 200, 320, 55, 55],
        header=dict(
            values=["Strategy", "Model", "Question",
                    "Gold answer", "RAG answer (truncated)",
                    "Quality", "Gold F1"],
            fill_color="#648FFF",
            font=dict(color="white", size=12),
            align="left",
            height=30,
        ),
        cells=dict(
            values=[
                selected["strategy"],
                selected["model"],
                question_col,
                gold_answer_col,
                rag_answer_col,
                selected["consensus_quality"].round(3),
                gold_f1_col,
            ],
            # Worst 10 on top in soft magenta, best 10 below in soft teal,
            # matching the IBM palette used elsewhere on the site.
            fill_color=[["#fde6ef"] * 10 + ["#e0f5ec"] * 10],
            font=dict(size=11),
            align="left",
            height=42,
        ),
    )])
    fig.update_layout(
        title="Per-Query Detail: Worst 10 + Best 10 (RAG answers truncated to ~140 chars)",
        height=850,
        margin=dict(l=0, r=0, t=44, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
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

    if df.empty or "consensus_quality" not in df.columns:
        logger.warning("Empty or invalid CSV at %s — returning empty figures", csv_path)
        return []

    if "strategy" not in df.columns or "model" not in df.columns:
        logger.warning("Missing strategy or model column — returning empty figures")
        return []

    figures: list[tuple[str, go.Figure]] = []

    # 1. Summary card
    figures.append(_chart_summary_card(df))
    # 2. CI-ranked headline charts — answer "are these means actually
    # distinguishable?" before any heatmaps or per-cell drill-downs.
    figures.append(_chart_strategy_ranking_ci(df))
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
