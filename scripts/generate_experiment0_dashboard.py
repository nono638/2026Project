#!/usr/bin/env python3
"""Generate interactive Plotly dashboard for Experiment 0 results.

Produces a single self-contained HTML file with 18 visualizations covering:
pipeline walkthrough, judge vs gold analysis, score distributions,
judge agreement, gold answer analysis, and data overview.

Why Plotly over matplotlib: interactive hover, zoom, and filter make
exploration far more effective than static PNGs. Self-contained HTML
requires no server — just open in a browser.

Usage:
    python scripts/generate_experiment0_dashboard.py
    python scripts/generate_experiment0_dashboard.py --skip-enrichment
"""

from __future__ import annotations

import argparse
import html as html_module
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so src imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column prefix → short display name for charts
JUDGE_DISPLAY_NAMES: dict[str, str] = {
    "google_gemini_2_5_flash_lite": "Flash-Lite",
    "google_gemini_2_5_flash": "Flash",
    "google_gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "anthropic_claude_haiku_4_5_20251001": "Claude Haiku",
    "anthropic_claude_sonnet_4_20250514": "Claude Sonnet",
    "anthropic_claude_opus_4_20250514": "Claude Opus",
}

_METRICS = ("faithfulness", "relevance", "conciseness")

# IBM Design colorblind-safe palette
_COLORS = [
    "#648FFF",  # blue
    "#785EF0",  # purple
    "#DC267F",  # magenta
    "#FE6100",  # orange
    "#FFB000",  # gold
    "#22A884",  # teal
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_valid_judges(
    df: pd.DataFrame,
    min_valid: int = 1,
) -> list[dict[str, Any]]:
    """Detect judge columns and return info for judges with enough valid data.

    Looks for columns matching `*_quality` pattern and checks how many
    non-NaN values exist. Judges with fewer than min_valid scores are
    excluded (e.g., gemini-2.5-pro with 0 valid scores).

    Args:
        df: Scores DataFrame with judge columns.
        min_valid: Minimum number of non-NaN quality scores to include.

    Returns:
        List of dicts with keys: prefix, display_name, valid_count, color.
    """
    judges = []
    # Find all *_quality columns
    quality_cols = [c for c in df.columns if c.endswith("_quality")]

    for col in quality_cols:
        prefix = col.replace("_quality", "")
        valid_count = int(df[col].notna().sum())

        if valid_count < min_valid:
            continue

        display_name = JUDGE_DISPLAY_NAMES.get(prefix, prefix)
        color = _COLORS[len(judges) % len(_COLORS)]

        judges.append({
            "prefix": prefix,
            "display_name": display_name,
            "valid_count": valid_count,
            "color": color,
        })

    return judges


def enrich_with_hotpotqa_metadata(
    df: pd.DataFrame,
    queries: list[Any],
) -> pd.DataFrame:
    """Add difficulty and question_type columns from HotpotQA query metadata.

    The queries list must be in the same order and length as the DataFrame
    rows (same seed/sample used in Experiment 0).

    Args:
        df: Scores DataFrame to enrich.
        queries: List of Query objects with metadata dicts.

    Returns:
        DataFrame with difficulty and question_type columns added.
    """
    df = df.copy()
    difficulties = []
    question_types = []

    for i in range(len(df)):
        if i < len(queries):
            meta = getattr(queries[i], "metadata", {}) or {}
            difficulties.append(meta.get("difficulty", "unknown"))
            question_types.append(meta.get("question_type", "unknown"))
        else:
            difficulties.append("unknown")
            question_types.append("unknown")

    df["difficulty"] = difficulties
    df["question_type"] = question_types
    return df


# ---------------------------------------------------------------------------
# Chart generators — each returns a Plotly figure or HTML string
# ---------------------------------------------------------------------------

def _chart_summary_card(
    scores_df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> str:
    """Generate HTML summary statistics card (#18).

    Args:
        scores_df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        HTML string for the summary card.
    """
    n = len(scores_df)
    exact_match_rate = scores_df["gold_exact_match"].mean() * 100 if "gold_exact_match" in scores_df.columns else 0
    mean_f1 = scores_df["gold_f1"].mean() if "gold_f1" in scores_df.columns else 0
    mean_bertscore = scores_df["gold_bertscore"].mean() if "gold_bertscore" in scores_df.columns else 0

    judge_lines = ""
    for j in judges:
        judge_lines += f'<li>{j["display_name"]}: {j["valid_count"]}/{n} scored</li>\n'

    return f"""
    <div class="summary-card">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item"><span class="stat-value">{n}</span><span class="stat-label">Examples</span></div>
            <div class="stat-item"><span class="stat-value">Qwen3 4B</span><span class="stat-label">Model (NaiveRAG)</span></div>
            <div class="stat-item"><span class="stat-value">{exact_match_rate:.0f}%</span><span class="stat-label">Exact Match Rate</span></div>
            <div class="stat-item"><span class="stat-value">{mean_bertscore:.3f}</span><span class="stat-label">Mean BERTScore</span></div>
            <div class="stat-item"><span class="stat-value">{mean_f1:.3f}</span><span class="stat-label">Mean F1</span></div>
            <div class="stat-item">
                <span class="stat-value">{len(judges)}</span>
                <span class="stat-label">Judges</span>
                <ul class="judge-list">{judge_lines}</ul>
            </div>
        </div>
    </div>
    """


def _chart_pipeline_walkthrough(
    scores_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> str:
    """Generate interactive pipeline walkthrough HTML widget (#1).

    Uses a <select> dropdown and inline JavaScript to show/hide content
    for each example. Not a Plotly chart — pure HTML/JS.

    Args:
        scores_df: Scores DataFrame.
        answers_df: Answers DataFrame with doc_text.
        judges: List of valid judge info dicts.

    Returns:
        HTML string with inline JS for the walkthrough widget.
    """
    # Merge doc_text from answers
    if "doc_text" not in scores_df.columns and answers_df is not None:
        merged = scores_df.merge(
            answers_df[["example_id", "doc_text"]],
            on="example_id",
            how="left",
        )
    else:
        merged = scores_df.copy()
        if "doc_text" not in merged.columns:
            merged["doc_text"] = ""

    options_html = ""
    examples_html = ""

    for _, row in merged.iterrows():
        eid = int(row["example_id"])
        q = html_module.escape(str(row.get("question", "")))
        q_short = q[:80] + "..." if len(q) > 80 else q
        options_html += f'<option value="example-{eid}">{eid}: {q_short}</option>\n'

        gold_answer = html_module.escape(str(row.get("gold_answer", "")))
        rag_answer = html_module.escape(str(row.get("rag_answer", "")))
        doc_text = html_module.escape(str(row.get("doc_text", ""))[:2000])
        exact_match = row.get("gold_exact_match", "N/A")
        f1 = row.get("gold_f1", 0)
        bertscore = row.get("gold_bertscore", 0)

        # Judge scores for this example
        judge_scores_html = ""
        for j in judges:
            prefix = j["prefix"]
            faith = row.get(f"{prefix}_faithfulness", float("nan"))
            rel = row.get(f"{prefix}_relevance", float("nan"))
            conc = row.get(f"{prefix}_conciseness", float("nan"))
            qual = row.get(f"{prefix}_quality", float("nan"))
            if pd.notna(qual):
                judge_scores_html += f"""
                <tr>
                    <td>{j['display_name']}</td>
                    <td>{faith:.0f}</td><td>{rel:.0f}</td><td>{conc:.0f}</td>
                    <td><strong>{qual:.2f}</strong></td>
                </tr>"""

        examples_html += f"""
        <div class="example-panel" id="example-{eid}" style="display:none;">
            <div class="pipeline-step">
                <h4>Question</h4>
                <p>{q}</p>
            </div>
            <div class="pipeline-step">
                <h4>Document (source text)</h4>
                <div class="scrollable-text">{doc_text}</div>
            </div>
            <div class="pipeline-step">
                <h4>RAG Answer</h4>
                <p class="answer-text">{rag_answer}</p>
            </div>
            <div class="pipeline-step">
                <h4>Gold Answer</h4>
                <p class="answer-text gold">{gold_answer}</p>
            </div>
            <div class="pipeline-step">
                <h4>Gold Metrics</h4>
                <p>Exact Match: <strong>{"Yes" if exact_match else "No"}</strong> |
                   F1: <strong>{f1:.3f}</strong> |
                   BERTScore: <strong>{bertscore:.3f}</strong></p>
            </div>
            <div class="pipeline-step">
                <h4>Judge Scores</h4>
                <table class="judge-table">
                    <tr><th>Judge</th><th>Faith.</th><th>Rel.</th><th>Conc.</th><th>Quality</th></tr>
                    {judge_scores_html}
                </table>
            </div>
        </div>
        """

    return f"""
    <div class="walkthrough-container">
        <label for="example-select"><strong>Select an example:</strong></label>
        <select id="example-select" onchange="showExample(this.value)">
            <option value="">-- Choose --</option>
            {options_html}
        </select>
        {examples_html}
    </div>
    """


def _chart_judge_vs_bertscore(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Judge quality vs BERTScore scatter with trendlines (#2).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    for j in judges:
        prefix = j["prefix"]
        mask = df[f"{prefix}_quality"].notna() & df["gold_bertscore"].notna()
        subset = df[mask]
        if len(subset) < 2:
            continue

        x = subset["gold_bertscore"].values
        y = subset[f"{prefix}_quality"].values
        q_text = [str(q)[:60] for q in subset["question"]]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=f'{j["display_name"]} (n={len(subset)})',
            marker=dict(color=j["color"], size=8),
            text=q_text,
            hovertemplate="<b>%{text}</b><br>BERTScore: %{x:.3f}<br>Quality: %{y:.2f}<extra></extra>",
        ))

        # OLS trendline
        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=j["color"], dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="Do judges agree with semantic similarity to the gold answer?",
        xaxis_title="Gold BERTScore",
        yaxis_title="Judge Quality Score",
        template="plotly_white",
        height=500,
    )
    return fig


def _chart_judge_vs_f1(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Judge quality vs gold F1 scatter with trendlines (#3).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    for j in judges:
        prefix = j["prefix"]
        mask = df[f"{prefix}_quality"].notna() & df["gold_f1"].notna()
        subset = df[mask]
        if len(subset) < 2:
            continue

        x = subset["gold_f1"].values
        y = subset[f"{prefix}_quality"].values
        q_text = [str(q)[:60] for q in subset["question"]]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=f'{j["display_name"]} (n={len(subset)})',
            marker=dict(color=j["color"], size=8),
            text=q_text,
            hovertemplate="<b>%{text}</b><br>F1: %{x:.3f}<br>Quality: %{y:.2f}<extra></extra>",
        ))

        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=j["color"], dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="Do judges agree with word-overlap correctness?",
        xaxis_title="Gold F1",
        yaxis_title="Judge Quality Score",
        template="plotly_white",
        height=500,
    )
    return fig


def _chart_judge_gold_correlation(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Judge-gold correlation bar chart (#4).

    Shows Pearson correlation of each judge's quality score with
    BERTScore and F1, sorted by BERTScore correlation descending.

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    data = []
    for j in judges:
        prefix = j["prefix"]
        mask = df[f"{prefix}_quality"].notna()
        subset = df[mask]
        if len(subset) < 3:
            continue

        quality = subset[f"{prefix}_quality"]
        # Pearson correlation with BERTScore
        bert_corr = quality.corr(subset["gold_bertscore"]) if "gold_bertscore" in subset.columns else 0
        f1_corr = quality.corr(subset["gold_f1"]) if "gold_f1" in subset.columns else 0
        data.append({
            "judge": j["display_name"],
            "BERTScore r": bert_corr if pd.notna(bert_corr) else 0,
            "F1 r": f1_corr if pd.notna(f1_corr) else 0,
        })

    # Sort by BERTScore correlation descending
    data.sort(key=lambda x: x["BERTScore r"], reverse=True)

    judges_sorted = [d["judge"] for d in data]
    bert_vals = [d["BERTScore r"] for d in data]
    f1_vals = [d["F1 r"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=judges_sorted, y=bert_vals, name="BERTScore correlation",
        text=[f"r={v:.3f}" for v in bert_vals], textposition="outside",
        marker_color=_COLORS[0],
    ))
    fig.add_trace(go.Bar(
        x=judges_sorted, y=f1_vals, name="F1 correlation",
        text=[f"r={v:.3f}" for v in f1_vals], textposition="outside",
        marker_color=_COLORS[3],
    ))

    fig.update_layout(
        title="Which judge best tracks ground truth?",
        xaxis_title="Judge",
        yaxis_title="Pearson Correlation",
        barmode="group",
        template="plotly_white",
        height=450,
    )
    return fig


def _chart_correct_vs_incorrect(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Box plots of judge scores split by exact match (#5).

    A good judge should give higher scores to correct answers.

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    for i, j in enumerate(judges):
        prefix = j["prefix"]
        mask = df[f"{prefix}_quality"].notna()
        subset = df[mask]
        if len(subset) < 2:
            continue

        for match_val, label, opacity in [(True, "Correct", 1.0), (False, "Incorrect", 0.6)]:
            group = subset[subset["gold_exact_match"] == match_val]
            if len(group) == 0:
                continue
            fig.add_trace(go.Box(
                y=group[f"{prefix}_quality"],
                name=f'{j["display_name"]} ({label})',
                marker_color=j["color"],
                opacity=opacity,
                boxmean=True,
            ))

    fig.update_layout(
        title="Can judges distinguish correct from incorrect answers?",
        yaxis_title="Judge Quality Score",
        template="plotly_white",
        height=500,
        showlegend=True,
    )
    return fig


def _chart_score_heatmap(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Per-example score heatmap (#6).

    Rows are examples sorted by BERTScore, columns are judges + gold metrics.

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    # Sort by BERTScore descending
    sorted_df = df.sort_values("gold_bertscore", ascending=False).copy()

    # Row labels: truncated questions
    row_labels = [str(q)[:40] for q in sorted_df["question"]]

    # Build matrix: judge quality scores (already 1-5) + gold metrics scaled to 1-5
    col_names = []
    z_data = []

    for j in judges:
        col_names.append(j["display_name"])
        z_data.append(sorted_df[f'{j["prefix"]}_quality'].values)

    # Gold metrics scaled to 1-5 range for visual consistency
    if "gold_bertscore" in sorted_df.columns:
        col_names.append("BERTScore (×5)")
        z_data.append(sorted_df["gold_bertscore"].values * 5)
    if "gold_f1" in sorted_df.columns:
        col_names.append("F1 (×5)")
        z_data.append(sorted_df["gold_f1"].values * 5)

    z_matrix = np.array(z_data).T  # rows=examples, cols=judges+gold

    # Hover text with full question and all scores
    hover_text = []
    for idx, row in sorted_df.iterrows():
        row_hover = []
        for col in col_names:
            row_hover.append(f"Q: {str(row['question'])[:80]}")
        hover_text.append(row_hover)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=col_names,
        y=row_labels,
        colorscale="RdYlGn",
        zmin=1,
        zmax=5,
        text=hover_text,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Score landscape across all judges and examples",
        template="plotly_white",
        height=max(400, len(sorted_df) * 20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _chart_violin_distributions(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Score distribution violin plots (#7).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    for j in judges:
        prefix = j["prefix"]
        vals = df[f"{prefix}_quality"].dropna()
        if len(vals) == 0:
            continue

        fig.add_trace(go.Violin(
            y=vals,
            name=j["display_name"],
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.3,
            marker_color=j["color"],
            line_color=j["color"],
        ))

    fig.update_layout(
        title="How do judges distribute their scores?",
        yaxis_title="Quality Score",
        template="plotly_white",
        height=450,
    )
    return fig


def _chart_metric_breakdown(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Grouped bar chart of mean faithfulness/relevance/conciseness per judge (#8).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    metric_colors = {"faithfulness": _COLORS[0], "relevance": _COLORS[2], "conciseness": _COLORS[4]}
    fig = go.Figure()

    for metric in _METRICS:
        means = []
        errors = []
        names = []
        for j in judges:
            col = f'{j["prefix"]}_{metric}'
            vals = df[col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            errors.append(vals.sem() if len(vals) > 1 else 0)
            names.append(j["display_name"])

        fig.add_trace(go.Bar(
            x=names, y=means, name=metric.capitalize(),
            error_y=dict(type="data", array=errors, visible=True),
            marker_color=metric_colors[metric],
        ))

    fig.update_layout(
        title="Which dimensions drive quality differences?",
        xaxis_title="Judge",
        yaxis_title="Mean Score (1-5)",
        barmode="group",
        template="plotly_white",
        height=450,
    )
    return fig


def _chart_score_vs_answer_length(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Score vs answer word count scatter (#9).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    # Compute average quality across all judges per example
    quality_cols = [f'{j["prefix"]}_quality' for j in judges]
    df = df.copy()
    df["avg_quality"] = df[quality_cols].mean(axis=1)
    df["answer_words"] = df["rag_answer"].astype(str).apply(lambda x: len(x.split()))

    color_col = "gold_exact_match" if "gold_exact_match" in df.columns else None
    fig = px.scatter(
        df, x="answer_words", y="avg_quality",
        color=color_col,
        hover_data=["question"],
        title="Does answer length affect judge scores?",
        labels={"answer_words": "Answer Word Count", "avg_quality": "Avg Quality (all judges)"},
        template="plotly_white",
    )
    fig.update_layout(height=450)
    return fig


def _chart_score_vs_question_length(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Score vs question word count scatter (#10).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    quality_cols = [f'{j["prefix"]}_quality' for j in judges]
    df = df.copy()
    df["avg_quality"] = df[quality_cols].mean(axis=1)
    df["question_words"] = df["question"].astype(str).apply(lambda x: len(x.split()))

    color_col = "difficulty" if "difficulty" in df.columns else None
    fig = px.scatter(
        df, x="question_words", y="avg_quality",
        color=color_col,
        hover_data=["question"],
        title="Does question complexity affect scores?",
        labels={"question_words": "Question Word Count", "avg_quality": "Avg Quality (all judges)"},
        template="plotly_white",
    )
    fig.update_layout(height=450)
    return fig


def _chart_inter_judge_correlation(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
) -> go.Figure:
    """Inter-judge correlation heatmap (#11).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.

    Returns:
        Plotly figure.
    """
    quality_cols = {j["display_name"]: f'{j["prefix"]}_quality' for j in judges}
    quality_df = df[list(quality_cols.values())].copy()
    quality_df.columns = list(quality_cols.keys())

    corr = quality_df.corr()
    # Format annotations
    text_vals = [[f"{v:.3f}" if pd.notna(v) else "" for v in row] for row in corr.values]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: r=%{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="How much do judges agree with each other?",
        template="plotly_white",
        height=450,
    )
    return fig


def _extract_article_titles(doc_text: str) -> list[str]:
    """Extract article titles from doc_text markdown headers.

    Doc text contains concatenated passages with ``## Title`` headers.
    Returns list of titles found.
    """
    import re
    return re.findall(r"^## (.+)$", str(doc_text), re.MULTILINE)


def _truncate_on_word_boundary(text: str, max_len: int) -> str:
    """Truncate text at a word boundary, adding ellipsis if shortened."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated + "..."


def _article_excerpt(doc_text: str, max_len: int = 200) -> str:
    """Return a substantive excerpt from doc_text, skipping headers/dividers.

    Collects body text (not ## headers or --- dividers) until max_len is
    reached, then truncates on a word boundary.
    """
    lines = []
    total = 0
    for line in str(doc_text).split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        lines.append(line)
        total += len(line)
        if total >= max_len:
            break
    excerpt = " ".join(lines)
    return _truncate_on_word_boundary(excerpt, max_len)


def _chart_biggest_disagreements(
    df: pd.DataFrame,
    judges: list[dict[str, Any]],
    answers_df: pd.DataFrame | None = None,
    model_name: str = "Qwen3 4B",
) -> str:
    """HTML table of top 10 examples with highest judge score variance (#12).

    Args:
        df: Scores DataFrame.
        judges: List of valid judge info dicts.
        answers_df: Optional answers DataFrame with doc_text and rag_answer.
        model_name: Display name for the RAG generation model.

    Returns:
        HTML string with table.
    """
    quality_cols = [f'{j["prefix"]}_quality' for j in judges]
    df = df.copy()
    df["quality_variance"] = df[quality_cols].var(axis=1)

    # Merge doc_text and rag_answer from answers if available
    if answers_df is not None:
        merge_cols = ["example_id"]
        if "doc_text" in answers_df.columns:
            merge_cols.append("doc_text")
        if "rag_answer" not in df.columns and "rag_answer" in answers_df.columns:
            merge_cols.append("rag_answer")
        df = df.merge(answers_df[merge_cols], on="example_id", how="left")

    top10 = df.nlargest(min(10, len(df)), "quality_variance")

    rows_html = ""
    for _, row in top10.iterrows():
        doc_text = str(row.get("doc_text", ""))
        titles = _extract_article_titles(doc_text)
        titles_display = html_module.escape(", ".join(titles)) if titles else "—"
        excerpt = html_module.escape(_article_excerpt(doc_text))

        # Full doc_text for hover (first 800 chars, word-boundary truncated)
        hover_doc = html_module.escape(
            _truncate_on_word_boundary(doc_text.replace("\n", " "), 800)
        )

        # Full question — no truncation in cell, full text in hover
        full_q = html_module.escape(str(row["question"]))

        # RAG answer — truncated in cell, full in hover
        full_answer = html_module.escape(str(row.get("rag_answer", "")))
        display_answer = html_module.escape(
            _truncate_on_word_boundary(str(row.get("rag_answer", "")), 150)
        )

        scores = " | ".join(
            f'{j["display_name"]}: {row.get(f"{j["prefix"]}_quality", float("nan")):.1f}'
            for j in judges
            if pd.notna(row.get(f'{j["prefix"]}_quality'))
        )
        em = "Yes" if row.get("gold_exact_match") else "No"
        bert = f'{row.get("gold_bertscore", 0):.3f}'
        var_val = f'{row["quality_variance"]:.2f}'
        rows_html += (
            f"<tr>"
            f'<td title="{hover_doc}">{titles_display}</td>'
            f'<td title="{hover_doc}">{excerpt or "—"}</td>'
            f'<td title="{full_q}">{full_q}</td>'
            f'<td title="{full_answer}">{display_answer}</td>'
            f"<td>{scores}</td><td>{em}</td><td>{bert}</td><td>{var_val}</td>"
            f"</tr>\n"
        )

    rag_col_name = html_module.escape(f"RAG Answer ({model_name})")
    return f"""
    <h3>Where do judges disagree most?</h3>
    <table class="data-table">
        <tr><th>Article(s)</th><th>Excerpt</th><th>Question</th><th>{rag_col_name}</th><th>Judge Scores</th><th>Exact Match</th><th>BERTScore</th><th>Judge Score Variance</th></tr>
        {rows_html}
    </table>
    """


def _chart_bertscore_distribution(df: pd.DataFrame) -> go.Figure:
    """BERTScore histogram (#13).

    Args:
        df: Scores DataFrame.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    if "gold_exact_match" in df.columns:
        for match_val, label, color in [(True, "Correct", _COLORS[0]), (False, "Incorrect", _COLORS[2])]:
            subset = df[df["gold_exact_match"] == match_val]
            fig.add_trace(go.Histogram(
                x=subset["gold_bertscore"], name=label,
                marker_color=color, opacity=0.7,
                nbinsx=15,
            ))
    else:
        fig.add_trace(go.Histogram(
            x=df["gold_bertscore"], nbinsx=15,
            marker_color=_COLORS[0],
        ))

    mean_val = df["gold_bertscore"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.3f}")

    fig.update_layout(
        title="Semantic similarity to gold answers (BERTScore)",
        xaxis_title="BERTScore",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig


def _chart_f1_distribution(df: pd.DataFrame) -> go.Figure:
    """F1 distribution histogram (#14).

    Args:
        df: Scores DataFrame.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    if "gold_exact_match" in df.columns:
        for match_val, label, color in [(True, "Correct", _COLORS[0]), (False, "Incorrect", _COLORS[2])]:
            subset = df[df["gold_exact_match"] == match_val]
            fig.add_trace(go.Histogram(
                x=subset["gold_f1"], name=label,
                marker_color=color, opacity=0.7,
                nbinsx=15,
            ))
    else:
        fig.add_trace(go.Histogram(
            x=df["gold_f1"], nbinsx=15,
            marker_color=_COLORS[0],
        ))

    mean_val = df["gold_f1"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.3f}")

    fig.update_layout(
        title="Word-overlap F1 with gold answers",
        xaxis_title="F1 Score",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig


def _chart_bertscore_vs_f1(df: pd.DataFrame) -> go.Figure:
    """BERTScore vs F1 scatter (#15).

    Args:
        df: Scores DataFrame.

    Returns:
        Plotly figure.
    """
    color_col = "gold_exact_match" if "gold_exact_match" in df.columns else None
    fig = px.scatter(
        df, x="gold_f1", y="gold_bertscore",
        color=color_col,
        hover_data=["question", "gold_answer", "rag_answer"],
        title="Do semantic and lexical metrics agree?",
        labels={"gold_f1": "Gold F1", "gold_bertscore": "Gold BERTScore"},
        template="plotly_white",
    )
    fig.update_layout(height=450)
    return fig


def _chart_question_length_distribution(df: pd.DataFrame) -> go.Figure:
    """Question word count histogram (#16).

    Args:
        df: Scores DataFrame.

    Returns:
        Plotly figure.
    """
    df = df.copy()
    df["question_words"] = df["question"].astype(str).apply(lambda x: len(x.split()))

    color_col = "question_type" if "question_type" in df.columns else None
    fig = px.histogram(
        df, x="question_words", color=color_col,
        title="Question Length Distribution",
        labels={"question_words": "Word Count"},
        template="plotly_white",
        nbins=15,
    )
    fig.update_layout(height=400)
    return fig


def _chart_answer_length_comparison(df: pd.DataFrame) -> go.Figure:
    """Side-by-side histograms of gold vs RAG answer lengths (#17).

    Args:
        df: Scores DataFrame.

    Returns:
        Plotly figure.
    """
    df = df.copy()
    df["gold_words"] = df["gold_answer"].astype(str).apply(lambda x: len(x.split()))
    df["rag_words"] = df["rag_answer"].astype(str).apply(lambda x: len(x.split()))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["gold_words"], name="Gold Answers",
        marker_color=_COLORS[0], opacity=0.7, nbinsx=15,
    ))
    fig.add_trace(go.Histogram(
        x=df["rag_words"], name="RAG Answers",
        marker_color=_COLORS[3], opacity=0.7, nbinsx=15,
    ))

    fig.update_layout(
        title="RAG answers vs gold answers: length comparison",
        xaxis_title="Word Count",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #fafafa;
    color: #222;
    line-height: 1.6;
}
h1 { color: #1a1a2e; border-bottom: 3px solid #648FFF; padding-bottom: 10px; }
h2 { color: #1a1a2e; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
h3 { color: #333; margin-top: 30px; }
.summary-card {
    background: white; border-radius: 12px; padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 20px 0;
}
.stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px; margin-top: 16px;
}
.stat-item { text-align: center; }
.stat-value { display: block; font-size: 1.8em; font-weight: 700; color: #648FFF; }
.stat-label { display: block; font-size: 0.85em; color: #666; margin-top: 4px; }
.judge-list { list-style: none; padding: 0; font-size: 0.8em; color: #666; text-align: left; }
.walkthrough-container { background: white; border-radius: 12px; padding: 24px; margin: 20px 0; }
.example-panel { margin-top: 20px; }
.pipeline-step {
    background: #f8f9fa; border-left: 4px solid #648FFF; padding: 12px 16px;
    margin: 10px 0; border-radius: 0 8px 8px 0;
}
.pipeline-step h4 { margin: 0 0 8px 0; color: #648FFF; }
.scrollable-text { max-height: 200px; overflow-y: auto; font-size: 0.9em; color: #555; }
.answer-text { font-size: 1.05em; }
.answer-text.gold { color: #2e7d32; font-weight: 600; }
.judge-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.judge-table th, .judge-table td { padding: 8px 12px; border: 1px solid #e0e0e0; text-align: center; }
.judge-table th { background: #f0f0f0; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin: 16px 0; }
.data-table th, .data-table td { padding: 8px; border: 1px solid #ddd; text-align: left; }
.data-table th { background: #f5f5f5; }
.chart-container { background: white; border-radius: 12px; padding: 16px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
select { font-size: 1em; padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; margin: 8px 0; min-width: 400px; }
"""

_JS_WALKTHROUGH = """
function showExample(id) {
    var panels = document.querySelectorAll('.example-panel');
    for (var i = 0; i < panels.length; i++) {
        panels[i].style.display = 'none';
    }
    if (id) {
        var el = document.getElementById(id);
        if (el) el.style.display = 'block';
    }
}
"""


def _fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div (no full page wrapper).

    Args:
        fig: Plotly figure.

    Returns:
        HTML string with the chart div.
    """
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def generate_dashboard(
    scores_path: Path | str,
    answers_path: Path | str,
    output_path: Path | str,
    skip_enrichment: bool = False,
    model_name: str = "Qwen3 4B",
) -> None:
    """Generate the complete Experiment 0 dashboard HTML.

    Args:
        scores_path: Path to raw_scores.csv.
        answers_path: Path to raw_answers.csv.
        output_path: Path for output HTML file.
        skip_enrichment: If True, skip HotpotQA metadata enrichment.
        model_name: Display name for the RAG generation model.
    """
    scores_path = Path(scores_path)
    answers_path = Path(answers_path)
    output_path = Path(output_path)

    scores_df = pd.read_csv(scores_path)
    answers_df = pd.read_csv(answers_path) if answers_path.exists() else None

    # Enrichment: add HotpotQA difficulty/type if possible
    if not skip_enrichment:
        try:
            from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
            docs, queries = load_hotpotqa(split="train")
            _, sampled_queries = sample_hotpotqa(docs, queries, n=len(scores_df), seed=42)
            scores_df = enrich_with_hotpotqa_metadata(scores_df, sampled_queries)
            # Save enriched CSV back
            scores_df.to_csv(scores_path, index=False)
            logger.info("Enriched CSV with HotpotQA metadata")
        except Exception as exc:
            logger.warning("HotpotQA enrichment failed (continuing without): %s", exc)

    # Detect valid judges
    judges = get_valid_judges(scores_df, min_valid=1)
    logger.info("Valid judges: %s", [j["display_name"] for j in judges])

    # Generate all chart HTML fragments
    parts: list[str] = []

    # Summary card (#18 — placed at top)
    parts.append(_chart_summary_card(scores_df, judges))

    # Section 1: Pipeline Walkthrough
    parts.append("<h2>Section 1: Pipeline Walkthrough</h2>")
    parts.append(_chart_pipeline_walkthrough(scores_df, answers_df, judges))

    # Section 2: Judge vs Gold
    parts.append("<h2>Section 2: Judge vs Gold (Primary Analysis)</h2>")
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_judge_vs_bertscore(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_judge_vs_f1(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_judge_gold_correlation(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_correct_vs_incorrect(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_score_heatmap(scores_df, judges))}</div>')

    # Section 3: Score Distributions
    parts.append("<h2>Section 3: Score Distributions</h2>")
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_violin_distributions(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_metric_breakdown(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_score_vs_answer_length(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_score_vs_question_length(scores_df, judges))}</div>')

    # Section 4: Judge Agreement
    parts.append("<h2>Section 4: Judge Agreement</h2>")
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_inter_judge_correlation(scores_df, judges))}</div>')
    parts.append(f'<div class="chart-container">{_chart_biggest_disagreements(scores_df, judges, answers_df, model_name)}</div>')

    # Section 5: Gold Answer Analysis
    parts.append("<h2>Section 5: Gold Answer Analysis</h2>")
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_bertscore_distribution(scores_df))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_f1_distribution(scores_df))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_bertscore_vs_f1(scores_df))}</div>')

    # Section 6: Data Overview
    parts.append("<h2>Section 6: Data Overview</h2>")
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_question_length_distribution(scores_df))}</div>')
    parts.append(f'<div class="chart-container">{_fig_to_html(_chart_answer_length_comparison(scores_df))}</div>')

    body_content = "\n".join(parts)

    # Assemble full HTML page
    # Use Plotly CDN for JS to keep file size manageable
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment 0: Scorer Validation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
{_CSS}
    </style>
</head>
<body>
    <h1>Experiment 0: Scorer Validation</h1>
    {body_content}
    <script>
{_JS_WALKTHROUGH}
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for generating the Experiment 0 dashboard."""
    parser = argparse.ArgumentParser(description="Generate Experiment 0 Plotly dashboard")
    parser.add_argument(
        "--skip-enrichment", action="store_true",
        help="Skip HotpotQA metadata enrichment (faster, no network needed)",
    )
    parser.add_argument(
        "--output", type=str, default="visuals/experiment_0.html",
        help="Output HTML path (default: visuals/experiment_0.html)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen3 4B",
        help="Display name for the RAG generation model (default: Qwen3 4B)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generate_dashboard(
        scores_path=Path("results/experiment_0/raw_scores.csv"),
        answers_path=Path("results/experiment_0/raw_answers.csv"),
        output_path=Path(args.output),
        skip_enrichment=args.skip_enrichment,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
