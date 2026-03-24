#!/usr/bin/env python3
"""Generate the RAGBench findings gallery — a static HTML site.

Produces a browsable set of HTML pages from experiment results:
- Index page with project overview and links to experiment dashboards
- Experiment 0 dashboard (from existing Plotly chart generators)
- Placeholder pages for Experiment 1 & 2 (auto-upgrade when data exists)

All output is self-contained — inline CSS/JS, no external CDN dependencies.
Plotly charts are embedded via plotly.io.to_html(full_html=False) wrapped in
a shared page template.

Usage:
    python scripts/generate_gallery.py
    python scripts/generate_gallery.py --output site_custom/
    python scripts/generate_gallery.py --experiments 0,1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so src/scripts imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

try:
    import plotly.io as pio
except ImportError:
    print("ERROR: plotly is required. Install with: pip install plotly")
    sys.exit(1)

logger = logging.getLogger(__name__)

# IBM Design colorblind-safe palette (same as existing dashboard)
_COLORS = [
    "#648FFF",  # blue
    "#785EF0",  # purple
    "#DC267F",  # magenta
    "#FE6100",  # orange
    "#FFB000",  # gold
    "#22A884",  # teal
]

# Experiment descriptions for placeholder pages
_EXPERIMENT_DESCRIPTIONS = {
    1: (
        "Strategy × Model Size — 5 RAG strategies (NaiveRAG, SelfRAG, "
        "CorrectiveRAG, AdaptiveRAG, MultiQueryRAG) × 6 models "
        "(Qwen3 0.6B/1.7B/4B/8B, Gemma 3 1B/4B) = 30 configurations. "
        "Held constant: Recursive chunker (500/100), mxbai-embed-large, "
        "hybrid retrieval."
    ),
    2: (
        "Chunking × Model Size — 4 chunking strategies "
        "(Fixed 512, Recursive 500/100, Sentence, Semantic) × 4 Qwen3 models "
        "(0.6B/1.7B/4B/8B) = 16 configurations. Held constant: NaiveRAG "
        "strategy, mxbai-embed-large, hybrid retrieval."
    ),
}


# ---------------------------------------------------------------------------
# Shared CSS theme
# ---------------------------------------------------------------------------

_GALLERY_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #fafafa;
    color: #222;
    line-height: 1.6;
}

/* Navigation bar */
.nav {
    background: #1a1a2e;
    padding: 12px 24px;
    display: flex;
    gap: 24px;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav a {
    color: #aab;
    text-decoration: none;
    font-size: 0.95em;
    padding: 6px 12px;
    border-radius: 6px;
    transition: background 0.2s, color 0.2s;
}
.nav a:hover { background: rgba(255,255,255,0.1); color: #fff; }
.nav a.active { background: #648FFF; color: #fff; font-weight: 600; }
.nav .brand {
    font-weight: 700;
    font-size: 1.1em;
    color: #648FFF;
    margin-right: 16px;
}

/* Main content */
.content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
}
h1 {
    color: #1a1a2e;
    border-bottom: 3px solid #648FFF;
    padding-bottom: 10px;
    margin-bottom: 24px;
}
h2 {
    color: #1a1a2e;
    margin-top: 32px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
}
h3 { color: #333; margin-top: 24px; }
p { margin: 12px 0; }

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* Experiment list on index */
.experiment-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 24px 0;
}
.experiment-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.experiment-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}
.experiment-card a {
    text-decoration: none;
    color: inherit;
    display: block;
}
.experiment-card h3 { color: #648FFF; margin-top: 0; }
.experiment-card .status {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
}
.status-ready { background: #e8f5e9; color: #2e7d32; }
.status-placeholder { background: #fff3e0; color: #e65100; }

/* Chart container */
.chart-container {
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin: 20px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Placeholder page */
.placeholder {
    text-align: center;
    padding: 60px 24px;
}
.placeholder h2 { border: none; color: #888; }
.placeholder p { color: #666; max-width: 600px; margin: 16px auto; }

/* Data tables */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin: 16px 0; }
.data-table th, .data-table td { padding: 10px 12px; border: 1px solid #ddd; text-align: left; vertical-align: top; }
.data-table th { background: #f5f5f5; font-weight: 600; }

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: #888;
    font-size: 0.85em;
    border-top: 1px solid #eee;
    margin-top: 40px;
}
"""


# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------

_NAV_ITEMS = [
    ("home", "Home", "index.html"),
    ("exp0", "Exp 0: Scorer Validation", "experiment_0.html"),
    ("exp1", "Exp 1: Strategy × Model", "experiment_1.html"),
    ("exp2", "Exp 2: Chunking × Model", "experiment_2.html"),
]


def _build_page_template(
    title: str,
    nav_active: str,
    content_html: str,
) -> str:
    """Wrap content HTML in the shared page template with nav and CSS.

    Args:
        title: Page title for ``<title>`` and ``<h1>``.
        nav_active: Key of the active nav item (e.g. ``"home"``, ``"exp0"``).
        content_html: Inner HTML for the page body.

    Returns:
        Complete HTML page string.
    """
    nav_links = []
    for key, label, href in _NAV_ITEMS:
        cls = ' class="active"' if key == nav_active else ""
        nav_links.append(f'<a href="{href}"{cls}>{label}</a>')
    nav_html = "\n    ".join(nav_links)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — RAGBench</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
{_GALLERY_CSS}
    </style>
</head>
<body>
    <nav class="nav">
        <span class="brand">RAGBench</span>
        {nav_html}
    </nav>
    <div class="content">
        <h1>{title}</h1>
        {content_html}
    </div>
    <div class="footer">
        RAGBench — A configurable RAG evaluation pipeline.
        Generated by <code>scripts/generate_gallery.py</code>.
    </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def _generate_index(experiments_info: list[dict[str, Any]]) -> str:
    """Build the index page HTML content (inside the template).

    Args:
        experiments_info: List of dicts with ``num``, ``title``, ``status``
            (``"ready"`` or ``"placeholder"``).

    Returns:
        HTML content string for the index page body.
    """
    # Experiment cards
    cards = []
    for exp in experiments_info:
        status_class = "status-ready" if exp["status"] == "ready" else "status-placeholder"
        status_label = "Results Available" if exp["status"] == "ready" else "Coming Soon"
        cards.append(f"""
        <div class="experiment-card">
            <a href="experiment_{exp['num']}.html">
                <span class="status {status_class}">{status_label}</span>
                <h3>Experiment {exp['num']}: {exp['title']}</h3>
                <p>{exp.get('description', '')}</p>
            </a>
        </div>""")

    cards_html = "\n".join(cards)

    return f"""
    <div class="card">
        <h2>About RAGBench</h2>
        <p>
            RAGBench is a configurable RAG evaluation pipeline that runs the full
            cartesian product of RAG configurations (chunker × embedder × strategy
            × language model) against any corpus, scores the results, and trains a
            meta-learner to predict the optimal configuration for new queries.
        </p>
        <p>
            This gallery presents pre-computed experimental results — interactive
            visualizations showing how different RAG configurations compare across
            quality, latency, and cost dimensions.
        </p>
    </div>

    <h2>Experiments</h2>
    <div class="experiment-grid">
        {cards_html}
    </div>

    <div class="card">
        <h2>Key Findings</h2>
        <p>
            <strong>Experiment 0 (Scorer Validation):</strong> Compared 6 LLM judges
            on 50 HotpotQA questions scored by NaiveRAG + Qwen3 4B. Finding: Claude
            Sonnet is the most accurate scorer (r&nbsp;=&nbsp;0.68 with gold-standard
            metrics). Gemini 2.5 Flash is a strong budget alternative at 50× lower cost.
        </p>
    </div>
    """


# ---------------------------------------------------------------------------
# Experiment 0 dashboard
# ---------------------------------------------------------------------------

def _create_workflow_diagram() -> str:
    """Create a Plotly-based workflow diagram showing the Experiment 0 pipeline.

    Returns:
        HTML string of the embedded Plotly figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Two-row layout with generous spacing.
    # Row 1 (y=2.0): HotpotQA → Question + Docs → RAG Pipeline → RAG Answer → 6 LLM Judges
    # Row 2 (y=0.0): Gold Answer ─────────────────────────────→ Compare ←─────┘
    #
    # Use unitless coordinates; aspect ratio set by height/xrange.
    boxes = [
        # (x, y, w, h, label, color)
        (1,  2, 2.4, 1.0, "HotpotQA<br>Dataset",           "#648FFF"),
        (5,  2, 2.8, 1.0, "Question +<br>Source Docs",      "#785EF0"),
        (9,  2, 3.2, 1.0, "RAG Pipeline<br><span style='font-size:11px'>(NaiveRAG + Qwen3 4B)</span>", "#DC267F"),
        (13, 2, 2.4, 1.0, "RAG Answer",                     "#FE6100"),
        (17, 2, 2.4, 1.0, "6 LLM Judges",                   "#FFB000"),
        (1,  0, 2.4, 1.0, "Gold Answer",                     "#22A884"),
        (9,  0, 3.6, 1.0, "Automated Metrics<br>(BERTScore, F1)", "#22A884"),
        (17, 0, 3.0, 1.0, "Which judge<br>tracks truth?",   "#648FFF"),
    ]

    shapes = []
    annotations = []

    for x, y, w, h, label, color in boxes:
        shapes.append(dict(
            type="rect",
            x0=x - w / 2, y0=y - h / 2,
            x1=x + w / 2, y1=y + h / 2,
            fillcolor=color, opacity=0.9,
            line=dict(color="white", width=2),
            layer="above",
        ))
        annotations.append(dict(
            x=x, y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color="white", size=13),
            align="center",
        ))

    # Arrows: (tail_x, tail_y, head_x, head_y)
    arrows = [
        (2.2, 2, 3.6, 2),     # HotpotQA → Q+Docs
        (6.4, 2, 7.6, 2),     # Q+Docs → RAG Pipeline
        (10.4, 2, 11.8, 2),   # RAG Pipeline → RAG Answer
        (14.2, 2, 15.8, 2),   # RAG Answer → 6 LLM Judges
        (1, 1.5, 1, 0.5),     # HotpotQA ↓ Gold Answer
        (2.2, 0, 7.2, 0),     # Gold Answer → Automated Metrics
        (13, 1.5, 13, 0.65),  # RAG Answer ↓ (toward Automated Metrics)
        (13, 0.35, 10.8, 0),  # ↓ into Automated Metrics
        (10.8, 0, 15.5, 0),   # Automated Metrics → Which judge
        (17, 1.5, 17, 0.5),   # 6 LLM Judges ↓ Which judge
    ]

    for ax, ay, x, y in arrows:
        annotations.append(dict(
            x=x, y=y, ax=ax, ay=ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#555", text="",
        ))

    fig.update_layout(
        xaxis=dict(range=[-1, 20], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1, 3.2], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        shapes=shapes, annotations=annotations,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False))

    return fig


def _build_row_examiner(scores_df: pd.DataFrame, answers_df: pd.DataFrame) -> str:
    """Build an interactive row examiner widget for the Experiment 0 page.

    Shows each example's full pipeline: document, question, gold answer,
    RAG answer, automated metrics, and all judge scores.  Supports sorting
    by question text, quality score, F1, or BERTScore.  Highlights the gold
    answer within the source document.

    Args:
        scores_df: Scores DataFrame (raw_scores.csv).
        answers_df: Answers DataFrame (raw_answers.csv) with doc_text.

    Returns:
        HTML string with inline CSS/JS for the widget.
    """
    import html as html_module
    from scripts.generate_experiment0_dashboard import get_valid_judges

    judges = get_valid_judges(scores_df, min_valid=1)

    # Merge doc_text from answers
    merged = scores_df.merge(
        answers_df[["example_id", "doc_text"]],
        on="example_id",
        how="left",
    )
    if "doc_text" not in merged.columns:
        merged["doc_text"] = ""

    # Compute a representative quality score (mean across all judges)
    quality_cols = [f"{j['prefix']}_quality" for j in judges]
    merged["_mean_quality"] = merged[quality_cols].mean(axis=1)

    # Build JSON-serializable data for JS sorting
    import json

    examples_data = []  # for JS sort metadata
    examples_html = ""

    for _, row in merged.iterrows():
        eid = int(row["example_id"])
        q = str(row.get("question", ""))
        q_escaped = html_module.escape(q)
        q_short = (q[:80] + "...") if len(q) > 80 else q

        gold_answer = str(row.get("gold_answer", ""))
        gold_answer_escaped = html_module.escape(gold_answer)
        rag_answer = html_module.escape(str(row.get("rag_answer", "")))
        doc_text_raw = str(row.get("doc_text", ""))
        exact_match = row.get("gold_exact_match", False)
        f1 = float(row.get("gold_f1", 0))
        bertscore = float(row.get("gold_bertscore", 0))
        mean_quality = float(row.get("_mean_quality", 0))

        # Highlight gold answer in doc text (case-insensitive)
        doc_text_escaped = html_module.escape(doc_text_raw)
        if gold_answer.strip():
            import re
            pattern = re.escape(html_module.escape(gold_answer.strip()))
            doc_text_highlighted = re.sub(
                f"({pattern})",
                r'<mark style="background:#FFD54F;padding:1px 3px;border-radius:3px">\1</mark>',
                doc_text_escaped,
                flags=re.IGNORECASE,
            )
        else:
            doc_text_highlighted = doc_text_escaped

        # Judge scores table
        judge_rows_html = ""
        for j in judges:
            prefix = j["prefix"]
            faith = row.get(f"{prefix}_faithfulness", float("nan"))
            rel = row.get(f"{prefix}_relevance", float("nan"))
            conc = row.get(f"{prefix}_conciseness", float("nan"))
            qual = row.get(f"{prefix}_quality", float("nan"))
            if pd.notna(qual):
                judge_rows_html += (
                    f"<tr><td>{j['display_name']}</td>"
                    f"<td>{faith:.0f}</td><td>{rel:.0f}</td><td>{conc:.0f}</td>"
                    f"<td><strong>{qual:.2f}</strong></td></tr>"
                )

        examples_data.append({
            "id": eid,
            "question": q,
            "label": f"{eid}: {q_short}",
            "quality": round(mean_quality, 3),
            "f1": round(f1, 3),
            "bertscore": round(bertscore, 3),
        })

        examples_html += f"""
        <div class="example-panel" id="rex-{eid}" style="display:none;">
            <div class="rex-step">
                <h4>Question</h4>
                <p style="font-size:1.05em;">{q_escaped}</p>
            </div>
            <div class="rex-step">
                <h4>Source Document <span style="font-weight:normal;color:#888;font-size:0.85em;">
                    (gold answer highlighted if found)</span></h4>
                <div class="rex-doc">{doc_text_highlighted}</div>
            </div>
            <div class="rex-step rex-answers">
                <div class="rex-answer-box">
                    <h4>Gold Answer</h4>
                    <p class="rex-gold">{gold_answer_escaped}</p>
                </div>
                <div class="rex-answer-box">
                    <h4>RAG Answer</h4>
                    <p class="rex-rag">{rag_answer}</p>
                </div>
            </div>
            <div class="rex-step">
                <h4>Automated Metrics</h4>
                <p>
                    Exact Match: <strong>{"Yes" if exact_match else "No"}</strong> &nbsp;|&nbsp;
                    F1 (word overlap): <strong>{f1:.3f}</strong> &nbsp;|&nbsp;
                    BERTScore (semantic): <strong>{bertscore:.3f}</strong>
                </p>
            </div>
            <div class="rex-step">
                <h4>Judge Scores</h4>
                <table class="data-table" style="max-width:600px;">
                    <tr><th>Judge</th><th>Faith.</th><th>Rel.</th><th>Conc.</th><th>Quality</th></tr>
                    {judge_rows_html}
                </table>
            </div>
        </div>
        """

    examples_json = json.dumps(examples_data)

    return f"""
    <style>
    .rex-controls {{
        display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
        margin-bottom: 16px;
    }}
    .rex-controls label {{ font-weight: 600; font-size: 0.9em; }}
    .rex-controls select {{
        padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;
        font-size: 0.95em; min-width: 200px;
    }}
    .rex-step {{
        background: #f8f9fa; border-radius: 8px; padding: 16px;
        margin: 12px 0; border-left: 4px solid #648FFF;
    }}
    .rex-step h4 {{ margin: 0 0 8px 0; color: #1a1a2e; }}
    .rex-doc {{
        max-height: 400px; overflow-y: auto; font-size: 0.88em;
        line-height: 1.7; white-space: pre-wrap; word-wrap: break-word;
        background: white; padding: 12px; border-radius: 6px;
        border: 1px solid #e0e0e0;
    }}
    .rex-answers {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
        background: transparent; border: none; padding: 0;
    }}
    .rex-answer-box {{
        background: #f8f9fa; border-radius: 8px; padding: 16px;
        border-left: 4px solid #22A884;
    }}
    .rex-answer-box:last-child {{ border-left-color: #FE6100; }}
    .rex-answer-box h4 {{ margin: 0 0 8px 0; color: #1a1a2e; }}
    .rex-gold {{ color: #2e7d32; font-weight: 600; }}
    .rex-rag {{ color: #e65100; }}
    </style>

    <div class="rex-controls">
        <label for="rex-sort">Sort by:</label>
        <select id="rex-sort" onchange="rexSort()">
            <option value="alpha">Question (A&rarr;Z)</option>
            <option value="quality_desc">Avg Judge Quality (high&rarr;low)</option>
            <option value="quality_asc">Avg Judge Quality (low&rarr;high)</option>
            <option value="bertscore_desc">BERTScore (high&rarr;low)</option>
            <option value="bertscore_asc">BERTScore (low&rarr;high)</option>
            <option value="f1_desc">F1 (high&rarr;low)</option>
            <option value="f1_asc">F1 (low&rarr;high)</option>
        </select>
        <label for="rex-select">Example:</label>
        <select id="rex-select" onchange="rexShow(this.value)" style="min-width:400px;">
            <option value="">-- Choose a question --</option>
        </select>
    </div>
    {examples_html}

    <script>
    (function() {{
        var data = {examples_json};
        var selectEl = document.getElementById('rex-select');
        var sortEl = document.getElementById('rex-sort');

        function populateDropdown(sorted) {{
            var current = selectEl.value;
            selectEl.innerHTML = '<option value="">-- Choose a question --</option>';
            sorted.forEach(function(d) {{
                var opt = document.createElement('option');
                opt.value = 'rex-' + d.id;
                var suffix = ' [Q=' + d.quality.toFixed(2) + ', BERT=' + d.bertscore.toFixed(3) + ', F1=' + d.f1.toFixed(3) + ']';
                opt.textContent = d.label + suffix;
                selectEl.appendChild(opt);
            }});
            if (current) selectEl.value = current;
        }}

        window.rexSort = function() {{
            var mode = sortEl.value;
            var sorted = data.slice();
            if (mode === 'alpha') sorted.sort(function(a,b) {{ return a.question.localeCompare(b.question); }});
            else if (mode === 'quality_desc') sorted.sort(function(a,b) {{ return b.quality - a.quality; }});
            else if (mode === 'quality_asc') sorted.sort(function(a,b) {{ return a.quality - b.quality; }});
            else if (mode === 'bertscore_desc') sorted.sort(function(a,b) {{ return b.bertscore - a.bertscore; }});
            else if (mode === 'bertscore_asc') sorted.sort(function(a,b) {{ return a.bertscore - b.bertscore; }});
            else if (mode === 'f1_desc') sorted.sort(function(a,b) {{ return b.f1 - a.f1; }});
            else if (mode === 'f1_asc') sorted.sort(function(a,b) {{ return a.f1 - b.f1; }});
            populateDropdown(sorted);
        }};

        window.rexShow = function(val) {{
            var panels = document.querySelectorAll('.example-panel');
            panels.forEach(function(p) {{ p.style.display = 'none'; }});
            if (val) {{
                var el = document.getElementById(val);
                if (el) el.style.display = 'block';
            }}
        }};

        // Initial populate
        rexSort();
    }})();
    </script>
    """


def _generate_experiment_0_v2(csv_path: Path) -> str:
    """Build the Experiment 0 v2 dashboard page with v2-specific charts.

    v2 adds answer_quality distribution, failure_stage breakdown, and
    reuses the standard Experiment 0 charts (correlation, distributions, etc.)
    from the v2 data.

    Args:
        csv_path: Path to ``results/experiment_0_v2/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    import plotly.graph_objects as go

    df = pd.read_csv(csv_path)

    # Try to reuse the standard Exp 0 chart builder for scorer charts
    try:
        from scripts.generate_experiment0_dashboard import (
            build_experiment0_figures,
            _fig_to_html,
        )
        figures = build_experiment0_figures(df)
    except Exception:
        figures = []

        def _fig_to_html(fig: Any) -> str:
            """Convert a Plotly figure to inline HTML."""
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    # Build v2-specific charts
    v2_charts: list[str] = []

    # Chart 1: answer_quality distribution
    if "answer_quality" in df.columns:
        counts = df["answer_quality"].value_counts()
        labels = ["good", "questionable", "poor"]
        values = [counts.get(l, 0) for l in labels]
        colors = ["#22A884", "#FFB000", "#DC267F"]  # teal, gold, magenta

        fig_aq = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="auto")
        ])
        fig_aq.update_layout(
            title="Answer Quality Distribution",
            xaxis_title="Quality Label",
            yaxis_title="Count",
            template="plotly_white",
            height=400,
        )
        v2_charts.append(f"""
        <div class="chart-container">
            <h3>Answer Quality Distribution</h3>
            <p class="chart-explanation" style="color: #555; font-size: 0.92em; line-height: 1.5; margin: 8px 0 16px 0; padding: 0 8px;">
                Triangulates BERTScore (semantic), F1 (lexical), and Sonnet (LLM judgment) to classify
                each answer. All three must agree for "good"; any single metric below threshold flags "poor".
            </p>
            {_fig_to_html(fig_aq)}
        </div>""")

    # Chart 2: failure_stage breakdown
    if "failure_stage" in df.columns:
        stage_counts = df["failure_stage"].value_counts()
        stage_labels = stage_counts.index.tolist()
        stage_values = stage_counts.values.tolist()

        fig_fs = go.Figure(data=[
            go.Bar(
                x=stage_labels, y=stage_values,
                marker_color=_COLORS[:len(stage_labels)],
                text=stage_values, textposition="auto",
            )
        ])
        fig_fs.update_layout(
            title="Failure Stage Breakdown",
            xaxis_title="Pipeline Stage",
            yaxis_title="Count",
            template="plotly_white",
            height=400,
        )
        v2_charts.append(f"""
        <div class="chart-container">
            <h3>Failure Stage Breakdown</h3>
            <p class="chart-explanation" style="color: #555; font-size: 0.92em; line-height: 1.5; margin: 8px 0 16px 0; padding: 0 8px;">
                Where in the pipeline did wrong answers go wrong? "correct" means the answer matched
                the gold standard. Other stages show where information was lost.
            </p>
            {_fig_to_html(fig_fs)}
        </div>""")

    # Standard scorer charts from build_experiment0_figures
    scorer_charts: list[str] = []
    for title, fig in figures:
        scorer_charts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>
            {_fig_to_html(fig)}
        </div>""")

    # Assemble page
    content = f"""
    <div class="card">
        <h2>Experiment 0 v2: Scorer Validation (Revised)</h2>
        <p>
            Version 2 fixes five oversights from v1: (1) captures what the LLM actually saw
            (context_sent_to_llm), (2) scorer judges against retrieved chunks not the full document,
            (3) adds BGE reranker (retrieve 10, keep 3), (4) filters to medium+hard questions only
            (150 total) to avoid ceiling effects, (5) adds composite answer_quality column.
        </p>
        <p style="margin-top: 8px; font-size: 0.9em;">
            <a href="raw_scores_v2.csv" style="color: #648FFF;">Download the v2 raw data (CSV)</a>
        </p>
    </div>

    {"".join(v2_charts)}

    <div class="card">
        <h2>Scorer Comparison Charts</h2>
        <p>Same scorer analysis as v1, but on the v2 dataset (harder questions, better context).</p>
    </div>

    {"".join(scorer_charts)}
    """

    return _build_page_template(
        "Experiment 0 v2 — Scorer Validation (Revised)",
        nav_active="exp0",
        content_html=content,
    )


def _generate_experiment_0(csv_path: Path) -> str:
    """Build the Experiment 0 dashboard page content with Plotly charts.

    Restructured for progressive disclosure: workflow diagram and headline
    result first, then judge comparison, then measurement details, then
    deep-dive charts.

    Args:
        csv_path: Path to ``results/experiment_0/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    from scripts.generate_experiment0_dashboard import (
        build_experiment0_figures,
        _fig_to_html,
    )

    df = pd.read_csv(csv_path)
    figures = build_experiment0_figures(df)

    # Load answers for the row examiner (doc_text lives here)
    answers_csv = csv_path.parent / "raw_answers.csv"
    answers_df = pd.read_csv(answers_csv) if answers_csv.exists() else None

    # Build a dict for random access by chart title
    figures_by_title: dict[str, str] = {}
    for title, fig in figures:
        figures_by_title[title] = _fig_to_html(fig)

    def _chart_block(title: str, explanation: str = "") -> str:
        """Render a chart container with optional explanation prose."""
        chart_html = figures_by_title.get(title)
        if chart_html is None:
            return ""
        expl_html = ""
        if explanation:
            expl_html = (
                '<p class="chart-explanation" style="color: #555; font-size: 0.92em;'
                ' line-height: 1.5; margin: 8px 0 16px 0; padding: 0 8px;">'
                f'{explanation}</p>'
            )
        return f"""
        <div class="chart-container">
            <h3>{title}</h3>{expl_html}
            {chart_html}
        </div>"""

    parts: list[str] = []

    # ------------------------------------------------------------------
    # Section 1: How This Experiment Works (workflow diagram)
    # ------------------------------------------------------------------
    workflow_fig = _create_workflow_diagram()
    workflow_html = _fig_to_html(workflow_fig)

    parts.append(f"""
    <div class="card">
        <h2>How This Experiment Works</h2>
        <p>
            We start with <strong><a href="https://hotpotqa.github.io/" target="_blank"
            style="color: #648FFF;">HotpotQA</a></strong>, a dataset of real questions where the
            correct answers are already known. Each question comes with source documents
            and a verified "gold" answer. We feed the question and documents into a
            <strong>RAG pipeline</strong> (NaiveRAG + Qwen3 4B) to generate an answer,
            then measure that answer two ways:
        </p>
        <ol style="margin: 12px 0 12px 24px; line-height: 1.8;">
            <li><strong>Automated metrics</strong> compare the RAG answer to the gold answer
                using BERTScore (semantic similarity) and F1 (word overlap) &mdash; these are
                objective, but can't capture everything.</li>
            <li><strong>LLM judges</strong> (6 different cloud models) read the question,
                context, and RAG answer, then rate quality on faithfulness, relevance, and
                conciseness &mdash; without ever seeing the gold answer.</li>
        </ol>
        <p>
            The question this experiment answers: <strong>which judges' ratings actually
            track the objective metrics?</strong> If a judge says an answer is good, is it
            really good?
        </p>
        {workflow_html}
        <p style="margin-top: 16px; font-size: 0.9em;">
            <a href="raw_scores.csv" style="color: #648FFF;">Download the raw data (CSV)</a>
            to explore the full dataset yourself.
        </p>
    </div>
    """)

    # ------------------------------------------------------------------
    # Section 1b: Row Examiner — see the pipeline in action
    # ------------------------------------------------------------------
    if answers_df is not None:
        row_examiner_html = _build_row_examiner(df, answers_df)
        parts.append(f"""
    <div class="card">
        <h2>See It in Action</h2>
        <p>
            Pick any of the 50 questions below to walk through the full pipeline: the
            source document (with the gold answer highlighted if it appears), the question,
            what the RAG system answered, what the correct answer is, and how each judge
            scored it. Sort by score to find the best and worst examples.
        </p>
        {row_examiner_html}
    </div>
        """)

    # ------------------------------------------------------------------
    # Section 2: Why This Matters
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card">
        <h2>Why This Matters</h2>
        <p>
            Experiments 1 and 2 will test 30+ RAG configurations with 200 questions each
            &mdash; thousands of answers that need scoring. We can't check them all by hand.
            We need an automated judge we can trust. If the scorer is unreliable,
            <strong>every downstream result is noise</strong>.
        </p>
        <p>
            This experiment validates the scorer <em>before</em> we rely on it.
            It's the foundation the rest of the project stands on.
        </p>
    </div>
    """)

    # ------------------------------------------------------------------
    # Section 3: The Bottom Line (headline result + correlation chart)
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card">
        <h2>The Bottom Line</h2>
        <p>
            <strong>Claude Sonnet is the most accurate scorer.</strong> It tracks
            objective correctness better than any other judge we tested. At ~$0.005
            per call, scoring all of Experiments 1 and 2 (~9,200 answers) costs roughly
            $46 &mdash; a modest investment for the best available accuracy.
        </p>
        <p>
            For larger-scale experiments where cost matters more, <strong>Gemini 2.5
            Flash</strong> is an excellent budget alternative at $0.0001 per call (50&times;
            cheaper). It's not far behind Sonnet in accuracy and would cost under $1 for
            the same workload.
        </p>
        <p>
            The chart below shows <strong>Pearson correlation (r)</strong> between each
            judge's scores and two objective metrics. Pearson r measures how closely two
            sets of numbers move together: r&nbsp;=&nbsp;1.0 means perfect agreement,
            r&nbsp;=&nbsp;0 means no relationship. Higher bars mean the judge better
            tracks real answer quality.
        </p>
    </div>
    """)
    parts.append(_chart_block("Judge-Gold Correlation", """
        <strong>Key takeaway:</strong> Sonnet leads on both metrics
        (r&nbsp;=&nbsp;0.68 BERTScore, r&nbsp;=&nbsp;0.60 F1). Gemini 3.1 Pro is
        close (r&nbsp;=&nbsp;0.63, 0.52) but costs 2&times; more than Sonnet per call.
        Flash is slightly behind (r&nbsp;=&nbsp;0.60, 0.49) at 1/50th the cost.
        Flash-Lite and Haiku fall well short.
        <strong>Sonnet will be used for Experiments 1 &amp; 2.</strong>
    """))

    # ------------------------------------------------------------------
    # Section 4: Judge Comparison (the interesting charts, moved up)
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card" style="margin-top: 40px;">
        <h2>How the Judges Compare</h2>
        <p>
            The correlation chart above summarizes each judge in a single number. The
            charts below show the raw data behind those numbers &mdash; how each judge
            scores individual answers compared to objective truth.
        </p>
    </div>
    """)
    parts.append(_chart_block("Judge Quality vs BERTScore", """
        Each dot is one question. X-axis: how semantically similar the RAG answer is to the
        gold answer (BERTScore, 0&ndash;1). Y-axis: the judge's quality rating (1&ndash;5).
        A good judge's dots should trend upward &mdash; higher scores for better answers.
        Use the dropdown to compare specific judges.
        <br><br>
        <strong>Key takeaway:</strong> Claude Sonnet (r&nbsp;=&nbsp;0.68) and Gemini
        3.1 Pro (r&nbsp;=&nbsp;0.63) track BERTScore most closely. Flash-Lite's dots
        are scattered randomly (r&nbsp;=&nbsp;0.07) &mdash; it can't tell good answers
        from bad.
    """))
    parts.append(_chart_block("Judge Quality vs Gold F1", """
        Same idea, but using word-overlap F1 instead of BERTScore. F1 is stricter &mdash;
        correct paraphrases score low because the words don't match literally. Judges that
        correlate with <em>both</em> metrics are tracking real quality, not just surface
        similarity.
        <br><br>
        <strong>Key takeaway:</strong> Same pattern &mdash; Sonnet (r&nbsp;=&nbsp;0.60)
        and Flash (r&nbsp;=&nbsp;0.49) track F1 well. Flash-Lite shows near-zero
        correlation (r&nbsp;=&nbsp;0.02).
    """))
    parts.append(_chart_block("Correct vs Incorrect Scores", """
        Answers split into "correct" (exact match with gold) and "incorrect." A good judge
        should score correct answers meaningfully higher.
        <br><br>
        <strong>Key takeaway:</strong> 74% of answers were exact matches, so there are
        only 13 "incorrect" examples. Still, most judges do score correct answers higher,
        confirming they detect real quality differences.
    """))

    # ------------------------------------------------------------------
    # Section 5: How We Measured — BERTScore and F1 explained in context
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card" style="margin-top: 40px;">
        <h2>How We Measured "Correctness"</h2>
        <p>
            The charts above keep referring to "BERTScore" and "F1" &mdash; these are the
            two automated metrics we use to measure how close a RAG answer is to the
            known-correct gold answer. They measure different things:
        </p>
        <ul style="margin: 12px 0 12px 24px; line-height: 1.8;">
            <li><strong>BERTScore (0&ndash;1):</strong> Uses a neural language model to
                measure <em>semantic</em> similarity. "Steve McQueen" and 'Terence Steven
                "Steve" McQueen' score high because they mean the same thing. This is our
                <strong>primary metric</strong>.</li>
            <li><strong>F1 (0&ndash;1):</strong> Counts shared <em>words</em> between the
                RAG answer and the gold answer. Strict and literal &mdash; penalizes correct
                paraphrases. Used as a <strong>secondary check</strong> to confirm BERTScore
                isn't being fooled.</li>
        </ul>
        <p>
            <strong>Why two metrics?</strong> If both agree, we're confident. If they
            diverge, the model is paraphrasing (high BERTScore, low F1) &mdash; which is
            fine, but worth knowing.
        </p>
    </div>
    """)
    parts.append(_chart_block("BERTScore vs F1", """
        Each dot is one RAG answer. Upper-right: both literally and semantically correct.
        Upper-left: correct paraphrase (high BERTScore, low F1). The cluster in the
        upper-left shows the model frequently gives correct answers in different words.
        <strong>This validates using BERTScore over F1 as the primary measure.</strong>
    """))
    parts.append(_chart_block("BERTScore Distribution", """
        Distribution of BERTScores across all 50 answers. Values cluster high
        (0.8&ndash;1.0) because even mediocre answers share some meaning with the gold
        answer, but the differences in this range are still meaningful. Median: 0.986,
        mean: 0.931 &mdash; Qwen3 4B + NaiveRAG produces semantically strong answers.
    """))
    parts.append(_chart_block("F1 Distribution", """
        Distribution of word-overlap F1. Mean F1 is 0.611 &mdash; much lower than the
        BERTScore mean of 0.931. This gap confirms the model paraphrases frequently,
        which is why BERTScore is the better quality signal.
    """))

    # ------------------------------------------------------------------
    # Section 6: Deep Dive (detailed charts for those who want more)
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card" style="margin-top: 40px;">
        <h2>Deep Dive: Judge Behavior</h2>
        <p>
            The sections above cover the key results. Below is a deeper look at how each
            judge behaves &mdash; scoring patterns, biases, agreement between judges, and
            effects of answer/question length.
        </p>
    </div>
    """)
    parts.append(_chart_block("Score Distributions", """
        Violin plots of each judge's score distribution. A judge that gives everything
        a 5 isn't discriminating &mdash; it's rubber-stamping. The ideal scorer uses
        the full 1&ndash;5 range.
        <br><br>
        <strong>Key takeaway:</strong> Claude Opus has the narrowest range (3&ndash;5,
        std&nbsp;=&nbsp;0.50) &mdash; the most lenient. Flash and Flash-Lite use the
        full 1&ndash;5 range, making them better at separating quality levels.
    """))
    parts.append(_chart_block("Score Heatmap", """
        Every cell is one judge scoring one question. Vertical stripes of similar color
        mean judges agree; scattered colors mean disagreement.
        <br><br>
        <strong>Key takeaway:</strong> Most rows are consistently dark (high scores),
        reflecting that Qwen3 4B answered most questions well. The few light rows are
        consistent across judges &mdash; genuinely bad answers.
    """))
    parts.append(_chart_block("Metric Breakdown", """
        Quality decomposed into faithfulness, relevance, and conciseness for each judge.
        <br><br>
        <strong>Key takeaway:</strong> All judges rate faithfulness and relevance higher
        than conciseness, suggesting the model gives correct but somewhat verbose answers.
    """))
    parts.append(_chart_block("Inter-Judge Correlation", """
        Pearson correlation between every pair of judges. High agreement (r &gt; 0.6)
        suggests they're measuring something real, not random noise.
        <br><br>
        <strong>Key takeaway:</strong> Flash and Gemini 3.1 Pro are nearly identical
        (r&nbsp;=&nbsp;0.96). Cross-provider agreement is moderate: Flash vs Sonnet
        (r&nbsp;=&nbsp;0.63). Flash-Lite is the outlier, agreeing weakly with everyone.
    """))
    parts.append(_chart_block("Score vs Answer Length", """
        Whether longer answers systematically receive higher or lower scores. A strong
        correlation could mean the judge rewards verbosity rather than quality.
        <br><br>
        <strong>Key takeaway:</strong> All judges penalize longer answers
        (r&nbsp;=&nbsp;&minus;0.39 to &minus;0.73). This is appropriate &mdash; gold
        answers average 16 characters, so verbose RAG answers are genuinely lower quality.
    """))
    parts.append(_chart_block("Score vs Question Length", """
        Whether harder (longer) questions tend to receive lower scores.
        <br><br>
        <strong>Key takeaway:</strong> Almost no effect (r&nbsp;&lt;&nbsp;0.15 for all
        judges). The model handles long and short questions roughly equally well.
    """))
    parts.append(_chart_block("Question Length Distribution", """
        Questions range from 48 to 254 characters (median 94) &mdash; a reasonable spread
        of complexity. The sample isn't skewed toward trivially short or unusually long
        questions.
    """))
    parts.append(_chart_block("Answer Length Comparison", """
        Gold answers are terse (median 14 chars). RAG answers average 189 chars but have
        a median of only 19 &mdash; most are concise but a few verbose outliers drive the
        length-vs-score penalty seen above.
    """))

    # ------------------------------------------------------------------
    # Section 7: Conclusions
    # ------------------------------------------------------------------
    parts.append("""
    <div class="card" style="margin-top: 40px;">
        <h2>Conclusions</h2>
        <p>
            <strong>Best scorer: Claude Sonnet.</strong>
            Highest correlation with both BERTScore (r&nbsp;=&nbsp;0.68) and
            F1 (r&nbsp;=&nbsp;0.60), good discrimination across the full scoring range.
            At ~$0.005 per call, scoring Experiments 1 and 2 (~9,200 answers) will cost
            roughly $46.
        </p>
        <p>
            <strong>Budget alternative: Gemini 2.5 Flash.</strong>
            Nearly as accurate (BERTScore r&nbsp;=&nbsp;0.60, F1 r&nbsp;=&nbsp;0.49) at
            $0.0001 per call &mdash; 50&times; cheaper. For larger experiments or
            tighter budgets, Flash is an excellent choice that sacrifices little accuracy.
        </p>
        <p>
            <strong>Gemini 3.1 Pro is not cost-effective</strong> &mdash; it scores
            between Sonnet and Flash on accuracy but costs 2&times; more than Sonnet
            per call ($0.01 vs $0.005).
        </p>
        <p>
            <strong>Flash-Lite is unreliable</strong> &mdash; near-zero correlation
            with gold metrics despite similar average scores. It rates everything
            highly without distinguishing quality.
        </p>
        <p>
            <strong>Decision:</strong> Experiments 1 and 2 will use Claude Sonnet
            as the primary scorer for maximum accuracy.
        </p>
        <p style="margin-top: 16px; font-size: 0.9em;">
            <a href="raw_scores.csv" style="color: #648FFF;">Download the raw data (CSV)</a>
            &mdash; all 50 questions, 6 judges, and gold metrics in one file.
        </p>
    </div>
    """)

    # ------------------------------------------------------------------
    # Lessons Learned
    # ------------------------------------------------------------------
    parts.append("""
    <h2>Lessons Learned</h2>
    <div class="card" style="border-left: 4px solid #FE6100;">
        <h3>What We Got Wrong in Experiment 0 (v1)</h3>
        <p>
            Experiment 0 was our first end-to-end pipeline run. It answered the question
            it was designed to answer &mdash; which scorer to trust &mdash; but post-analysis
            revealed five methodological oversights that we are addressing in
            <strong>Experiment 0 v2</strong>.
        </p>

        <h3 style="margin-top: 20px;">1. We didn't capture what the LLM saw</h3>
        <p>
            The pipeline generated answers but didn't record which chunks were retrieved,
            what context was assembled, or what the LLM actually received as input. When
            the Church of St. Anne question failed (example&nbsp;5), we could see the answer
            was wrong but couldn't determine <em>why</em> &mdash; was it a retrieval miss,
            a chunking problem, or a generation error? Without pipeline observability,
            failure analysis is guesswork.
        </p>
        <p>
            <strong>Fix:</strong> Pipeline diagnostics now capture retrieved chunks,
            filtered chunks, context sent to the LLM, and automatically attribute failures
            to the responsible pipeline stage (chunker, retrieval, filtering, or generation).
        </p>

        <h3 style="margin-top: 20px;">2. The scorer judged against information the LLM never received</h3>
        <p>
            When scoring each answer, we passed the <strong>full source document</strong>
            as context to the LLM judge. But the answering model only saw the
            <strong>retrieved chunks</strong> &mdash; a small subset of the document.
            This means the faithfulness score measured whether the answer was consistent
            with the entire document, not with what the model actually had access to.
            A hallucinated detail that happened to appear elsewhere in the document
            would score as &ldquo;faithful.&rdquo;
        </p>
        <p>
            <strong>Fix:</strong> v2 passes the actual context sent to the LLM
            (<code>context_sent_to_llm</code>) to the scorer, so faithfulness is
            evaluated against what the model truly saw.
        </p>

        <h3 style="margin-top: 20px;">3. No reranker in the pipeline</h3>
        <p>
            The v1 pipeline used raw hybrid retrieval (dense + BM25 with RRF fusion)
            without a cross-encoder reranker. This is a weaker pipeline than what
            Experiments 1 and 2 will use, which means we validated our scorers on
            a different pipeline configuration than the one they'll actually score.
            Reranking improves retrieval precision and changes the distribution of
            answer quality &mdash; scorer validation should reflect the real pipeline.
        </p>
        <p>
            <strong>Fix:</strong> v2 adds BGE Reranker v2 M3 (568M parameters) as the
            default reranker. Retrieve 10 candidates, rerank down to 3.
        </p>

        <h3 style="margin-top: 20px;">4. Ceiling effect &mdash; too many easy questions</h3>
        <p>
            With 50 HotpotQA questions sampled proportionally across difficulties,
            74% of answers were correct (exact match). Most judges rated most answers
            5/5 &mdash; Flash-Lite gave a perfect 5.0 on 78% of examples, Opus on 78%.
            With only ~13 wrong answers, there wasn't enough signal to meaningfully
            compare how well judges discriminate between good and bad answers.
        </p>
        <p>
            <strong>Fix:</strong> v2 increases the sample to 150 questions and filters
            to <strong>medium and hard difficulty only</strong> (dropping easy questions
            entirely). This produces more wrong and partial answers, giving us real
            statistical power to compare scorer discrimination.
        </p>

        <h3 style="margin-top: 20px;">5. No composite answer quality metric</h3>
        <p>
            We had three independent signals for answer correctness &mdash; BERTScore
            (semantic similarity to the gold answer), word-overlap F1, and LLM judge
            scores &mdash; but no way to ask: &ldquo;do all three agree this answer is
            good?&rdquo; A judge that gives 5/5 to an answer with low BERTScore and
            low F1 has a blind spot. Example&nbsp;31 illustrates this perfectly: the RAG
            answer was &ldquo;Not specified in the context&rdquo; (a polite refusal),
            Flash and Opus both gave it 5/5 for faithfulness (it <em>was</em> faithful
            to the empty context), but the gold answer was &ldquo;El Alma Argentina&rdquo;
            &mdash; a complete miss.
        </p>
        <p>
            <strong>Fix:</strong> v2 adds an <code>answer_quality</code> column that
            requires agreement across all three metrics: BERTScore &ge; 0.90,
            word-overlap F1 &ge; 0.50, <em>and</em> Sonnet quality &ge; 4.0.
            An answer is only &ldquo;good&rdquo; if the gold metrics and the best
            judge all agree. This triangulation exposes the blind spots that any
            single metric misses.
        </p>
    </div>
    <div class="card" style="border-left: 4px solid #648FFF; margin-top: 16px;">
        <p style="margin: 0;">
            <strong>Experiment 0 v2</strong> reruns this scorer validation with all
            five fixes in place. The v1 results above are preserved as-is &mdash; they
            are the baseline that motivated these improvements.
        </p>
    </div>
    """)

    content = "\n".join(parts)
    return _build_page_template(
        "Experiment 0: Scorer Validation",
        nav_active="exp0",
        content_html=content,
    )


# ---------------------------------------------------------------------------
# Experiment 1 dashboard
# ---------------------------------------------------------------------------

# Per-chart explanations for Experiment 1 (Strategy x Model Size)
_EXP1_EXPLANATIONS: dict[str, str] = {
    "Summary": """
        <strong>What this shows:</strong> Key statistics for the experiment &mdash;
        total configurations tested, best and worst performing config, and overall
        mean quality.
    """,
    "Quality Heatmap": """
        <strong>What this shows:</strong> Mean quality score for every strategy-model
        combination. Rows are RAG strategies, columns are models ordered by parameter
        count (smallest left, largest right). Brighter cells = higher quality.
        <br><br>
        <strong>How to read it:</strong> Look for whether the rightmost column
        (largest model) always dominates &mdash; if not, a smart strategy is
        compensating for model size. Also look for rows (strategies) that are
        consistently bright or dark across all models.
    """,
    "Latency Heatmap": """
        <strong>What this shows:</strong> Mean strategy latency (in seconds) for each
        configuration. Same layout as the quality heatmap for easy comparison.
        <br><br>
        <strong>How to read it:</strong> Compare this with the quality heatmap above.
        Is the highest-quality config also the slowest? Are there configs that achieve
        nearly the same quality at a fraction of the time?
    """,
    "Quality vs Model Size": """
        <strong>What this shows:</strong> Each line is one RAG strategy. The x-axis is
        model size (billions of parameters), the y-axis is mean quality. Error bars
        show standard deviation across individual questions.
        <br><br>
        <strong>How to read it:</strong> If all lines slope upward, bigger models always
        win. If a strategy line is flat or inverted, that strategy doesn't benefit from
        scale &mdash; or the small model is already good enough. Lines that <em>cross</em>
        are the most interesting: they show where strategy choice matters more than model size.
    """,
    "Latency vs Model Size": """
        <strong>What this shows:</strong> Same layout as quality vs model size, but y-axis
        is latency (log scale). Strategies with more LLM calls (MultiQuery, Corrective)
        should show steeper slopes.
        <br><br>
        <strong>How to read it:</strong> The gap between strategies at each model size
        shows the latency cost of "smarter" RAG. If a strategy doubles latency for
        minimal quality gain, that's a bad tradeoff.
    """,
    "Strategy Beats Size": """
        <strong>What this shows:</strong> The core research question &mdash; how often does
        strategy + small model beat NaiveRAG + larger model? Each bar shows how many
        such "upset" cases a strategy produces. The label shows the average quality
        advantage.
        <br><br>
        <strong>How to read it:</strong> Tall bars mean the strategy consistently
        compensates for model size. A strategy with zero upsets isn't worth the
        complexity over simply using a bigger model with NaiveRAG.
    """,
    "Per-Metric Breakdown": """
        <strong>What this shows:</strong> Quality decomposed into three dimensions
        (faithfulness, relevance, conciseness) for the best and worst configurations.
        <br><br>
        <strong>How to read it:</strong> If the worst configs fail primarily on one
        dimension (e.g., faithfulness), that tells you which aspect of the RAG pipeline
        breaks down &mdash; retrieval quality vs answer formulation vs verbosity.
    """,
    "Score Distributions by Strategy": """
        <strong>What this shows:</strong> Violin plots of quality scores for each strategy
        across all models and questions. The shape shows where scores concentrate.
        <br><br>
        <strong>How to read it:</strong> A tall, narrow violin centered high means
        consistently good. A wide, spread violin means the strategy is unreliable &mdash;
        sometimes excellent, sometimes terrible.
    """,
    "Score Distributions by Model": """
        <strong>What this shows:</strong> Same as above but grouped by model instead of
        strategy. Reveals whether model size gives more consistent (narrower) or just
        higher-mean results.
        <br><br>
        <strong>How to read it:</strong> If larger models have narrower distributions,
        they're more reliable, not just better on average. If the spread is similar
        across sizes, reliability comes from strategy, not scale.
    """,
    "Gold Metrics Heatmap": """
        <strong>What this shows:</strong> Gold F1 (word-overlap with the known-correct
        answer) for each configuration. This is an objective measure independent of the
        LLM judge.
        <br><br>
        <strong>How to read it:</strong> Compare with the quality heatmap. If the
        patterns match, the judge is tracking real correctness. If they diverge, the
        judge may be rewarding style over substance.
    """,
    "Quality vs Latency (Pareto)": """
        <strong>What this shows:</strong> Every dot is one configuration. X-axis is
        latency, y-axis is quality. The dashed line connects Pareto-optimal configs
        &mdash; those where no other config is both faster AND better.
        <br><br>
        <strong>How to read it:</strong> Points on or near the frontier are the only
        rational choices. Points well below the frontier are dominated &mdash; another
        config is both faster and better. The shape of the frontier shows the
        quality-speed tradeoff curve.
    """,
    "Per-Query Detail": """
        <strong>What this shows:</strong> The 10 worst and 10 best individual answers
        across all configurations. Reveals what kinds of questions the pipeline handles
        well vs. poorly.
        <br><br>
        <strong>How to read it:</strong> Look for patterns in the worst answers &mdash;
        are they all from one strategy, one model, or one type of question?
    """,
}


def _generate_experiment_1(csv_path: Path) -> str:
    """Build the Experiment 1 dashboard page with explanatory prose.

    Args:
        csv_path: Path to ``results/experiment_1/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    from scripts.generate_experiment1_dashboard import build_experiment1_figures
    from scripts.generate_experiment0_dashboard import _fig_to_html

    figures = build_experiment1_figures(csv_path)

    parts = []
    parts.append("""
    <div class="card">
        <h2>What This Experiment Tests</h2>
        <p>
            <strong>Does a smarter RAG strategy compensate for a smaller language model?</strong>
            We test 5 RAG strategies (NaiveRAG, SelfRAG, CorrectiveRAG, AdaptiveRAG,
            MultiQueryRAG) across 6 models ranging from 0.6B to 8B parameters.
            All other variables are held constant: Recursive chunker (500/100),
            mxbai-embed-large embedder, hybrid retrieval.
        </p>
        <p>
            200 HotpotQA questions are scored by Gemini 2.5 Flash (validated in Experiment 0).
            The key question: when does investing in a complex strategy pay off vs. just
            using a bigger model with simple NaiveRAG?
        </p>
    </div>
    """)

    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        explanation = _EXP1_EXPLANATIONS.get(title, "")
        explanation_html = ""
        if explanation:
            explanation_html = f"""
            <p class="chart-explanation" style="color: #555; font-size: 0.92em;
               line-height: 1.5; margin: 8px 0 16px 0; padding: 0 8px;">
                {explanation}
            </p>"""
        parts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>{explanation_html}
            {chart_html}
        </div>""")

    content = "\n".join(parts)
    return _build_page_template(
        "Experiment 1: Strategy × Model Size",
        nav_active="exp1",
        content_html=content,
    )


# ---------------------------------------------------------------------------
# Experiment 2 dashboard
# ---------------------------------------------------------------------------

# Per-chart explanations for Experiment 2 (Chunking x Model Size)
_EXP2_EXPLANATIONS: dict[str, str] = {
    "Summary": """
        <strong>What this shows:</strong> Key statistics for the experiment &mdash;
        total configurations tested, best and worst performing config, and overall
        mean quality.
    """,
    "Quality Heatmap": """
        <strong>What this shows:</strong> Mean quality score for every chunker-model
        combination. Rows are chunking strategies, columns are Qwen3 models ordered
        by parameter count. Brighter cells = higher quality.
        <br><br>
        <strong>How to read it:</strong> Look for whether chunker choice matters
        at all &mdash; if all rows look the same, chunking doesn't matter much.
        If one row is consistently brighter, that chunker is the clear winner.
    """,
    "Latency Heatmap": """
        <strong>What this shows:</strong> Mean strategy latency for each configuration.
        Chunking itself is fast, but different chunk sizes affect retrieval and
        generation time.
        <br><br>
        <strong>How to read it:</strong> Semantic chunking may be slower (requires
        embedding each chunk boundary). If its latency is high but quality is only
        marginally better, it's not worth the cost.
    """,
    "Quality vs Model Size": """
        <strong>What this shows:</strong> Each line is one chunking strategy.
        X-axis is model size, y-axis is mean quality with error bars.
        <br><br>
        <strong>How to read it:</strong> If the lines are nearly parallel, chunking
        choice doesn't interact with model size &mdash; the same chunker wins everywhere.
        If lines cross, the optimal chunker depends on model size, which is a more
        interesting finding.
    """,
    "Latency vs Model Size": """
        <strong>What this shows:</strong> Latency by chunker and model size (log scale).
        <br><br>
        <strong>How to read it:</strong> Since all configs use NaiveRAG (same strategy),
        latency differences come from chunk count affecting retrieval and context length
        affecting generation time.
    """,
    "Chunking Impact Analysis": """
        <strong>What this shows:</strong> For each model, the quality gap between the
        best and worst chunker. Labels show which chunker won and which lost.
        <br><br>
        <strong>How to read it:</strong> Tall bars mean chunking choice matters a lot
        for that model. If bars are short across all models, chunking is a minor
        variable &mdash; effort is better spent on strategy or model selection.
    """,
    "Per-Metric Breakdown": """
        <strong>What this shows:</strong> Quality decomposed into faithfulness,
        relevance, and conciseness for all 16 configurations, ordered by overall
        quality.
        <br><br>
        <strong>How to read it:</strong> Does chunking primarily affect faithfulness
        (getting the right context) or conciseness (answer verbosity)? Relevance
        should be relatively stable if questions are well-formed.
    """,
    "Score Distributions by Chunker": """
        <strong>What this shows:</strong> Violin plots of quality for each chunker
        across all models and questions.
        <br><br>
        <strong>How to read it:</strong> A chunker with a narrow, high violin is
        both good and reliable. Wide spread means inconsistent &mdash; it helps
        some queries but hurts others.
    """,
    "Score Distributions by Model": """
        <strong>What this shows:</strong> Violin plots grouped by model. Since strategy
        is held constant (NaiveRAG), this isolates the pure effect of model size.
        <br><br>
        <strong>How to read it:</strong> Compare the shapes, not just the means.
        Does a larger model reduce the "tail" of bad answers, or does it just
        shift the whole distribution up?
    """,
    "Gold Metrics Heatmap": """
        <strong>What this shows:</strong> Gold F1 for each configuration &mdash;
        the objective correctness measure independent of the LLM judge.
        <br><br>
        <strong>How to read it:</strong> Same pattern as quality heatmap? Good &mdash;
        the judge agrees with ground truth. Different pattern? Investigate why.
    """,
    "Quality vs Latency (Pareto)": """
        <strong>What this shows:</strong> Each dot is one chunker-model config.
        The Pareto frontier connects configs where no other is both faster and better.
        <br><br>
        <strong>How to read it:</strong> Configs below the frontier are dominated.
        With only 16 configs, the frontier shape reveals whether bigger chunks
        (fewer, faster) or smaller chunks (more, better retrieval) win the
        speed/quality tradeoff.
    """,
    "Chunk Count Analysis": """
        <strong>What this shows:</strong> Mean number of chunks produced vs. mean
        quality, colored by chunker. Models within each chunker appear as separate points.
        <br><br>
        <strong>How to read it:</strong> Is there an optimal chunk count? Too few
        means important context is missed. Too many means the model drowns in
        irrelevant text. The "sweet spot" is where the quality peaks.
    """,
}


def _generate_experiment_2(csv_path: Path) -> str:
    """Build the Experiment 2 dashboard page with explanatory prose.

    Args:
        csv_path: Path to ``results/experiment_2/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    from scripts.generate_experiment2_dashboard import build_experiment2_figures
    from scripts.generate_experiment0_dashboard import _fig_to_html

    figures = build_experiment2_figures(csv_path)

    parts = []
    parts.append("""
    <div class="card">
        <h2>What This Experiment Tests</h2>
        <p>
            <strong>Does how you split documents into chunks affect answer quality,
            and does it interact with model size?</strong> We test 4 chunking strategies
            (Fixed 512, Recursive 500/100, Sentence, Semantic) across 4 Qwen3 models
            (0.6B to 8B). Strategy is held constant at NaiveRAG to isolate the
            chunking variable.
        </p>
        <p>
            This is an understudied question &mdash; most RAG research treats chunking
            as a fixed preprocessing step. We test whether it deserves the same attention
            as strategy and model selection.
        </p>
    </div>
    """)

    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        explanation = _EXP2_EXPLANATIONS.get(title, "")
        explanation_html = ""
        if explanation:
            explanation_html = f"""
            <p class="chart-explanation" style="color: #555; font-size: 0.92em;
               line-height: 1.5; margin: 8px 0 16px 0; padding: 0 8px;">
                {explanation}
            </p>"""
        parts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>{explanation_html}
            {chart_html}
        </div>""")

    content = "\n".join(parts)
    return _build_page_template(
        "Experiment 2: Chunking × Model Size",
        nav_active="exp2",
        content_html=content,
    )


# ---------------------------------------------------------------------------
# Placeholder pages
# ---------------------------------------------------------------------------

def _generate_placeholder(
    experiment_num: int,
    description: str,
) -> str:
    """Build a placeholder page for an experiment that hasn't run yet.

    Args:
        experiment_num: Experiment number (1 or 2).
        description: Text describing the planned experiment.

    Returns:
        Full HTML page string.
    """
    nav_key = f"exp{experiment_num}"
    content = f"""
    <div class="placeholder">
        <h2>Coming Soon</h2>
        <p>
            Experiment {experiment_num} has not yet been completed.
            When results are available, this page will automatically
            show interactive visualizations.
        </p>
        <div class="card" style="text-align: left; max-width: 600px; margin: 24px auto;">
            <h3>Planned Experiment</h3>
            <p>{description}</p>
        </div>
    </div>
    """
    return _build_page_template(
        f"Experiment {experiment_num}",
        nav_active=nav_key,
        content_html=content,
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def main(
    results_dir: Path | None = None,
    output_dir: Path | None = None,
    experiments: list[int] | None = None,
) -> None:
    """Generate the complete findings gallery static site.

    Discovers available experiment results in *results_dir* and generates
    HTML pages in *output_dir*.

    Args:
        results_dir: Directory containing ``experiment_0/``, etc.
            Defaults to ``results/`` in the project root.
        output_dir: Output directory for HTML files.
            Defaults to ``site/``.
        experiments: List of experiment numbers to generate.
            Defaults to ``[0, 1, 2]``.
    """
    if results_dir is None:
        results_dir = _PROJECT_ROOT / "results"
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = _PROJECT_ROOT / "site"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiments is None:
        experiments = [0, 1, 2]

    # Discover available experiment data
    experiments_info: list[dict[str, Any]] = []

    # Experiment 0 — supports both v1 and v2 results
    if 0 in experiments:
        import shutil

        exp0_v1_csv = results_dir / "experiment_0" / "raw_scores.csv"
        exp0_v2_csv = results_dir / "experiment_0_v2" / "raw_scores.csv"
        has_v1 = exp0_v1_csv.exists() and exp0_v1_csv.stat().st_size > 0
        has_v2 = exp0_v2_csv.exists() and exp0_v2_csv.stat().st_size > 0

        if has_v1 or has_v2:
            desc_parts = []
            if has_v1:
                desc_parts.append("v1: 50 HotpotQA × NaiveRAG × Qwen3 4B")
            if has_v2:
                desc_parts.append("v2: 150 medium+hard × BGE reranker × diagnostics")
            experiments_info.append({
                "num": 0,
                "title": "Scorer Validation",
                "status": "ready",
                "description": "; ".join(desc_parts),
            })

            # Build the page content — v1 first, then v2 below
            page_parts = []

            if has_v1:
                logger.info("Generating Experiment 0 v1 dashboard from %s", exp0_v1_csv)
                exp0_v1_html = _generate_experiment_0(exp0_v1_csv)
                # Copy raw CSV for download link
                shutil.copy2(exp0_v1_csv, output_dir / "raw_scores.csv")

                if has_v2:
                    # Both versions: wrap v1 content and add v2 below
                    (output_dir / "experiment_0.html").write_text(
                        exp0_v1_html, encoding="utf-8",
                    )
                else:
                    (output_dir / "experiment_0.html").write_text(
                        exp0_v1_html, encoding="utf-8",
                    )

            if has_v2:
                logger.info("Generating Experiment 0 v2 dashboard from %s", exp0_v2_csv)
                try:
                    exp0_v2_html = _generate_experiment_0_v2(exp0_v2_csv)
                    (output_dir / "experiment_0_v2.html").write_text(
                        exp0_v2_html, encoding="utf-8",
                    )
                    shutil.copy2(exp0_v2_csv, output_dir / "raw_scores_v2.csv")
                    # Update experiments_info for navigation
                    experiments_info.append({
                        "num": "0v2",
                        "title": "Scorer Validation v2",
                        "status": "ready",
                        "description": "150 medium+hard HotpotQA × NaiveRAG + BGE reranker × diagnostics + answer quality.",
                    })
                except Exception as exc:
                    logger.warning("Experiment 0 v2 dashboard generation failed: %s", exc)

            if not has_v1:
                # Only v2 exists — make it the main page
                shutil.copy2(
                    output_dir / "experiment_0_v2.html",
                    output_dir / "experiment_0.html",
                )
        else:
            print(f"WARNING: No Experiment 0 data found — generating placeholder")
            experiments_info.append({
                "num": 0,
                "title": "Scorer Validation",
                "status": "placeholder",
                "description": "50 HotpotQA × NaiveRAG × Qwen3 4B, scored by 6 LLM judges.",
            })
            placeholder = _generate_placeholder(0, "Scorer validation — comparing LLM judges on gold-standard data.")
            (output_dir / "experiment_0.html").write_text(placeholder, encoding="utf-8")

    # Experiments 1 and 2 — use dedicated generators with prose when data exists
    _exp_titles = {1: "Strategy × Model Size", 2: "Chunking × Model Size"}
    _exp_page_generators = {1: _generate_experiment_1, 2: _generate_experiment_2}

    for exp_num in [1, 2]:
        if exp_num not in experiments:
            continue
        exp_csv = results_dir / f"experiment_{exp_num}" / "raw_scores.csv"
        desc = _EXPERIMENT_DESCRIPTIONS.get(exp_num, f"Experiment {exp_num}")

        if exp_csv.exists() and exp_csv.stat().st_size > 0:
            experiments_info.append({
                "num": exp_num,
                "title": _exp_titles.get(exp_num, f"Experiment {exp_num}"),
                "status": "ready",
                "description": desc,
            })
            logger.info("Generating Experiment %d dashboard from %s", exp_num, exp_csv)
            try:
                page_html = _exp_page_generators[exp_num](exp_csv)
                (output_dir / f"experiment_{exp_num}.html").write_text(page_html, encoding="utf-8")
            except Exception as exc:
                logger.warning("Experiment %d dashboard generation failed: %s — using placeholder", exp_num, exc)
                placeholder = _generate_placeholder(exp_num, desc)
                (output_dir / f"experiment_{exp_num}.html").write_text(placeholder, encoding="utf-8")
        else:
            experiments_info.append({
                "num": exp_num,
                "title": _exp_titles.get(exp_num, f"Experiment {exp_num}"),
                "status": "placeholder",
                "description": desc,
            })
            placeholder = _generate_placeholder(exp_num, desc)
            (output_dir / f"experiment_{exp_num}.html").write_text(placeholder, encoding="utf-8")

    # Generate index page
    index_html = _build_page_template(
        "RAGBench Findings Gallery",
        nav_active="home",
        content_html=_generate_index(experiments_info),
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    logger.info("Gallery generated: %d pages in %s", len(experiments) + 1, output_dir)
    print(f"Gallery generated in {output_dir}/ ({len(experiments) + 1} pages)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAGBench findings gallery")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: site/)",
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment numbers to generate (default: 0,1,2)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output) if args.output else None
    experiments = (
        [int(x.strip()) for x in args.experiments.split(",")]
        if args.experiments
        else None
    )

    main(output_dir=output_dir, experiments=experiments)
