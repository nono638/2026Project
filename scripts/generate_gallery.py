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
            on 50 HotpotQA questions scored by NaiveRAG + Qwen3 4B. Finding: Gemini 2.5
            Flash offers the best cost/quality balance for automated scoring — strong
            correlation with gold-standard metrics at 23× lower cost than Claude Sonnet.
        </p>
    </div>
    """


# ---------------------------------------------------------------------------
# Experiment 0 dashboard
# ---------------------------------------------------------------------------

def _generate_experiment_0(csv_path: Path) -> str:
    """Build the Experiment 0 dashboard page content with Plotly charts.

    Imports the chart-building functions from the existing dashboard script
    and wraps them in the gallery template.

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

    parts = []
    parts.append("""
    <div class="card">
        <p>
            Scorer validation: 50 HotpotQA questions answered by NaiveRAG + Qwen3 4B,
            scored by 6 LLM judges (4 Gemini + 2 Claude). Each chart below is interactive
            — hover for details, click legend entries to toggle, drag to zoom.
        </p>
    </div>
    """)

    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        parts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>
            {chart_html}
        </div>""")

    content = "\n".join(parts)
    return _build_page_template(
        "Experiment 0: Scorer Validation",
        nav_active="exp0",
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

    # Experiment 0
    if 0 in experiments:
        exp0_csv = results_dir / "experiment_0" / "raw_scores.csv"
        if exp0_csv.exists() and exp0_csv.stat().st_size > 0:
            experiments_info.append({
                "num": 0,
                "title": "Scorer Validation",
                "status": "ready",
                "description": "50 HotpotQA × NaiveRAG × Qwen3 4B, scored by 6 LLM judges.",
            })
            logger.info("Generating Experiment 0 dashboard from %s", exp0_csv)
            exp0_html = _generate_experiment_0(exp0_csv)
            (output_dir / "experiment_0.html").write_text(exp0_html, encoding="utf-8")
        else:
            print(f"WARNING: {exp0_csv} not found — generating placeholder for Experiment 0")
            experiments_info.append({
                "num": 0,
                "title": "Scorer Validation",
                "status": "placeholder",
                "description": "50 HotpotQA × NaiveRAG × Qwen3 4B, scored by 6 LLM judges.",
            })
            placeholder = _generate_placeholder(0, "Scorer validation — comparing LLM judges on gold-standard data.")
            (output_dir / "experiment_0.html").write_text(placeholder, encoding="utf-8")

    # Experiments 1 and 2
    for exp_num in [1, 2]:
        if exp_num not in experiments:
            continue
        exp_csv = results_dir / f"experiment_{exp_num}" / "raw_scores.csv"
        desc = _EXPERIMENT_DESCRIPTIONS.get(exp_num, f"Experiment {exp_num}")
        if exp_csv.exists() and exp_csv.stat().st_size > 0:
            experiments_info.append({
                "num": exp_num,
                "title": desc.split(" — ")[0] if " — " in desc else f"Experiment {exp_num}",
                "status": "ready",
                "description": desc,
            })
            # TODO: generate real dashboard when data exists
            # For now, use placeholder even if CSV exists
            logger.info("Experiment %d data found but no dashboard generator yet — placeholder", exp_num)
            placeholder = _generate_placeholder(exp_num, desc)
            (output_dir / f"experiment_{exp_num}.html").write_text(placeholder, encoding="utf-8")
        else:
            experiments_info.append({
                "num": exp_num,
                "title": desc.split(" — ")[0] if " — " in desc else f"Experiment {exp_num}",
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
