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
    and wraps them in the gallery template.  Adds explanatory prose around
    each chart so readers understand what they're looking at.

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

    # Per-chart explanations keyed by chart title.  Charts whose title
    # doesn't appear here get no extra prose (just the chart).
    chart_explanations: dict[str, str] = {
        "Judge Quality vs BERTScore": """
            <strong>What this shows:</strong> Each dot is one question. The x-axis is
            BERTScore (how semantically similar the RAG answer is to the known-correct
            answer, 0&ndash;1). The y-axis is the judge's quality score (1&ndash;5).
            A good judge should give higher scores to answers that are actually more
            correct &mdash; meaning the dots should trend upward from left to right.
            <br><br>
            <strong>How to read it:</strong> What matters is how tightly the dots cluster
            around the trendline, not the slope itself. A tight cluster (high r) means the
            judge reliably tracks quality; scattered dots (low r) mean it's guessing.
            Use the dropdown above the chart to compare specific judges, or click legend
            entries to show/hide them.
            <br><br>
            <strong>Key takeaway:</strong> Claude Sonnet (r&nbsp;=&nbsp;0.68) and Gemini
            3.1 Pro (r&nbsp;=&nbsp;0.63) track BERTScore most closely &mdash; their dots
            cluster tightly around the trendline. Flash-Lite's dots are scattered randomly
            (r&nbsp;=&nbsp;0.07) &mdash; it can't distinguish good answers from bad.
        """,
        "Judge Quality vs Gold F1": """
            <strong>What this shows:</strong> Same idea as above, but using word-overlap
            F1 instead of BERTScore. F1 measures how many words the RAG answer shares with
            the gold answer (0&ndash;1). It's a stricter, more literal measure &mdash;
            paraphrased answers score low on F1 even if semantically correct.
            <br><br>
            <strong>How to read it:</strong> Judges that correlate with both BERTScore
            <em>and</em> F1 are tracking real quality, not just surface similarity.
            Use the dropdown to compare specific judges side-by-side.
            <br><br>
            <strong>Key takeaway:</strong> The same pattern holds &mdash; Sonnet
            (r&nbsp;=&nbsp;0.60) and Flash (r&nbsp;=&nbsp;0.49) track F1 well.
            Flash-Lite again shows near-zero correlation (r&nbsp;=&nbsp;0.02).
        """,
        "Judge-Gold Correlation": """
            <strong>What this shows:</strong> A summary bar chart &mdash; Pearson
            correlation (r) between each judge's quality scores and the two gold metrics.
            Higher bars mean the judge better tracks objective correctness.
            <br><br>
            <strong>How to read it:</strong> This is the key chart for choosing a scorer.
            We want the cheapest judge with high correlation. An r above 0.5 indicates
            a meaningful relationship; above 0.7 is strong.
            <br><br>
            <strong>Key takeaway:</strong> Sonnet is the most accurate judge overall, but
            Flash is close behind at 1/50th the cost. Flash-Lite and Haiku fall well short.
            This chart drove the decision to use Flash for Experiments 1 &amp; 2.
        """,
        "Correct vs Incorrect Scores": """
            <strong>What this shows:</strong> Splits answers into "correct" (exact match
            with gold) and "incorrect," then compares the average judge score for each
            group. A good judge should score correct answers meaningfully higher.
            <br><br>
            <strong>How to read it:</strong> A large gap between the two bars means the
            judge distinguishes right from wrong. A small gap means it can't tell.
            <br><br>
            <strong>Key takeaway:</strong> 74% of answers were exact matches, so there
            are only 13 "incorrect" examples &mdash; a small group. Still, most judges
            do score correct answers higher, confirming they detect real quality differences.
        """,
        "Score Heatmap": """
            <strong>What this shows:</strong> Every cell is one judge scoring one question.
            Color intensity shows the quality score (darker = higher). Rows are questions,
            columns are judges.
            <br><br>
            <strong>How to read it:</strong> Vertical stripes of similar color mean judges
            agree. Scattered colors mean disagreement. Look for questions where judges
            wildly disagree &mdash; those reveal what kinds of answers are hard to evaluate.
            <br><br>
            <strong>Key takeaway:</strong> Most rows are consistently dark (high scores),
            reflecting that Qwen3 4B answered most questions well. The few light rows
            (low scores) tend to be consistent across judges &mdash; genuinely bad answers.
        """,
        "Score Distributions": """
            <strong>What this shows:</strong> Violin plots of each judge's score
            distribution across all 50 questions. The shape shows where scores cluster.
            <br><br>
            <strong>How to read it:</strong> A judge that gives everything a 5 isn't
            discriminating &mdash; it's rubber-stamping. A wide distribution using the
            full 1&ndash;5 range means the judge is actually evaluating. The ideal scorer
            uses the full range and concentrates mass where the true quality distribution is.
            <br><br>
            <strong>Key takeaway:</strong> Claude Opus has the narrowest range (3&ndash;5,
            std&nbsp;=&nbsp;0.50) &mdash; it's the most lenient. Flash and Flash-Lite use
            the full 1&ndash;5 range (std&nbsp;~0.85&ndash;0.95), making them better at
            separating quality levels.
        """,
        "Metric Breakdown": """
            <strong>What this shows:</strong> Each judge scores three sub-dimensions:
            faithfulness (does the answer match the retrieved context?), relevance
            (does it answer the question?), and conciseness (is it appropriately brief?).
            This breaks down the overall quality score into those components.
            <br><br>
            <strong>How to read it:</strong> If a judge rates everything high on
            faithfulness but low on conciseness, that tells you about its evaluation
            biases, not necessarily about the answers.
            <br><br>
            <strong>Key takeaway:</strong> All judges tend to rate faithfulness and
            relevance higher than conciseness, suggesting the model gives correct but
            somewhat verbose answers.
        """,
        "Score vs Answer Length": """
            <strong>What this shows:</strong> Whether longer RAG answers systematically
            receive higher or lower scores.
            <br><br>
            <strong>How to read it:</strong> A strong correlation here is a red flag &mdash;
            it could mean the judge rewards verbosity rather than quality. Ideally, length
            and score should be weakly related.
            <br><br>
            <strong>Key takeaway:</strong> All judges penalize longer answers
            (r&nbsp;=&nbsp;&minus;0.39 to &minus;0.73). This is actually appropriate here
            &mdash; the gold answers average 16 characters, so verbose RAG answers
            (mean 189 chars) are genuinely lower quality, not just wordier.
        """,
        "Score vs Question Length": """
            <strong>What this shows:</strong> Whether longer (typically harder) questions
            tend to receive lower scores.
            <br><br>
            <strong>How to read it:</strong> A downward trend is expected &mdash; harder
            questions are harder to answer well. But a very steep drop might mean the
            model struggles disproportionately with complex queries.
            <br><br>
            <strong>Key takeaway:</strong> Question length has almost no effect on scores
            (r&nbsp;&lt;&nbsp;0.15 for all judges). The model handles long and short
            questions roughly equally well.
        """,
        "Inter-Judge Correlation": """
            <strong>What this shows:</strong> Pearson correlation between every pair of
            judges. Values near 1.0 mean two judges rank answers the same way; near 0
            means they're unrelated.
            <br><br>
            <strong>How to read it:</strong> High inter-judge agreement (r &gt; 0.6)
            suggests the judges are measuring something real, not random noise. If two
            judges from different providers agree, that's especially meaningful.
            <br><br>
            <strong>Key takeaway:</strong> Flash and Gemini 3.1 Pro are nearly identical
            (r&nbsp;=&nbsp;0.96) &mdash; essentially the same scorer. Cross-provider
            agreement is moderate: Flash vs Sonnet (r&nbsp;=&nbsp;0.63), Flash vs Opus
            (r&nbsp;=&nbsp;0.67). Flash-Lite is the outlier, agreeing weakly with everyone.
        """,
        "BERTScore Distribution": """
            <strong>What this shows:</strong> The distribution of BERTScore F1 across
            all 50 RAG answers. BERTScore (0&ndash;1) uses neural embeddings to measure
            semantic similarity between the RAG answer and the known-correct answer.
            Values above 0.85 indicate strong semantic match.
            <br><br>
            <strong>Why the range looks narrow:</strong> BERTScore naturally clusters
            high (0.8&ndash;1.0) because even mediocre answers share some meaning with
            the gold answer. The differences in this range are still meaningful &mdash;
            0.85 vs 0.95 is a real quality gap.
            <br><br>
            <strong>Key takeaway:</strong> Median BERTScore is 0.986 and mean is 0.931,
            confirming that Qwen3 4B + NaiveRAG produces semantically strong answers on
            HotpotQA. The low outliers (below 0.85) are where the model genuinely
            struggled.
        """,
        "F1 Distribution": """
            <strong>What this shows:</strong> Distribution of word-overlap F1 scores.
            F1 (0&ndash;1) counts shared words between the RAG answer and the gold
            answer, penalizing both missing words and extra words.
            <br><br>
            <strong>How to read it:</strong> F1 is strict &mdash; a perfect paraphrase
            ("Steve McQueen" vs 'Terence Steven "Steve" McQueen') gets a low F1 despite
            being correct. That's why we use BERTScore as the primary gold metric.
            <br><br>
            <strong>Key takeaway:</strong> Mean F1 is 0.611 &mdash; much lower than the
            BERTScore mean of 0.931. This gap confirms the model paraphrases frequently
            rather than echoing gold wording, which is why BERTScore is the better
            quality signal.
        """,
        "BERTScore vs F1": """
            <strong>What this shows:</strong> The relationship between the two gold metrics
            themselves. Points in the upper-right are answers that are both literally and
            semantically correct. Points in the upper-left are correct paraphrases (high
            BERTScore, low F1).
            <br><br>
            <strong>How to read it:</strong> Divergence between the two metrics reveals
            how much paraphrasing the model does. A tight diagonal means it echoes the gold
            wording; spread means it paraphrases freely.
            <br><br>
            <strong>Key takeaway:</strong> The cluster in the upper-left (high BERTScore,
            variable F1) shows the model frequently gives correct answers in different words
            than the gold standard. This validates using BERTScore over F1 as the primary
            measure.
        """,
        "Question Length Distribution": """
            <strong>What this shows:</strong> How long the 50 test questions are.
            <br><br>
            <strong>Why it matters:</strong> Question length correlates with complexity.
            If the sample is skewed toward short, easy questions, the results may not
            generalize to harder queries.
            <br><br>
            <strong>Key takeaway:</strong> Questions range from 48 to 254 characters
            (median 94), giving a reasonable spread of complexity. The sample isn't
            dominated by trivially short or unusually long questions.
        """,
        "Answer Length Comparison": """
            <strong>What this shows:</strong> Side-by-side comparison of RAG answer length
            vs gold answer length.
            <br><br>
            <strong>How to read it:</strong> If RAG answers are consistently much longer
            than gold answers, the model is being verbose. If shorter, it may be
            truncating or losing information.
            <br><br>
            <strong>Key takeaway:</strong> Gold answers are terse (median 14 chars &mdash;
            typically a name or short phrase). RAG answers average 189 chars but have a
            median of only 19, meaning most answers are concise but a few are very verbose.
            Those verbose outliers are what drive the strong length-vs-score penalty
            seen earlier.
        """,
    }

    parts = []

    # --- Intro: what this experiment is and why it matters ---
    parts.append("""
    <div class="card">
        <h2>What This Experiment Tests</h2>
        <p>
            Before running thousands of RAG configurations in Experiments 1 and 2,
            we need to know: <strong>can an LLM reliably judge the quality of a RAG
            answer?</strong> If the scorer is unreliable, all downstream results are noise.
        </p>
        <p>
            We took 50 questions from HotpotQA (a dataset where the correct answers are
            known), generated RAG answers using NaiveRAG + Qwen3 4B, then asked 6 different
            LLM judges to score each answer. By comparing the judges' scores against the
            known-correct answers, we can measure which judges actually detect quality and
            which ones are just rubber-stamping everything as "good."
        </p>
    </div>
    """)

    # --- Key concepts: explain the scales ---
    parts.append("""
    <div class="card">
        <h2>Understanding the Metrics</h2>
        <p>The charts below use three types of scores on different scales:</p>
        <table class="data-table" style="max-width: 700px;">
            <tr>
                <th>Metric</th><th>Scale</th><th>What It Measures</th>
            </tr>
            <tr>
                <td><strong>Judge Quality Score</strong></td>
                <td>1&ndash;5</td>
                <td>An LLM judge reads the question, retrieved context, and RAG answer,
                    then rates quality on three dimensions (faithfulness, relevance,
                    conciseness). The average of these three is the quality score.
                    The judge does <em>not</em> see the correct answer.</td>
            </tr>
            <tr>
                <td><strong>BERTScore</strong></td>
                <td>0&ndash;1</td>
                <td>Semantic similarity between the RAG answer and the <em>known-correct</em>
                    ("gold") answer, computed by a neural language model. Values above 0.85
                    indicate strong match. This is the primary objective metric.</td>
            </tr>
            <tr>
                <td><strong>F1 (word overlap)</strong></td>
                <td>0&ndash;1</td>
                <td>How many words the RAG answer shares with the gold answer. Strict and
                    literal &mdash; penalizes correct paraphrases. Used as a secondary check.</td>
            </tr>
        </table>
        <p>
            <strong>What "gold" means:</strong> HotpotQA provides human-verified correct
            answers for every question. These are the "gold standard" &mdash; ground truth
            we can compare against. The whole point of this experiment is to test whether
            LLM judges agree with this ground truth.
        </p>
    </div>
    """)

    # --- Charts with per-chart explanations ---
    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        explanation = chart_explanations.get(title, "")
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

    # --- Conclusions ---
    parts.append("""
    <div class="card">
        <h2>Conclusions</h2>
        <p>
            <strong>Best cost/quality scorer: Gemini 2.5 Flash.</strong>
            It showed strong correlation with both BERTScore (r&nbsp;=&nbsp;0.60) and
            F1 (r&nbsp;=&nbsp;0.49), used the full 1&ndash;5 scoring range (good
            discrimination), and costs ~$0.0001 per call &mdash; 23&times; cheaper than
            Claude Sonnet.
        </p>
        <p>
            <strong>Claude Sonnet had the highest gold correlation</strong>
            (BERTScore r&nbsp;=&nbsp;0.68, F1 r&nbsp;=&nbsp;0.60) but at 50&times;
            the cost of Flash. For 2,000+ scoring calls in Experiments 1 and 2,
            the cost difference matters.
        </p>
        <p>
            <strong>Flash-Lite is unreliable</strong> &mdash; near-zero correlation
            with gold metrics despite similar average scores. It rates everything
            highly without distinguishing quality.
        </p>
        <p>
            <strong>Decision:</strong> Experiments 1 and 2 will use Gemini 2.5 Flash
            as the primary scorer, with the option to spot-check a sample with Sonnet.
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

    # Experiments 1 and 2 — use real dashboards when data exists
    _exp_generators = {}
    try:
        from scripts.generate_experiment1_dashboard import build_experiment1_figures
        _exp_generators[1] = build_experiment1_figures
    except ImportError:
        logger.warning("generate_experiment1_dashboard not importable — Exp 1 will use placeholder")
    try:
        from scripts.generate_experiment2_dashboard import build_experiment2_figures
        _exp_generators[2] = build_experiment2_figures
    except ImportError:
        logger.warning("generate_experiment2_dashboard not importable — Exp 2 will use placeholder")

    _exp_titles = {1: "Strategy × Model Size", 2: "Chunking × Model Size"}
    _exp_intros = {
        1: ("5 RAG strategies × 6 models. Each chart below is interactive "
            "— hover for details, click legend entries to toggle, drag to zoom."),
        2: ("4 chunking strategies × 4 Qwen3 models. Each chart below is interactive "
            "— hover for details, click legend entries to toggle, drag to zoom."),
    }

    for exp_num in [1, 2]:
        if exp_num not in experiments:
            continue
        exp_csv = results_dir / f"experiment_{exp_num}" / "raw_scores.csv"
        desc = _EXPERIMENT_DESCRIPTIONS.get(exp_num, f"Experiment {exp_num}")
        nav_key = f"exp{exp_num}"

        if exp_csv.exists() and exp_csv.stat().st_size > 0 and exp_num in _exp_generators:
            experiments_info.append({
                "num": exp_num,
                "title": _exp_titles.get(exp_num, f"Experiment {exp_num}"),
                "status": "ready",
                "description": desc,
            })
            logger.info("Generating Experiment %d dashboard from %s", exp_num, exp_csv)
            figures = _exp_generators[exp_num](exp_csv)
            parts = [f"""
    <div class="card">
        <p>{_exp_intros.get(exp_num, '')}</p>
    </div>
    """]
            from scripts.generate_experiment0_dashboard import _fig_to_html
            for title, fig in figures:
                chart_html = _fig_to_html(fig)
                parts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>
            {chart_html}
        </div>""")
            content = "\n".join(parts)
            page_html = _build_page_template(
                f"Experiment {exp_num}: {_exp_titles.get(exp_num, '')}",
                nav_active=nav_key,
                content_html=content,
            )
            (output_dir / f"experiment_{exp_num}.html").write_text(page_html, encoding="utf-8")
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
