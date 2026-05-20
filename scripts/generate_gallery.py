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

# Experiment descriptions for placeholder pages and the index cards.
# Kept in sync with the live metadata.json files in results/experiment_*/ —
# update both this dict and the corresponding "What This Experiment Tests"
# card when either matrix changes.
_EXPERIMENT_DESCRIPTIONS = {
    1: (
        "Strategy × Model Size — 5 RAG strategies (NaiveRAG, SelfRAG, "
        "MultiQueryRAG, CorrectiveRAG, AdaptiveRAG) × 6 models "
        "(Qwen 3.5: 0.8B / 2B / 4B / 9B; Gemma 4 e-tier: e2b / e4b) "
        "= 30 configurations. Held constant: Recursive chunker (500 / 100), "
        "embeddinggemma:300m embedder, hybrid retrieval (top-k 5), no reranker. "
        "Scored by a 2-judge panel: Claude Haiku 4.5 + GPT-5.4 mini."
    ),
    2: (
        "Chunking × Model Size — 4 chunking strategies "
        "(Fixed 512, Recursive 500 / 100, Sentence, Semantic) × 4 Qwen 3.5 models "
        "(0.8B / 2B / 4B / 9B) = 16 configurations. Held constant: NaiveRAG "
        "strategy, mxbai-embed-large embedder, hybrid retrieval (top-k 5). "
        "Scored by a 2-judge panel: Claude Haiku 4.5 + GPT-5.4 mini."
    ),
}


# ---------------------------------------------------------------------------
# Shared CSS theme
# ---------------------------------------------------------------------------

_GALLERY_CSS = """
/* RAGBench gallery — editorial/academic theme.
   Cream paper, serif display type, IBM-palette accents as data markers. */

:root {
    /* Paper + ink */
    --paper:       #f6f3ea;
    --paper-2:     #fdfbf4;
    --paper-edge:  #e6dfcc;
    --ink:         #18181b;
    --ink-2:       #3a3a3f;
    --ink-3:       #6a6a72;
    --ink-muted:   #9d9d9d;
    /* IBM colorblind-safe palette — chart-locked, do not change */
    --c-blue:      #648FFF;
    --c-purple:    #785EF0;
    --c-magenta:   #DC267F;
    --c-orange:    #FE6100;
    --c-gold:      #FFB000;
    --c-teal:      #22A884;
    /* Type stacks — system fonts only (no CDN) */
    --serif:  ui-serif, Charter, "Bitstream Charter", "Sitka Text", "Source Serif Pro", Cambria, Georgia, serif;
    --sans:   -apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", Roboto, system-ui, sans-serif;
    --mono:   ui-monospace, "SF Mono", "Cascadia Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

body {
    font-family: var(--sans);
    background: var(--paper);
    color: var(--ink);
    line-height: 1.65;
    font-size: 17px;
    /* subtle vignette at the top of the viewport — sets atmosphere without
       competing with the Plotly charts further down the page */
    background-image: radial-gradient(ellipse 70% 35% at 50% -8%,
                                       rgba(100,143,255,0.06),
                                       transparent 70%);
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* === Navigation === */
.nav {
    background: rgba(246, 243, 234, 0.92);
    backdrop-filter: saturate(140%) blur(8px);
    -webkit-backdrop-filter: saturate(140%) blur(8px);
    border-bottom: 1px solid var(--paper-edge);
    padding: 18px 32px;
    display: flex;
    gap: 4px;
    align-items: baseline;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav .brand {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.45em;
    color: var(--ink);
    letter-spacing: -0.015em;
    margin-right: auto;
    position: relative;
}
.nav .brand::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -3px;
    width: 26px;
    height: 2px;
    background: var(--c-blue);
}
.nav a {
    color: var(--ink-2);
    text-decoration: none;
    font-family: var(--sans);
    font-size: 0.74em;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 8px 12px;
    border-radius: 2px;
    transition: color 0.15s, background 0.15s;
    position: relative;
}
.nav a:hover { color: var(--ink); background: rgba(24,24,27,0.04); }
.nav a.active { color: var(--ink); background: transparent; }
.nav a.active::after {
    content: "";
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: 2px;
    height: 2px;
    background: var(--c-blue);
}
.nav__source {
    margin-left: 16px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--ink-2);
    border: 1px solid var(--paper-edge);
    background: var(--paper-2);
}
.nav__source:hover {
    color: var(--ink);
    background: var(--paper);
    border-color: var(--ink-muted);
}
.nav__source svg { display: inline-block; vertical-align: middle; }

/* === Version sub-nav === */
.version-nav {
    background: var(--paper-2);
    border-bottom: 1px solid var(--paper-edge);
    padding: 10px 32px;
    display: flex;
    gap: 6px;
    align-items: baseline;
    font-size: 0.82em;
}
.version-nav .version-label {
    color: var(--ink-muted);
    font-size: 0.85em;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-right: 10px;
}
.version-nav a {
    color: var(--ink-3);
    text-decoration: none;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.95em;
    transition: color 0.15s, background 0.15s;
}
.version-nav a:hover { color: var(--ink); background: rgba(24,24,27,0.04); }
.version-nav a.active {
    color: var(--ink);
    background: var(--paper);
    border: 1px solid var(--paper-edge);
    padding: 2px 9px;
}

/* === Main content === */
.content {
    max-width: 1180px;
    margin: 0 auto;
    padding: 56px 32px 80px;
}

/* === Typography === */
h1, h2, h3, h4 {
    font-family: var(--serif);
    color: var(--ink);
    font-weight: 600;
    letter-spacing: -0.012em;
    line-height: 1.22;
}
h1 {
    font-size: 2.4em;
    margin-bottom: 32px;
    padding-bottom: 16px;
    position: relative;
}
h1::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0;
    width: 56px;
    height: 3px;
    background: var(--c-blue);
}
h2 {
    font-size: 1.55em;
    margin-top: 52px;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--paper-edge);
}
h3 {
    font-size: 1.18em;
    margin-top: 28px;
    margin-bottom: 10px;
    color: var(--ink);
}
h4 {
    font-size: 1em;
    margin-bottom: 6px;
    color: var(--ink);
}
p {
    margin: 14px 0;
    color: var(--ink-2);
    max-width: 72ch;
}
strong { color: var(--ink); font-weight: 600; }
em { font-style: italic; color: var(--ink-2); }

a {
    color: var(--ink);
    text-decoration: underline;
    text-decoration-color: var(--c-blue);
    text-decoration-thickness: 1.5px;
    text-underline-offset: 3px;
    transition: text-decoration-color 0.15s;
}
a:hover { text-decoration-color: var(--c-magenta); }

/* === Cards === */
.card {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 34px 38px;
    margin: 28px 0;
    position: relative;
}
/* Editorial section-marker — short colored tick at the top of every card */
.card::before {
    content: "";
    position: absolute;
    top: -1px;
    left: 38px;
    width: 44px;
    height: 3px;
    background: var(--c-blue);
}
.card h2:first-child { margin-top: 0; }

/* === Experiment grid === */
.experiment-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 22px;
    margin: 28px 0;
}
.experiment-card {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 28px 30px;
    transition: transform 0.2s ease, border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.experiment-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--c-blue);
    transform: scaleX(0.18);
    transform-origin: left;
    transition: transform 0.35s ease;
}
.experiment-card:nth-child(2)::before { background: var(--c-purple); }
.experiment-card:nth-child(3)::before { background: var(--c-magenta); }
.experiment-card:nth-child(4)::before { background: var(--c-orange); }
.experiment-card:nth-child(5)::before { background: var(--c-teal); }
.experiment-card:hover {
    transform: translateY(-2px);
    border-color: var(--ink-muted);
    box-shadow: 0 10px 30px -16px rgba(24,24,27,0.22);
}
.experiment-card:hover::before { transform: scaleX(1); }
.experiment-card a {
    text-decoration: none;
    color: inherit;
    display: block;
}
.experiment-card h3 {
    color: var(--ink);
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 1.28em;
}
.experiment-card p {
    color: var(--ink-3);
    font-size: 0.94em;
    margin: 0;
    max-width: none;
}
.experiment-card .status {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 0.7em;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: var(--mono);
}
.status-ready {
    background: rgba(34,168,132,0.12);
    color: #0d5a3c;
    border: 1px solid rgba(34,168,132,0.32);
}
.status-placeholder {
    background: rgba(254,97,0,0.10);
    color: #82340a;
    border: 1px solid rgba(254,97,0,0.32);
}

/* === Chart container === */
.chart-container {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 22px;
    margin: 24px 0;
}

/* === Feature card (cards that want an extra-loud left accent) === */
.feature-card {
    border-left: 4px solid var(--c-blue);
    padding-left: 34px;
    margin-bottom: 32px;
}
.feature-card::before { display: none; }
.feature-card .cta-btn { margin-top: 16px; }
.feature-card--warn { border-left-color: var(--c-orange); }

/* Italic kicker line under a section title */
.kicker-italic {
    color: var(--ink-3);
    font-style: italic;
    margin-bottom: 14px;
    font-family: var(--serif);
}

/* Mono-caps label that sits above an h2 ("HEADLINE FINDINGS" etc.) */
.kicker-mono {
    font-family: var(--mono);
    font-size: 0.74em;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--c-magenta);
    margin-bottom: 0;
}

/* Opening paragraph on long-form pages — larger, serif, leading higher */
.methodology-content .lede,
.lede {
    font-family: var(--serif);
    font-size: 1.18em;
    line-height: 1.55;
    color: var(--ink);
    margin: 8px 0 28px;
    max-width: 72ch;
}
.lede em { font-style: italic; color: var(--ink); }

/* Inline "Further reading" / footnote-style paragraph */
.further-reading {
    font-size: 0.9em;
    color: var(--ink-3);
    margin-top: 18px;
    line-height: 1.55;
}
.further-reading strong { color: var(--ink-2); }

/* === RAG explainer figure (Wikipedia CC-BY-SA diagram on the index) === */
.rag-figure {
    margin: 22px auto 6px;
    max-width: 560px;
    text-align: center;
    background: var(--paper);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 22px 22px 16px;
}
.rag-figure__img {
    display: block;
    width: 100%;
    height: auto;
    max-width: 480px;
    margin: 0 auto;
}
.rag-figure__caption {
    font-family: var(--serif);
    font-size: 0.92em;
    color: var(--ink-2);
    line-height: 1.5;
    margin-top: 14px;
    padding: 0 6px;
    text-align: left;
    font-style: italic;
}
.rag-figure__caption strong { font-style: normal; color: var(--ink); }
.rag-figure__caption em { font-style: italic; color: var(--ink); }
.rag-figure__credit {
    display: block;
    margin-top: 8px;
    font-family: var(--mono);
    font-style: normal;
    font-size: 0.78em;
    letter-spacing: 0.04em;
    color: var(--ink-3);
}

/* === Callout box (high-emphasis acknowledgement / warning) === */
.callout {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-left: 4px solid var(--c-orange);
    border-radius: 3px;
    padding: 20px 26px 22px;
    margin: 26px 0;
}
.callout--critical { border-left-color: var(--c-magenta); }
.callout__label {
    font-family: var(--mono);
    font-size: 0.72em;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--c-magenta);
    margin-bottom: 10px;
}
.callout--critical .callout__label { color: var(--c-magenta); }
.callout__title {
    font-family: var(--serif);
    font-size: 1.2em;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 12px;
}
.callout p {
    margin: 8px 0;
    color: var(--ink-2);
    max-width: none;
}

/* === Audit / shortcomings table (severity pills + tight columns) === */
.audit-table { font-size: 0.9em; }
.audit-table td { vertical-align: top; line-height: 1.5; }
.audit-table td:first-child {
    font-family: var(--mono);
    color: var(--ink-3);
    width: 32px;
    text-align: right;
    padding-right: 8px;
}
.audit-table td:nth-child(4) { white-space: nowrap; }
.sev {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.72em;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.sev--critical {
    background: rgba(220,38,127,0.10);
    color: #8a0a4a;
    border-color: rgba(220,38,127,0.4);
}
.sev--high {
    background: rgba(254,97,0,0.10);
    color: #82340a;
    border-color: rgba(254,97,0,0.36);
}
.sev--medium {
    background: rgba(255,176,0,0.13);
    color: #6b4a00;
    border-color: rgba(255,176,0,0.5);
}
.sev--low {
    background: rgba(34,168,132,0.10);
    color: #0d5a3c;
    border-color: rgba(34,168,132,0.32);
}

/* === Catch-all muted note (footnotes, chart caveats, asides) === */
.muted-note,
p.muted-note,
li.muted-note {
    color: var(--ink-3);
    font-size: 0.88em;
    font-style: italic;
    line-height: 1.5;
}
.section-intro {
    color: var(--ink-2);
    margin-bottom: 16px;
}

/* === Chart explanation prose (sits between chart heading and chart) === */
.chart-explanation {
    color: var(--ink-2);
    font-size: 0.94em;
    line-height: 1.55;
    margin: 6px 0 18px;
    padding: 0 4px;
    max-width: 78ch;
}
.chart-explanation strong { color: var(--ink); }

/* === Experiment design callout (axes varied vs held constant) === */
.axis-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1.2fr;
    gap: 14px;
    margin: 22px 0 4px;
}
.axis-block {
    background: var(--paper);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 16px 18px;
    position: relative;
}
.axis-block::before {
    content: "";
    position: absolute;
    top: -1px;
    left: -1px;
    width: 28px;
    height: 2px;
}
.axis-varies::before { background: var(--c-magenta); }
.axis-held::before   { background: var(--ink-muted); }
.axis-label {
    font-family: var(--mono);
    font-size: 0.7em;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 6px;
}
.axis-varies .axis-label { color: var(--c-magenta); }
.axis-title {
    font-family: var(--serif);
    font-size: 1.05em;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 6px;
}
.axis-detail {
    font-size: 0.88em;
    color: var(--ink-2);
    line-height: 1.5;
}

/* === Strategy explainer table (index "RAG Strategies in this study") === */
.strategy-list {
    margin: 18px 0 4px;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.strategy-row {
    display: grid;
    grid-template-columns: 180px 1fr;
    gap: 24px;
    padding: 16px 0;
    border-top: 1px solid var(--paper-edge);
}
.strategy-row:last-child { border-bottom: 1px solid var(--paper-edge); }
.strategy-name {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.08em;
    color: var(--ink);
    position: relative;
    padding-left: 14px;
}
.strategy-name::before {
    content: "";
    position: absolute;
    left: 0; top: 0.45em;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--c-blue);
}
.strategy-row:nth-child(2) .strategy-name::before { background: var(--c-purple); }
.strategy-row:nth-child(3) .strategy-name::before { background: var(--c-magenta); }
.strategy-row:nth-child(4) .strategy-name::before { background: var(--c-orange); }
.strategy-row:nth-child(5) .strategy-name::before { background: var(--c-teal); }
.strategy-desc { color: var(--ink-2); font-size: 0.96em; line-height: 1.55; }
.strategy-desc strong { color: var(--ink); }
.strategy-meta {
    margin-top: 6px;
    font-family: var(--mono);
    font-size: 0.78em;
    color: var(--ink-3);
    letter-spacing: 0.04em;
}

/* === Worked example callout (one annotated row from the data) === */
.worked-example {
    background: var(--paper);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 22px 26px;
    margin-top: 26px;
    position: relative;
}
.worked-example::before {
    content: "EXAMPLE";
    position: absolute;
    top: -8px;
    left: 22px;
    background: var(--paper-2);
    padding: 0 8px;
    font-family: var(--mono);
    font-size: 0.66em;
    letter-spacing: 0.18em;
    color: var(--ink-muted);
}
.worked-example h3 {
    margin-top: 0;
    color: var(--ink);
    font-size: 1.18em;
}
.worked-example__lede {
    font-size: 0.92em;
    color: var(--ink-3);
    margin: 6px 0 18px;
}
.worked-example__rows {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 8px 18px;
    font-size: 0.95em;
    line-height: 1.6;
    margin: 0;
}
.worked-example__rows dt { margin: 0; }
.worked-example__rows dd { margin: 0; color: var(--ink-2); }
.worked-example__label {
    font-family: var(--serif);
    font-weight: 600;
    color: var(--ink);
}
.worked-example__label[data-accent="question"] { color: var(--c-purple); }
.worked-example__label[data-accent="gold"]     { color: var(--c-gold); }
.worked-example__label[data-accent="rag"]      { color: var(--c-orange); }
.worked-example__rag { font-style: italic; }
.worked-example code {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    padding: 1px 6px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.92em;
}

.worked-example__section {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px dashed var(--paper-edge);
}
.worked-example__section-label {
    font-family: var(--mono);
    font-size: 0.74em;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 12px;
}
.worked-example__section-label[data-accent="metrics"] { color: var(--c-teal); }
.worked-example__section-label[data-accent="judges"]  { color: var(--c-purple); }

.worked-example__metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.worked-example__metric {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 12px 14px;
}
.worked-example__metric-label {
    font-size: 0.78em;
    color: var(--ink-3);
    font-family: var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.worked-example__metric-value {
    font-family: var(--serif);
    font-size: 1.55em;
    font-weight: 600;
    color: var(--ink);
    margin: 4px 0;
    font-variant-numeric: tabular-nums;
}
.worked-example__metric-value--good { color: var(--c-teal); }
.worked-example__metric-value--warn { color: var(--c-orange); }
.worked-example__metric-note {
    font-size: 0.82em;
    color: var(--ink-3);
    line-height: 1.4;
}

.worked-example__judges {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px 14px;
    font-size: 0.91em;
    color: var(--ink-2);
}
.worked-example__judges strong { color: var(--ink); font-variant-numeric: tabular-nums; }
.worked-example__note {
    font-size: 0.88em;
    color: var(--ink-3);
    margin-top: 12px;
    line-height: 1.55;
}

/* === Captions === */
.caption {
    font-family: var(--serif);
    font-size: 0.94em;
    color: var(--ink-3);
    font-style: italic;
    max-width: 72ch;
    margin: 8px auto 30px;
    line-height: 1.5;
    padding: 0 12px;
    text-align: center;
}

/* === Placeholder === */
.placeholder { text-align: center; padding: 80px 24px; }
.placeholder h2 {
    border: none;
    color: var(--ink-3);
    font-style: italic;
    font-weight: 500;
}
.placeholder p {
    color: var(--ink-3);
    max-width: 600px;
    margin: 16px auto;
}

/* === Data tables — Tufte-style rules === */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92em;
    margin: 20px 0;
    font-variant-numeric: tabular-nums;
}
.data-table th, .data-table td {
    padding: 10px 14px;
    text-align: left;
    vertical-align: top;
    border: none;
    border-bottom: 1px solid var(--paper-edge);
}
.data-table thead th {
    background: transparent;
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    font-weight: 600;
    font-size: 0.76em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-2);
    font-family: var(--sans);
}
.data-table tbody tr:last-child td { border-bottom: 2px solid var(--ink); }
.data-table tbody tr:hover { background: rgba(100,143,255,0.05); }

/* === Footer === */
.footer {
    text-align: center;
    padding: 40px 24px 32px;
    color: var(--ink-3);
    font-size: 0.88em;
    border-top: 1px solid var(--paper-edge);
    margin-top: 60px;
    font-family: var(--serif);
    font-style: italic;
}
.footer a {
    color: var(--ink);
    text-decoration-color: var(--c-blue);
}
.footer a:hover { text-decoration-color: var(--c-magenta); }

/* === Hero (index landing) — oversized serif, editorial kicker === */
.hero {
    background: transparent;
    color: var(--ink);
    text-align: left;
    padding: 16px 0 56px;
    margin: -16px 0 48px;
    border-bottom: 1px solid var(--paper-edge);
    position: relative;
}
.hero::before {
    content: "Findings Gallery · 2026";
    display: block;
    font-family: var(--mono);
    font-size: 0.74em;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 22px;
}
.hero h1 {
    color: var(--ink);
    border-bottom: none;
    padding-bottom: 0;
    margin-bottom: 18px;
    font-size: 4.6em;
    font-weight: 600;
    letter-spacing: -0.038em;
    line-height: 0.96;
    font-family: var(--serif);
}
.hero h1::after { display: none; }
.hero .tagline {
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.4em;
    color: var(--ink-2);
    margin: 0 0 24px 0;
    max-width: 36ch;
    line-height: 1.32;
}
.hero .description {
    max-width: 64ch;
    margin: 0 0 30px 0;
    color: var(--ink-2);
    line-height: 1.7;
    font-size: 1.02em;
}
.hero .cta-btn {
    display: inline-block;
    background: var(--ink);
    color: var(--paper);
    padding: 12px 26px;
    border-radius: 2px;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.82em;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: var(--sans);
    transition: background 0.18s, transform 0.18s;
    border: 1px solid var(--ink);
}
.hero .cta-btn:hover {
    background: var(--c-blue);
    border-color: var(--c-blue);
    color: var(--paper-2);
    transform: translateY(-1px);
}

/* === Key findings cards === */
.findings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin: 28px 0;
}
.finding-card {
    background: var(--paper-2);
    border: 1px solid var(--paper-edge);
    border-radius: 3px;
    padding: 22px 26px;
    border-left: 3px solid var(--c-blue);
    transition: transform 0.18s, border-left-color 0.18s, box-shadow 0.18s;
}
.finding-card:nth-child(2) { border-left-color: var(--c-purple); }
.finding-card:nth-child(3) { border-left-color: var(--c-magenta); }
.finding-card:nth-child(4) { border-left-color: var(--c-orange); }
.finding-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px -14px rgba(24,24,27,0.18);
}
.finding-card h4 {
    color: var(--ink);
    margin-bottom: 8px;
    font-size: 1.02em;
    font-family: var(--serif);
    line-height: 1.3;
}
.finding-card p {
    color: var(--ink-3);
    font-size: 0.91em;
    margin: 0;
    max-width: none;
}

/* === Methodology page — readable prose column === */
.methodology-content {
    max-width: 72ch;
    margin: 0 auto;
}
.methodology-content h2 { margin-top: 56px; }
.methodology-content p {
    margin: 16px 0;
    color: var(--ink-2);
    max-width: none;
}
.methodology-content pre {
    background: var(--ink);
    color: #e8e4d6;
    padding: 22px 24px;
    border-radius: 3px;
    overflow-x: auto;
    font-size: 0.86em;
    line-height: 1.7;
    font-family: var(--mono);
    margin: 22px 0;
}
.methodology-content code {
    background: var(--paper-2);
    padding: 1px 6px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.9em;
    border: 1px solid var(--paper-edge);
    color: var(--ink);
}
.methodology-content pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: inherit;
}

/* === Responsive === */
@media (max-width: 768px) {
    body { font-size: 16px; }
    .content { padding: 36px 20px 56px; }
    .hero { padding: 12px 0 36px; margin: -12px 0 32px; }
    .hero h1 { font-size: 2.5em; }
    .hero .tagline { font-size: 1.12em; }
    .hero::before { font-size: 0.68em; margin-bottom: 16px; }
    .experiment-grid { grid-template-columns: 1fr; }
    .findings-grid { grid-template-columns: 1fr; }
    .nav { flex-wrap: wrap; gap: 4px; padding: 14px 18px; }
    .nav a { font-size: 0.7em; padding: 6px 8px; }
    .nav .brand { font-size: 1.25em; width: 100%; margin-right: 0; margin-bottom: 8px; }
    .version-nav { padding: 8px 18px; flex-wrap: wrap; }
    .card { padding: 24px 22px; }
    .card::before { left: 22px; }
    h1 { font-size: 1.9em; }
    h2 { font-size: 1.32em; }
    .axis-grid { grid-template-columns: 1fr; }
    .strategy-row { grid-template-columns: 1fr; gap: 6px; }
    .strategy-name { padding-left: 14px; }
    .worked-example__rows { grid-template-columns: 1fr; gap: 4px 0; }
    .worked-example__rows dd { margin-bottom: 8px; }
    .worked-example__metrics { grid-template-columns: 1fr; }
    .worked-example__judges  { grid-template-columns: repeat(2, 1fr); }
    .feature-card { padding-left: 22px; }
}
"""


# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------

_NAV_ITEMS = [
    ("home", "Home", "index.html"),
    ("exp0v3", "Exp 0v3: Scorer Validation", "experiment_0_v3.html"),
    ("exp1", "Exp 1: Strategy × Model", "experiment_1.html"),
    ("exp2", "Exp 2: Chunking × Model", "experiment_2.html"),
    ("methodology", "Methodology", "methodology.html"),
    ("writeup", "Writeup", "writeup.html"),
    ("data", "Data", "data.html"),
]


_EXP0_VERSIONS = [
    ("v1", "v1 (Initial)", "experiment_0.html"),
    ("v2", "v2 (Revised)", "experiment_0_v2.html"),
    ("v3", "v3 (Definitive)", "experiment_0_v3.html"),
]


def _build_page_template(
    title: str,
    nav_active: str,
    content_html: str,
    exp0_version: str | None = None,
) -> str:
    """Wrap content HTML in the shared page template with nav and CSS.

    Args:
        title: Page title for ``<title>`` and ``<h1>``.
        nav_active: Key of the active nav item (e.g. ``"home"``, ``"exp0"``).
        content_html: Inner HTML for the page body.
        exp0_version: If set (``"v1"``, ``"v2"``, ``"v3"``), renders the
            Experiment 0 version sub-navigation bar.

    Returns:
        Complete HTML page string.
    """
    nav_links = []
    for key, label, href in _NAV_ITEMS:
        cls = ' class="active"' if key == nav_active else ""
        nav_links.append(f'<a href="{href}"{cls}>{label}</a>')
    nav_html = "\n    ".join(nav_links)

    version_nav_html = ""
    if exp0_version:
        ver_links = []
        for key, label, href in _EXP0_VERSIONS:
            cls = ' class="active"' if key == exp0_version else ""
            ver_links.append(f'<a href="{href}"{cls}>{label}</a>')
        version_nav_html = (
            '\n    <div class="version-nav">'
            '<span class="version-label">Version:</span>'
            + "".join(ver_links)
            + "</div>"
        )

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
        <a class="nav__source" href="https://github.com/nono638/2026Project"
           target="_blank" rel="noopener" aria-label="RAGBench source code on GitHub">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
                <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            <span>Source</span>
        </a>
    </nav>{version_nav_html}
    <div class="content">
        <h1>{title}</h1>
        {content_html}
    </div>
    <div class="footer">
        Built by Noah &middot; CUNY SPS, Spring 2026 &middot;
        <a href="methodology.html">Methodology</a> &middot;
        <a href="https://github.com/nono638/2026Project" target="_blank" rel="noopener">Source on GitHub</a>
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
    # Experiment cards — exclude Experiment 0 entries (v1/v2/v3) since the
    # hero section handles Exp 0. Only show Exp 1, Exp 2, etc.
    cards = []
    for exp in experiments_info:
        # Skip all Experiment 0 variants — they're covered by the hero
        num_str = str(exp["num"])
        if num_str.startswith("0"):
            continue
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

    # Methodology link card alongside experiment cards
    methodology_card = """
        <div class="experiment-card">
            <a href="methodology.html">
                <span class="status status-ready">Reference</span>
                <h3>Methodology</h3>
                <p>How RAGBench works: pipeline overview, evaluation approach, and experiment design.</p>
            </a>
        </div>"""

    workflow_fig = _create_workflow_diagram()
    rag_pipeline_fig = _create_rag_pipeline_diagram()
    try:
        from scripts.generate_experiment0_dashboard import _fig_to_html
    except Exception:
        import plotly.io as pio
        def _fig_to_html(fig: Any) -> str:
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    workflow_html = _fig_to_html(workflow_fig)
    rag_pipeline_html = _fig_to_html(rag_pipeline_fig)
    worked_example_html = _generate_worked_example()

    return f"""
    <div class="hero">
        <h1>RAGBench</h1>
        <p class="tagline">A configurable evaluation pipeline for Retrieval-Augmented Generation</p>
        <p class="description">
            RAGBench runs the full cartesian product of RAG configurations
            (chunker × embedder × strategy × language model), scores the results
            with LLM judges, and identifies optimal configurations for different
            constraints.
        </p>
    </div>

    <div class="card">
        <h2>What is RAG?</h2>
        <p>
            <strong>Retrieval-Augmented Generation</strong> (RAG) is the dominant pattern
            for grounding language models in external knowledge. Instead of relying solely
            on what the model memorized during training, a RAG system <em>retrieves</em>
            relevant passages from a corpus &mdash; documentation, papers, internal wikis &mdash;
            and feeds them to the model alongside the question. The model then <em>generates</em>
            an answer grounded in the retrieved context.
        </p>
        <p>
            That means a RAG pipeline has many moving parts, each of which independently
            affects answer quality: how the corpus is chunked, how passages are retrieved,
            how retrieval results are reranked, how the prompt is constructed, and which
            language model produces the final answer. Small changes in any one component
            can swing accuracy by double-digit percentages.
        </p>
        <figure class="rag-figure">
            <img class="rag-figure__img"
                 src="https://upload.wikimedia.org/wikipedia/commons/1/14/RAG_diagram.svg"
                 alt="Block diagram of a Retrieval-Augmented Generation system: a user query and retrieved external documents are combined into a prompt and passed to a large language model, which produces a tailored response."
                 loading="lazy"
                 width="652" height="576">
            <figcaption class="rag-figure__caption">
                <strong>The general picture.</strong> A user&rsquo;s query is sent both
                to a language model <em>and</em> to a retriever that pulls relevant
                passages from an external corpus; the retrieved passages are merged
                with the query into an augmented prompt; the model answers using
                that grounded context.
                <span class="rag-figure__credit">
                    Diagram by <a href="https://commons.wikimedia.org/wiki/User:Turtlecrown" target="_blank" rel="noopener">Turtlecrown</a>,
                    Wikimedia Commons (<a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener">CC&nbsp;BY-SA&nbsp;4.0</a>).
                </span>
            </figcaption>
        </figure>
        <p class="further-reading">
            <strong>For a deeper look:</strong>
            <a href="https://developer.nvidia.com/blog/what-is-retrieval-augmented-generation/" target="_blank" rel="noopener">
                NVIDIA: What is RAG? (illustrated overview with multi-stage architecture diagram)
            </a> &middot;
            <a href="https://github.com/langchain-ai/rag-from-scratch" target="_blank" rel="noopener">
                LangChain &mdash; RAG from Scratch (in-depth notebooks &amp; diagrams covering query
                translation, routing, indexing, retrieval, and generation)
            </a>
        </p>
        <p class="further-reading">
            <strong>Foundational reading:</strong>
            <a href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener">
                Lewis et al. 2020 (the original RAG paper)
            </a> &middot;
            <a href="https://en.wikipedia.org/wiki/Retrieval-augmented_generation" target="_blank" rel="noopener">
                Wikipedia overview
            </a> &middot;
            <a href="https://huggingface.co/docs/transformers/model_doc/rag" target="_blank" rel="noopener">
                Hugging Face RAG docs
            </a>
        </p>
    </div>

    <div class="card">
        <h2>What is RAGBench?</h2>
        <p>
            RAGBench is a configurable harness for measuring how those moving parts
            affect quality. It runs the full cartesian product of RAG configurations
            (chunker &times; embedder &times; reranker &times; strategy &times; LLM) over a
            shared question set, scores every answer with both <strong>automated
            metrics</strong> (BERTScore, F1, exact-match against gold answers) and a
            panel of <strong>LLM judges</strong> rating faithfulness, relevance, and
            conciseness, then compares results across configurations to identify
            best-per-cost frontiers.
        </p>
        <p>
            The diagram below shows one row of the pipeline: a question and its source
            documents flow through the RAG configuration on top, the gold answer is held
            out as ground truth on the bottom, and both objective metrics and LLM judges
            score the result.
        </p>
        {workflow_html}
        <p>
            Inside that pink <strong>RAG Pipeline</strong> box are five components.
            Any of them can be a test axis depending on the experiment &mdash;
            Experiment&nbsp;1 varies the RAG strategy, Experiment&nbsp;2 varies the
            chunker, and so on &mdash; while the others are held constant within
            each experiment so the effect being measured is not confounded.
            Indexing happens once per corpus on top; every query traverses the
            bottom row.
        </p>
        {rag_pipeline_html}
        {worked_example_html}
    </div>

    <div class="card feature-card">
        <h2>RAG Strategies in This Study</h2>
        <p>
            Five strategies, each adding a different layer of intelligence between the
            user&rsquo;s question and the language model&rsquo;s answer. The first is the
            baseline; the rest each spend extra LLM calls to compensate for retrieval or
            reasoning gaps.
        </p>
        <div class="strategy-list">
            <div class="strategy-row">
                <div class="strategy-name">NaiveRAG</div>
                <div class="strategy-desc">
                    <strong>Retrieve, then answer.</strong> One round of hybrid retrieval;
                    the top-5 chunks are concatenated into the prompt; one LLM call produces
                    the answer. The control condition every other strategy is measured against.
                    <div class="strategy-meta">LLM calls per question: 1 &middot;
                    Tradeoff: cheap and predictable, no recovery from bad retrieval.</div>
                </div>
            </div>
            <div class="strategy-row">
                <div class="strategy-name">SelfRAG</div>
                <div class="strategy-desc">
                    <strong>Retrieve, critique, answer.</strong> After retrieval, the model
                    is prompted to judge whether the retrieved context is sufficient. If not,
                    it answers from parametric knowledge instead. Trades 1 extra LLM call for
                    awareness of its own retrieval failures.
                    <div class="strategy-meta">LLM calls per question: 2 &middot;
                    Tradeoff: handles retrieval misses, but adds latency on every query.</div>
                </div>
            </div>
            <div class="strategy-row">
                <div class="strategy-name">MultiQueryRAG</div>
                <div class="strategy-desc">
                    <strong>Reformulate, then retrieve N times.</strong> The original question
                    is rewritten into 3 paraphrases; each runs a retrieval pass; the union of
                    chunks is deduplicated and answered against. Helps when the question&rsquo;s
                    wording happens to mismatch the corpus vocabulary.
                    <div class="strategy-meta">LLM calls per question: ~5
                    (1 reformulation + ~3 retrievals + 1 answer) &middot;
                    Tradeoff: better recall, more chunks to sort through.</div>
                </div>
            </div>
            <div class="strategy-row">
                <div class="strategy-name">CorrectiveRAG</div>
                <div class="strategy-desc">
                    <strong>Retrieve, filter chunk-by-chunk, optionally retry.</strong>
                    Every retrieved chunk gets a per-chunk relevance rating from the LLM.
                    If fewer than 2 chunks survive the filter, the query is reformulated and
                    a second retrieval round runs. Based on
                    <a href="https://arxiv.org/abs/2401.15884" target="_blank" rel="noopener">Shi
                    et al. 2024</a>. Branchy &mdash; one of the strategies where logging
                    matters most.
                    <div class="strategy-meta">LLM calls per question: ~6&ndash;13
                    (1 per retrieved chunk + optional reformulation + final answer) &middot;
                    Tradeoff: precise context, high cost on long retrieval lists.</div>
                </div>
            </div>
            <div class="strategy-row">
                <div class="strategy-name">AdaptiveRAG</div>
                <div class="strategy-desc">
                    <strong>Classify the question, then pick a strategy.</strong> The model
                    first labels the question (factoid vs reasoning vs multi-hop), then
                    routes to a strategy matched to that type. Tries to give each question
                    the right amount of pipeline, instead of paying the heaviest cost on
                    every query.
                    <div class="strategy-meta">LLM calls per question: variable
                    (1 classifier + however many calls the chosen sub-strategy uses) &middot;
                    Tradeoff: best amortized cost; harder to reason about per-row behaviour.</div>
                </div>
            </div>
        </div>
        <p class="further-reading">
            The per-row substituted prompts as actually sent, the branch
            path taken, and the exact LLM-call count were <em>not</em>
            stamped on the rows that produced Experiments 1 and 2
            &mdash; an experimental-design oversight we&rsquo;re owning
            rather than hiding. The prompt <em>templates</em> themselves
            are version-controlled in
            <code>src/strategies/*.py</code>, so the prompt shape is
            recoverable; what isn&rsquo;t recoverable per row is the
            substituted content for any call whose input came from a
            prior LLM output. Historical CSVs have been backfilled
            where possible (NaiveRAG&rsquo;s substituted prompt is
            byte-deterministic and has been reconstructed from the
            recorded context + question; multi-step strategies are
            schema-tagged but not fabricated). New runs from now on
            capture the full trace per row. See
            <a href="methodology.html#logging-gaps">Methodology &raquo;
            Logging Gaps &amp; What We&rsquo;re Fixing</a> for the
            audit and the fix.
        </p>
    </div>

    <div class="card feature-card">
        <h2>Experiment 0: Which LLM Judge Tracks Truth?</h2>
        <p class="kicker-italic">Frontier-class judges converge &mdash; version matters more than provider.</p>
        <p>
            500 HotpotQA questions. 9 LLM judges. <strong>GPT-5.4</strong> leads at
            r&nbsp;=&nbsp;0.605, with <strong>Claude Sonnet 4.6</strong> (0.575) and
            <strong>GPT-5.4 Mini</strong> (0.553) close behind. Older Claude versions
            (Sonnet 4, Opus 4) score ~30% lower than current Sonnet 4.6 &mdash; model
            version drift outweighs provider differences.
        </p>
        <a href="experiment_0_v3.html" class="cta-btn">
            View Experiment 0 Results &rarr;
        </a>
    </div>

    <h2>Key Findings</h2>
    <p>From Experiment 0v3 — 500 HotpotQA questions scored by 9 LLM judges (3 Gemini + 4 Claude + 2 OpenAI).</p>
    <div class="findings-grid">
        <div class="finding-card">
            <h4>GPT-5.4 is the most accurate LLM judge</h4>
            <p>Highest correlation with gold F1 (r=0.605) at n=500. Sonnet 4.6 (0.575) and GPT-5.4 Mini (0.553) follow.</p>
        </div>
        <div class="finding-card">
            <h4>Model version dominates provider differences</h4>
            <p>Claude Sonnet jumped from r=0.397 (4) → 0.575 (4.6) — a ~45% relative gain. Same provider, different generation.</p>
        </div>
        <div class="finding-card">
            <h4>GPT-5.4 Mini is the best value</h4>
            <p>r=0.553 at $0.0015/call — most accurate among cheap judges, and cross-validates with Sonnet 4.6 (r=0.720).</p>
        </div>
        <div class="finding-card">
            <h4>76% exact match on HotpotQA — answer quality is solid</h4>
            <p>Retrieval failures at 14%, generation failures at 10%. Pipeline works well out of the box.</p>
        </div>
    </div>

    <h2 id="experiments">Experiments</h2>
    <div class="experiment-grid">
        {cards_html}
        {methodology_card}
    </div>
    """


def _generate_worked_example() -> str:
    """Build a styled callout walking one real Experiment 0 v3 row through the pipeline.

    The example is a verbose-but-correct answer pulled from
    ``results/experiment_0_v3/raw_scores.csv`` — chosen to show why multiple
    metrics are needed: word-overlap F1 looks bad, BERTScore looks moderate,
    exact-match passes, and the LLM judges all rate it as a strong answer.

    Returns:
        HTML string of the worked-example callout.
    """
    return """
    <div class="worked-example">
        <h3>Worked example</h3>
        <p class="worked-example__lede">
            One real row from Experiment 0 v3, tracing through the diagram above.
        </p>

        <dl class="worked-example__rows">
            <dt class="worked-example__label" data-accent="question">Question</dt>
            <dd>Who is the older mixed martial artist, Yushin Okami or Nate Marquardt?</dd>

            <dt class="worked-example__label" data-accent="gold">Gold answer</dt>
            <dd><code>Nate Marquardt</code></dd>

            <dt class="worked-example__label" data-accent="rag">RAG answer</dt>
            <dd class="worked-example__rag">
                &ldquo;Nate Marquardt is the older mixed martial artist. Yushin Okami was born on
                July 21, 1981, while Nate Marquardt was born on April 20, 1979. Marquardt is
                therefore older by two years and three months.&rdquo;
            </dd>
        </dl>

        <div class="worked-example__section">
            <div class="worked-example__section-label" data-accent="metrics">
                Bottom row &mdash; automated metrics (RAG vs gold)
            </div>
            <div class="worked-example__metrics">
                <div class="worked-example__metric">
                    <div class="worked-example__metric-label">BERTScore (semantic)</div>
                    <div class="worked-example__metric-value worked-example__metric-value--good">0.862</div>
                    <div class="worked-example__metric-note">strong semantic match</div>
                </div>
                <div class="worked-example__metric">
                    <div class="worked-example__metric-label">Word-overlap F1</div>
                    <div class="worked-example__metric-value worked-example__metric-value--warn">0.138</div>
                    <div class="worked-example__metric-note">low &mdash; verbose RAG answer adds many extra tokens</div>
                </div>
                <div class="worked-example__metric">
                    <div class="worked-example__metric-label">Exact match (gold &sub; answer)</div>
                    <div class="worked-example__metric-value worked-example__metric-value--good">true</div>
                    <div class="worked-example__metric-note">&ldquo;Nate Marquardt&rdquo; appears verbatim</div>
                </div>
            </div>
        </div>

        <div class="worked-example__section">
            <div class="worked-example__section-label" data-accent="judges">
                Top row &mdash; LLM judges (rate quality 1&ndash;5, never see the gold answer)
            </div>
            <div class="worked-example__judges">
                <div>Gemini 2.5 Pro: <strong>5.00</strong></div>
                <div>Claude Haiku 4.5: <strong>4.67</strong></div>
                <div>GPT-5.4 Mini: <strong>5.00</strong></div>
                <div>Claude Sonnet 4.6: <strong>4.67</strong></div>
                <div>GPT-5.4: <strong>4.67</strong></div>
                <div>Claude Opus 4: <strong>5.00</strong></div>
            </div>
            <div class="worked-example__note">
                All nine judges that scored this row rated it &ge;4.67 / 5 &mdash; they agree the
                answer is good, even though F1 alone would call it a poor match. This kind of
                disagreement between automated metrics is exactly why Experiment 0 exists:
                to find which judges&rsquo; ratings actually track the objective metrics.
            </div>
        </div>
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
        (1,  2, 2.4, 1.0, "Q&amp;A Dataset<br><span style='font-size:11px'>(HotpotQA)</span>", "#648FFF"),
        (5,  2, 2.8, 1.0, "Question +<br>Source Docs",      "#785EF0"),
        (9,  2, 3.2, 1.0, "RAG Pipeline<br><span style='font-size:11px'>(chunker × retriever × LLM)</span>", "#DC267F"),
        (13, 2, 2.4, 1.0, "RAG Answer",                     "#FE6100"),
        (17, 2, 2.4, 1.0, "LLM Judges",                     "#6B5B95"),
        (1,  0, 2.4, 1.0, "Gold Answer",                     "#FFB000"),
        (9,  0, 3.6, 1.0, "Automated Metrics<br>(BERTScore, F1)", "#22A884"),
        (17, 0, 3.0, 1.0, "Compare<br>judge vs gold",       "#648FFF"),
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


def _create_rag_pipeline_diagram() -> Any:
    """Plotly diagram showing the RAG pipeline internals (chunker, embedder,
    retriever, reranker, strategy, LLM) and which stages are RAGBench's test
    axes vs. held constant.

    Color coding:
    - Blue (#648FFF):    inputs (Documents, Question)
    - Magenta (#DC267F): test axes (Chunker, Reranker, RAG Strategy, LLM)
    - Slate (#6B7280):   passive infra / held constant (Embedder, Vector Index, Retriever)
    - Orange (#FE6100):  output (Answer)

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    AXIS = "#DC267F"
    INPUT = "#648FFF"
    INFRA = "#6B7280"
    OUTPUT = "#FE6100"

    # (x, y, w, h, label, color, sublabel)
    boxes = [
        # Top row (offline indexing): Documents -> Chunker -> Embedder -> Vector Index
        (2.5,  2, 2.6, 1.0, "Documents",     INPUT, "corpus"),
        (6.5,  2, 2.6, 1.0, "Chunker",       AXIS,  "axis · 4 strategies"),
        (10.5, 2, 2.8, 1.0, "Embedder",      INFRA, "held constant"),
        (14.5, 2, 2.8, 1.0, "Vector Index",  INFRA, "FAISS"),
        # Bottom row (online query): Question -> Retriever -> Reranker -> Strategy -> LLM -> Answer
        (10.5, 0, 2.6, 1.0, "Question",      INPUT, "user query"),
        (14.5, 0, 2.6, 1.0, "Retriever",     INFRA, "top-K hybrid"),
        (18.5, 0, 2.6, 1.0, "Reranker",      AXIS,  "axis · 3 options"),
        (22.5, 0, 2.8, 1.0, "RAG Strategy",  AXIS,  "axis · 5 strategies"),
        (26.0, 0, 2.0, 1.0, "LLM",           AXIS,  "axis · 6 models"),
        (29.0, 0, 2.4, 1.0, "Answer",        OUTPUT, "to scorer"),
    ]

    shapes = []
    annotations = []

    for x, y, w, h, label, color, sublabel in boxes:
        shapes.append(dict(
            type="rect",
            x0=x - w / 2, y0=y - h / 2,
            x1=x + w / 2, y1=y + h / 2,
            fillcolor=color, opacity=0.92,
            line=dict(color="white", width=2),
            layer="above",
        ))
        annotations.append(dict(
            x=x, y=y + 0.08,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color="white", size=12),
            align="center",
        ))
        # Sublabel inside the box, smaller
        annotations.append(dict(
            x=x, y=y - 0.22,
            text=f"<span style='font-size:10px;opacity:0.92'>{sublabel}</span>",
            showarrow=False,
            font=dict(color="white", size=10),
            align="center",
        ))

    # Arrows: (tail_x, tail_y, head_x, head_y)
    arrows = [
        # Top row offline indexing flow
        (3.8,  2,    5.2,  2),    # Documents → Chunker
        (7.8,  2,    9.1,  2),    # Chunker → Embedder
        (11.9, 2,    13.1, 2),    # Embedder → Vector Index
        # Vertical: Vector Index ↓ Retriever
        (14.5, 1.5,  14.5, 0.5),
        # Bottom row online query flow
        (11.8, 0,    13.2, 0),    # Question → Retriever
        (15.8, 0,    17.2, 0),    # Retriever → Reranker
        (19.8, 0,    21.1, 0),    # Reranker → RAG Strategy
        (23.9, 0,    25.0, 0),    # RAG Strategy → LLM
        (27.0, 0,    27.8, 0),    # LLM → Answer
    ]

    for ax, ay, x, y in arrows:
        annotations.append(dict(
            x=x, y=y, ax=ax, ay=ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#555", text="",
        ))

    # Section labels above each row
    annotations.append(dict(
        x=2.5, y=2.85,
        text="<span style='font-size:11px;color:#888'><b>OFFLINE INDEXING</b> (per corpus, once)</span>",
        showarrow=False, xanchor="left",
    ))
    annotations.append(dict(
        x=10.5, y=0.85,
        text="<span style='font-size:11px;color:#888'><b>ONLINE QUERY</b> (per question)</span>",
        showarrow=False, xanchor="left",
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 31], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1, 3.4], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        shapes=shapes, annotations=annotations,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
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
                <h4>Source Document <span class="muted-note" style="font-weight:normal;">
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
    from the v2 data.  Judges with < 50% non-null scores are flagged as
    "partial" and excluded from correlations/gold charts to avoid misleading
    statistics from too few data points.

    Args:
        csv_path: Path to ``results/experiment_0_v2/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    import plotly.graph_objects as go

    df = pd.read_csv(csv_path)
    total_rows = len(df)

    # --- Detect partial judges (< 50% non-null quality scores) ---
    # Correlations from very few data points are statistically meaningless,
    # so partial judges are excluded from correlation/gold charts and flagged
    # with a distinct color + asterisk in bar charts.
    quality_cols = [c for c in df.columns if c.endswith("_quality") and c != "answer_quality"]
    partial_judges: dict[str, int] = {}  # prefix -> valid count

    for col in quality_cols:
        prefix = col.replace("_quality", "")
        valid_count = int(df[col].notna().sum())
        if valid_count < total_rows * 0.5:
            partial_judges[prefix] = valid_count

    # Build a filtered DataFrame for correlation/gold charts —
    # drop all columns belonging to partial judges so
    # build_experiment0_figures() never sees them.
    df_filtered = df.copy()
    for prefix in partial_judges:
        cols_to_drop = [c for c in df_filtered.columns if c.startswith(prefix + "_")]
        df_filtered = df_filtered.drop(columns=cols_to_drop, errors="ignore")

    # Try to reuse the standard Exp 0 chart builder for scorer charts
    # (using filtered data that excludes partial judges from correlations).
    # Helpers imported up front so they remain available even if figure
    # construction itself raises (e.g. dtype mismatch in legacy CSVs).
    try:
        from scripts.generate_experiment0_dashboard import (
            _fig_to_html,
            _caption_html,
        )
    except Exception:
        def _fig_to_html(fig: Any) -> str:
            """Convert a Plotly figure to inline HTML."""
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        def _caption_html(title: str) -> str:
            """Fallback no-op caption helper if dashboard import fails."""
            return ""

    try:
        from scripts.generate_experiment0_dashboard import build_experiment0_figures
        figures = build_experiment0_figures(df_filtered)
    except Exception as exc:
        logger.warning("build_experiment0_figures failed for v2: %s", exc)
        figures = []

    # --- Partial judge exclusion note ---
    partial_notes: list[str] = []
    for prefix, count in partial_judges.items():
        name = prefix.replace("_", " ").replace("google ", "").replace("anthropic ", "")
        partial_notes.append(f"{name} excluded ({count}/{total_rows} scores due to API rate limit)")
    partial_note_html = ""
    if partial_notes:
        partial_note_html = (
            '<p class="muted-note">'
            "Note: " + "; ".join(partial_notes) + ".</p>"
        )

    # --- Build v2-specific charts ---
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
            {_fig_to_html(fig_aq)}
        </div>
        {_caption_html("Answer Quality Distribution")}""")

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
            {_fig_to_html(fig_fs)}
        </div>
        {_caption_html("Failure Stage Breakdown")}""")

    # Chart 3: Mean quality scores per judge — partial judges shown in gray
    # with asterisk, full judges in blue, so the reader can see all judges
    # but knows which ones have incomplete data.
    judge_means: list[tuple[str, float, bool]] = []
    for col in quality_cols:
        prefix = col.replace("_quality", "")
        mean_val = float(df[col].dropna().mean()) if df[col].notna().any() else 0.0
        is_partial = prefix in partial_judges
        name = prefix.replace("_", " ").replace("google ", "").replace("anthropic ", "")
        if is_partial:
            count = partial_judges[prefix]
            name += f" * ({count}/{total_rows})"
        judge_means.append((name, mean_val, is_partial))

    if judge_means:
        names = [j[0] for j in judge_means]
        means = [j[1] for j in judge_means]
        bar_colors = ["#aaa" if j[2] else "#648FFF" for j in judge_means]

        fig_means = go.Figure(data=[
            go.Bar(
                x=names, y=means,
                marker_color=bar_colors,
                text=[f"{m:.3f}" for m in means],
                textposition="auto",
            )
        ])
        fig_means.update_layout(
            title="Mean Quality Score per Judge",
            xaxis_title="Judge",
            yaxis_title="Mean Quality (0-1)",
            template="plotly_white",
            height=400,
        )
        footnote = ""
        if partial_judges:
            footnote = '<p class="muted-note">* partial data (gray bars) — excluded from correlation analysis</p>'
        v2_charts.append(f"""
        <div class="chart-container">
            <h3>Mean Quality Score per Judge</h3>
            {_fig_to_html(fig_means)}
            {footnote}
        </div>
        {_caption_html("Mean Quality Score per Judge")}""")

    # Standard scorer charts from build_experiment0_figures
    scorer_charts: list[str] = []
    for title, fig in figures:
        scorer_charts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>
            {_fig_to_html(fig)}
        </div>
        {_caption_html(title)}""")

    # --- Findings summary at top of page ---
    findings_html = """
    <div class="card feature-card">
        <h2>Key Findings</h2>
        <p class="section-intro">
            v2 — 150 medium+hard HotpotQA questions, 7 LLM judges
        </p>
        <ul style="list-style: none; padding: 0; margin: 0;">
            <li style="margin: 8px 0;"><strong>Best judge by BERTScore correlation:</strong> Claude Haiku (r=0.640)</li>
            <li style="margin: 8px 0;"><strong>Best free judge:</strong> Gemini 2.5 Pro (r=0.518)</li>
            <li style="margin: 8px 0;"><strong>Pipeline accuracy:</strong> 74% exact match, 0.917 mean BERTScore</li>
            <li style="margin: 8px 0;"><strong>Answer quality:</strong> 49% good, 47% poor, 5% questionable</li>
            <li style="margin: 8px 0;"><strong>Failure stages:</strong> 74% none, 13% retrieval, 13% generation</li>
            <li class="muted-note" style="margin: 8px 0;">Note: gemini-3.1-pro-preview scored only 11/150 (API rate limit) — excluded from correlations</li>
        </ul>
    </div>
    """

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
            <a href="raw_scores_v2.csv">Download the v2 raw data (CSV)</a>
        </p>
    </div>

    {findings_html}

    {"".join(v2_charts)}

    <div class="card">
        <h2>Scorer Comparison Charts</h2>
        <p>Same scorer analysis as v1, but on the v2 dataset (harder questions, better context).</p>
        {partial_note_html}
    </div>

    {"".join(scorer_charts)}
    """

    return _build_page_template(
        "Experiment 0 v2 — Scorer Validation (Revised)",
        nav_active="exp0v3",
        content_html=content,
        exp0_version="v2",
    )


def _generate_experiment_0_v3(csv_path: Path) -> str:
    """Build the Experiment 0 v3 dashboard page with narrative + charts.

    v3 is the definitive scorer validation run (n=500). This page adds a
    "Road to v3" narrative explaining the journey through v1 and v2, a
    key findings card with v3-specific stats, and reuses the chart-building
    logic from the v2 generator.

    Args:
        csv_path: Path to ``results/experiment_0_v3/raw_scores.csv``.

    Returns:
        Full HTML page string.
    """
    import plotly.graph_objects as go

    df = pd.read_csv(csv_path)
    total_rows = len(df)

    # --- Detect partial judges (< 50% non-null quality scores) ---
    # Same logic as v2: partial judges excluded from correlation charts
    quality_cols = [c for c in df.columns if c.endswith("_quality") and c != "answer_quality"]
    partial_judges: dict[str, int] = {}

    for col in quality_cols:
        prefix = col.replace("_quality", "")
        valid_count = int(df[col].notna().sum())
        if valid_count < total_rows * 0.5:
            partial_judges[prefix] = valid_count

    # Build filtered DataFrame for correlation charts (excludes partial judges)
    df_filtered = df.copy()
    for prefix in partial_judges:
        cols_to_drop = [c for c in df_filtered.columns if c.startswith(prefix + "_")]
        df_filtered = df_filtered.drop(columns=cols_to_drop, errors="ignore")

    # Reuse the standard Exp 0 chart builder for scorer charts.
    # Helpers imported up front so they remain available even if figure
    # construction itself raises (e.g. dtype mismatch in legacy CSVs).
    try:
        from scripts.generate_experiment0_dashboard import (
            _fig_to_html,
            _caption_html,
        )
    except Exception:
        def _fig_to_html(fig: Any) -> str:
            """Convert a Plotly figure to inline HTML."""
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        def _caption_html(title: str) -> str:
            """Fallback no-op caption helper if dashboard import fails."""
            return ""

    try:
        from scripts.generate_experiment0_dashboard import build_experiment0_figures
        figures = build_experiment0_figures(df_filtered)
    except Exception as exc:
        logger.warning("build_experiment0_figures failed for v3: %s", exc)
        figures = []

    # --- Partial judge exclusion note ---
    partial_notes: list[str] = []
    for prefix, count in partial_judges.items():
        name = prefix.replace("_", " ").replace("google ", "").replace("anthropic ", "")
        partial_notes.append(f"{name} excluded ({count}/{total_rows} scores due to API rate limit)")
    partial_note_html = ""
    if partial_notes:
        partial_note_html = (
            '<p class="muted-note">'
            "Note: " + "; ".join(partial_notes) + ".</p>"
        )

    # --- Build v3 charts (same as v2 chart logic) ---
    v3_charts: list[str] = []

    # Chart 1: answer_quality distribution
    if "answer_quality" in df.columns:
        counts = df["answer_quality"].value_counts()
        labels = ["good", "questionable", "poor"]
        values = [counts.get(l, 0) for l in labels]
        colors = ["#22A884", "#FFB000", "#DC267F"]

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
        v3_charts.append(f"""
        <div class="chart-container">
            <h3>Answer Quality Distribution</h3>
            {_fig_to_html(fig_aq)}
        </div>
        {_caption_html("Answer Quality Distribution")}""")

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
        v3_charts.append(f"""
        <div class="chart-container">
            <h3>Failure Stage Breakdown</h3>
            {_fig_to_html(fig_fs)}
        </div>
        {_caption_html("Failure Stage Breakdown")}""")

    # Chart 3: Mean quality scores per judge
    judge_means: list[tuple[str, float, bool]] = []
    for col in quality_cols:
        prefix = col.replace("_quality", "")
        mean_val = float(df[col].dropna().mean()) if df[col].notna().any() else 0.0
        is_partial = prefix in partial_judges
        name = prefix.replace("_", " ").replace("google ", "").replace("anthropic ", "")
        if is_partial:
            count = partial_judges[prefix]
            name += f" * ({count}/{total_rows})"
        judge_means.append((name, mean_val, is_partial))

    if judge_means:
        names = [j[0] for j in judge_means]
        means = [j[1] for j in judge_means]
        bar_colors = ["#aaa" if j[2] else "#648FFF" for j in judge_means]

        fig_means = go.Figure(data=[
            go.Bar(
                x=names, y=means,
                marker_color=bar_colors,
                text=[f"{m:.3f}" for m in means],
                textposition="auto",
            )
        ])
        fig_means.update_layout(
            title="Mean Quality Score per Judge",
            xaxis_title="Judge",
            yaxis_title="Mean Quality (0-1)",
            template="plotly_white",
            height=400,
        )
        footnote = ""
        if partial_judges:
            footnote = '<p class="muted-note">* partial data (gray bars) — excluded from correlation analysis</p>'
        v3_charts.append(f"""
        <div class="chart-container">
            <h3>Mean Quality Score per Judge</h3>
            {_fig_to_html(fig_means)}
            {footnote}
        </div>
        {_caption_html("Mean Quality Score per Judge")}""")

    # Standard scorer charts from build_experiment0_figures
    scorer_charts: list[str] = []
    for title, fig in figures:
        scorer_charts.append(f"""
        <div class="chart-container">
            <h3>{title}</h3>
            {_fig_to_html(fig)}
        </div>
        {_caption_html(title)}""")

    # --- v3 key findings card ---
    v3_findings_html = """
    <div class="card feature-card">
        <h2>Key Findings</h2>
        <p class="section-intro">
            v3 — 500 medium+hard HotpotQA questions, 8 LLM judges (3 Gemini + 3 Claude + 2 OpenAI)
        </p>
        <ul style="list-style: none; padding: 0; margin: 0;">
            <li style="margin: 8px 0;"><strong>Best judge:</strong> GPT-5.4 (r=0.605 gold F1), Claude Sonnet 4.6 (0.575) close behind</li>
            <li style="margin: 8px 0;"><strong>Best value:</strong> GPT-5.4 Mini at $0.0015/call (r=0.553) — cheapest paid judge with high correlation</li>
            <li style="margin: 8px 0;"><strong>Version drift dominates:</strong> Sonnet 4.6 (0.575) vs Sonnet 4 (0.397) — same provider, ~45% better</li>
            <li style="margin: 8px 0;"><strong>Best free judge:</strong> Gemini 2.5 Pro (r=0.348)</li>
            <li style="margin: 8px 0;"><strong>Pipeline accuracy:</strong> 76.2% exact match, mean F1 0.546</li>
            <li style="margin: 8px 0;"><strong>Failure stages:</strong> 76% none, 14% retrieval, 10% generation</li>
        </ul>
    </div>
    """

    # --- Assemble page ---
    content = f"""
    <div class="card">
        <h2>The Road to v3</h2>
        <p>
            Experiment 0 asks a simple question: <em>which LLM judge most reliably tracks
            whether a RAG answer is actually correct?</em> It took three iterations to get
            a confident answer.
        </p>
        <p>
            <strong><a href="experiment_0.html">v1</a></strong> (n=50) was our first attempt.
            It had two critical flaws: judges scored against the full source document instead
            of the retrieved chunks the LLM actually saw, and there was no reranker in the
            pipeline. Sonnet came out on top — but the methodology was unsound.
        </p>
        <p>
            <strong><a href="experiment_0_v2.html">v2</a></strong> (n=150) fixed the tracking,
            added a BGE reranker (retrieve 10, keep 3), and filtered to medium+hard questions
            to avoid ceiling effects. Haiku beat Sonnet — directly contradicting v1. But with
            only 150 questions, the margin left room for doubt.
        </p>
        <p>
            <strong>v3</strong> (n=500) was meant to be the tiebreaker at scale. Among the
            original 6 judges (3 Gemini 2.5, 3 Claude 4-family), Haiku 4.5 led (r=0.450 with
            gold F1), followed by Sonnet 4 (0.397) and Opus 4 (0.382). Among the free Gemini
            judges, Flash and Pro are nearly interchangeable (r=0.892).
        </p>
        <p>
            Then we added <strong>GPT-5.4 and GPT-5.4 Mini</strong> retroactively (n=500, same
            questions and answers) — and they outperformed every Claude and Gemini judge by a
            wide margin. GPT-5.4 leads at r=0.605, with GPT-5.4 Mini close behind at r=0.553.
        </p>
        <p>
            That raised an obvious question: were Sonnet 4 and Opus 4 actually that bad, or just
            old? The May 2025 versions were configured back when they were the latest Claude
            models, but Anthropic shipped Sonnet 4.5, 4.6, Opus 4.5, 4.6, and 4.7 in the months
            between then and now. We re-ran <strong>Claude Sonnet 4.6</strong> on the same v3
            answers — and it jumped to <strong>r=0.575</strong>, a ~45% relative improvement
            over Sonnet 4 (0.397). That puts Sonnet 4.6 in <strong>second place overall</strong>,
            ahead of GPT-5.4 Mini and behind only GPT-5.4. Model version drift turns out to be a
            larger effect than provider differences.
        </p>
        <p class="muted-note" style="margin-top: 16px;">
            <strong>Note on versions:</strong> these are model snapshots in time. Haiku 4.5
            (Oct 2025) is the current latest in the Haiku line. Opus 4 (May 2025) is two
            generations old. We did spot-check <strong>Opus 4.7</strong> (Jan 2026) on a
            partial sample (N=66, $5 budget cap — Opus 4.7 is 15&times; the per-call cost of
            Sonnet 4.6); on apples-to-apples 66-row comparison Opus 4.7 jumped to r=0.698
            (vs Opus 4's 0.452 on the same rows), confirming the version-drift pattern.
            Sonnet 4.6 still led that subset at r=0.733. We didn't extend Opus 4.7 to the
            full N=500 (would have cost ~$32 more) because the same dollars are better
            spent on Experiments 1 and 2. GPT-5.5 was available but GPT-5.4 was chosen for cost.
        </p>
    </div>

    {v3_findings_html}

    <div class="card">
        <p style="font-size: 0.9em;">
            <a href="raw_scores_v3.csv">Download the v3 raw data (CSV)</a>
        </p>
    </div>

    <div class="card">
        <h2>Summary</h2>
        <p>
            Across 500 medium+hard HotpotQA questions, <strong>GPT-5.4</strong> tracked the
            gold-standard word-overlap F1 most reliably (Pearson r=0.605), with <strong>Claude
            Sonnet 4.6</strong> (0.575) and <strong>GPT-5.4 Mini</strong> (0.553) close behind.
            All three substantially outperformed Haiku 4.5 (0.450), Sonnet 4 (0.397), Opus 4
            (0.382), Gemini 2.5 Pro (0.348), Gemini 2.5 Flash (0.301), and Gemini 2.5 Flash-Lite
            (0.139). Inter-judge correlation within the Gemini 2.5 family is high (Flash and Pro
            at r=0.892), within OpenAI as well (5.4 and 5.4 Mini at r=0.784), and Sonnet 4.6
            agrees strongly with both OpenAI judges (r=0.796 with GPT-5.4, r=0.720 with GPT-5.4
            Mini) — pointing toward "frontier-class" judges converging on the same evaluative
            signal regardless of provider. The figures below visualize each step of that analysis.
        </p>
    </div>

    {"".join(v3_charts)}

    <div class="card">
        <h2>Scorer Comparison Charts</h2>
        <p>Same scorer analysis as v1/v2, but on the v3 dataset (500 questions, 6 judges).</p>
        {partial_note_html}
    </div>

    {"".join(scorer_charts)}
    """

    return _build_page_template(
        "Experiment 0 v3 — Scorer Validation (Definitive)",
        nav_active="exp0v3",
        content_html=content,
        exp0_version="v3",
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
        _caption_html,
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
        """Render a chart container with optional explanation prose, plus a caption below."""
        chart_html = figures_by_title.get(title)
        if chart_html is None:
            return ""
        expl_html = ""
        if explanation:
            expl_html = f'<p class="chart-explanation">{explanation}</p>'
        return f"""
        <div class="chart-container">
            <h3>{title}</h3>{expl_html}
            {chart_html}
        </div>
        {_caption_html(title)}"""

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
           >HotpotQA</a></strong>, a dataset of real questions where the
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
            <a href="raw_scores.csv">Download the raw data (CSV)</a>
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
            <strong>Decision (v1):</strong> Experiments 1 and 2 will use Claude
            Sonnet as the primary scorer for maximum accuracy.
        </p>
        <p class="muted-note">
            Superseded in v3 &mdash; after re-running on a larger, harder
            150-question sample with current-generation judges, the panel was
            changed to <strong>Claude Haiku 4.5 + GPT-5.4 mini</strong> (cheaper,
            cross-provider, both current-generation). See the
            <a href="experiment_0_v3.html">Experiment 0 v3 page</a> and the
            Methodology &rsquo;Scorer Selection&rsquo; section for the revised
            rationale.
        </p>
        <p style="margin-top: 16px; font-size: 0.9em;">
            <a href="raw_scores.csv">Download the raw data (CSV)</a>
            &mdash; all 50 questions, 6 judges, and gold metrics in one file.
        </p>
    </div>
    """)

    # ------------------------------------------------------------------
    # Lessons Learned
    # ------------------------------------------------------------------
    parts.append("""
    <h2>Lessons Learned</h2>
    <div class="card feature-card feature-card--warn">
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
    <div class="card feature-card" style="margin-top: 16px;">
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
        nav_active="exp0v3",
        content_html=content,
        exp0_version="v1",
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
    <div class="card feature-card">
        <div class="kicker-mono">Headline findings</div>
        <h2 style="margin-top:6px;">Smart strategies don&rsquo;t beat plain RAG here &mdash; and 4B is the sweet spot</h2>
        <div class="findings-grid">
            <div class="finding-card">
                <h4>Strategy effects are small at best, hurtful at worst</h4>
                <p>Naive (4.34), Multi-Query (4.33), and Corrective (4.27)
                cluster within ~0.07 quality points &mdash; statistically and
                practically indistinguishable on this benchmark.
                Adaptive (3.59) and Self-RAG (2.99) <em>hurt</em> every
                model they touched, dropping quality by 0.7&ndash;1.4
                points. The cost of strategy complexity didn&rsquo;t pay off.</p>
            </div>
            <div class="finding-card">
                <h4>Model size helps, but not as much as you&rsquo;d expect</h4>
                <p>Across 11&times; parameter count
                (qwen3.5:0.8B &rarr; 9B) mean quality moves only 3.27 &rarr; 4.03
                &mdash; about 0.76 points. The smallest model is meaningfully
                weaker; everything 2B and up is in the same band, and
                <strong>qwen3.5:4B (4.05) slightly outscores qwen3.5:9B
                (4.03)</strong>, suggesting a sweet spot rather than a
                monotonic &ldquo;bigger is better.&rdquo;</p>
            </div>
            <div class="finding-card">
                <h4>The winner is the simplest combination</h4>
                <p>Best config: <strong>NaiveRAG &times; qwen3.5:4B</strong>
                at 4.59 quality &middot; 59% exact match &middot;
                0.21 mean gold F1 &middot; 0.86 BERTScore F1. No clever
                strategy on a smaller model beats it by more than
                ~0.10 points; see &ldquo;Strategy Beats Size&rdquo;
                below for the upset cases that do exist.</p>
            </div>
            <div class="finding-card">
                <h4>Caveat: row-level results aren&rsquo;t reproducible</h4>
                <p>Generation ran at Ollama&rsquo;s default
                <code>temperature&nbsp;&asymp;&nbsp;0.8</code> with no
                seed, so a given row re-run today would produce a
                different answer. Aggregate rankings are stable in
                expectation but the small inter-strategy gaps above
                are within the noise that a temperature-0 + fixed-seed
                regime would eliminate. See
                <a href="writeup.html#reproducibility">Writeup
                &raquo; Stochastic generation</a>.</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>What This Experiment Tests</h2>
        <p>
            <strong>Does a smarter RAG strategy compensate for a smaller language model?</strong>
            A two-axis factorial design: every one of 5 RAG strategies is run against
            every one of 6 models, giving 30 configurations &times; 200 HotpotQA
            questions = 6,000 scored answers. With both axes varied we can measure
            main effects (does Corrective beat Naive on average? does 9B beat 4B?)
            <em>and</em> their interaction (does Corrective help small models
            more than large ones?).
        </p>
        <p>
            The key question: when does investing in a complex strategy pay off vs.
            just using a bigger model with simple NaiveRAG?
        </p>
        <div class="axis-grid">
            <div class="axis-block axis-varies">
                <div class="axis-label">Axis 1 &middot; varied</div>
                <div class="axis-title">RAG strategy (5 levels)</div>
                <div class="axis-detail">NaiveRAG, SelfRAG, MultiQueryRAG,
                CorrectiveRAG, AdaptiveRAG</div>
            </div>
            <div class="axis-block axis-varies">
                <div class="axis-label">Axis 2 &middot; varied</div>
                <div class="axis-title">Model size (6 levels)</div>
                <div class="axis-detail">Qwen 3.5 &mdash; 0.8B, 2B, 4B, 9B<br>
                Gemma 4 e-tier &mdash; e2b, e4b</div>
            </div>
            <div class="axis-block axis-held">
                <div class="axis-label">Held constant</div>
                <div class="axis-title">Pipeline scaffolding</div>
                <div class="axis-detail">RecursiveChunker (500 / 100) &middot;
                embeddinggemma:300m &middot; hybrid retrieval, top-k 5 &middot;
                no reranker &middot; 200 HotpotQA questions (seed 42) &middot;
                judges: Claude Haiku 4.5 + GPT-5.4 mini</div>
            </div>
        </div>
    </div>
    """)

    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        explanation = _EXP1_EXPLANATIONS.get(title, "")
        explanation_html = ""
        if explanation:
            explanation_html = f"""
            <p class="chart-explanation">{explanation}</p>"""
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
    <div class="card feature-card">
        <div class="kicker-mono">Headline findings</div>
        <h2 style="margin-top:6px;">Chunker choice barely moves the needle &mdash; model size is the dominant lever</h2>
        <div class="findings-grid">
            <div class="finding-card">
                <h4>Chunker effects are small</h4>
                <p>Mean quality by chunker spans only 0.20 points:
                fixed (4.32), semantic (4.27), recursive (4.23),
                sentence (4.11). Within any single model the chunker
                spread is 0.16&ndash;0.27 quality points &mdash; about
                the same order as the noise from stochastic generation.
                This is a meaningful <em>negative</em> result: spending
                effort on chunker tuning is unlikely to pay back vs
                spending it on model selection.</p>
            </div>
            <div class="finding-card">
                <h4>Model size matters about 3&times; more than chunker</h4>
                <p>Mean quality by model spans 0.79 points
                (qwen3.5:0.8B = 3.79 &rarr; qwen3.5:4B = 4.58) &mdash;
                roughly four times the largest chunker effect. As in
                Experiment 1, <strong>qwen3.5:4B (4.58) edges out
                qwen3.5:9B (4.45)</strong>: the 4B mark is the
                quality sweet spot for this benchmark.</p>
            </div>
            <div class="finding-card">
                <h4>The winner: fixed-size chunking on 4B</h4>
                <p>Best config: <strong>fixed-size chunker &times;
                qwen3.5:4B</strong> at 4.65 quality &middot; 74% exact
                match &middot; 0.27 mean gold F1. Plain fixed-size
                chunking outperformed the more elaborate sentence- and
                semantic-aware approaches at every model size we tested.</p>
            </div>
            <div class="finding-card">
                <h4>Same reproducibility caveat as Experiment 1</h4>
                <p>This run also used Ollama&rsquo;s default
                <code>temperature&nbsp;&asymp;&nbsp;0.8</code> with no
                seed. The chunker rankings are stable in expectation
                but the 0.16&ndash;0.27-point within-model spreads are
                comparable to the stochastic-generation noise band.
                See <a href="methodology.html#retrospective">Methodology
                &raquo; Retrospective</a>.</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>What This Experiment Tests</h2>
        <p>
            <strong>Does how you split documents into chunks affect answer quality,
            and does it interact with model size?</strong> A two-axis factorial
            design: 4 chunking strategies &times; 4 Qwen 3.5 models = 16
            configurations &times; 200 HotpotQA questions = 3,200 scored answers.
            RAG strategy is held constant at NaiveRAG to isolate the chunking
            variable from the strategy effects measured in Experiment 1.
        </p>
        <p>
            This is an understudied question &mdash; most RAG research treats
            chunking as a fixed preprocessing step. Here we test whether it deserves
            the same attention as strategy and model selection.
        </p>
        <div class="axis-grid">
            <div class="axis-block axis-varies">
                <div class="axis-label">Axis 1 &middot; varied</div>
                <div class="axis-title">Chunking strategy (4 levels)</div>
                <div class="axis-detail">Fixed 512 &middot; Recursive 500 / 100 &middot;
                Sentence &middot; Semantic</div>
            </div>
            <div class="axis-block axis-varies">
                <div class="axis-label">Axis 2 &middot; varied</div>
                <div class="axis-title">Model size (4 levels)</div>
                <div class="axis-detail">Qwen 3.5 &mdash; 0.8B, 2B, 4B, 9B</div>
            </div>
            <div class="axis-block axis-held">
                <div class="axis-label">Held constant</div>
                <div class="axis-title">Pipeline scaffolding</div>
                <div class="axis-detail">NaiveRAG strategy &middot;
                mxbai-embed-large embedder &middot; hybrid retrieval, top-k 5 &middot;
                no reranker &middot; 200 HotpotQA questions (seed 42) &middot;
                judges: Claude Haiku 4.5 + GPT-5.4 mini</div>
            </div>
        </div>
    </div>
    """)

    for title, fig in figures:
        chart_html = _fig_to_html(fig)
        explanation = _EXP2_EXPLANATIONS.get(title, "")
        explanation_html = ""
        if explanation:
            explanation_html = f"""
            <p class="chart-explanation">{explanation}</p>"""
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

def _generate_methodology() -> str:
    """Build the methodology page explaining the RAGBench pipeline and experiments.

    Returns:
        Full HTML page string for methodology.html.
    """
    rag_pipeline_fig = _create_rag_pipeline_diagram()
    try:
        from scripts.generate_experiment0_dashboard import _fig_to_html
    except Exception:
        import plotly.io as pio
        def _fig_to_html(fig: Any) -> str:
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    rag_pipeline_html = _fig_to_html(rag_pipeline_fig)

    content = f"""
    <div class="methodology-content">

        <h2>Pipeline Overview</h2>
        <p>
            RAGBench evaluates RAG configurations by running each query through every
            combination of five independent axes, scoring the outputs, and comparing
            them against gold-standard answers when available. The diagram below shows
            the RAG pipeline itself: indexing happens once per corpus on top, every
            query traverses the bottom row. Magenta boxes are the test axes that vary
            across experiments; gray boxes are passive infrastructure or held constant.
        </p>
        {rag_pipeline_html}

        <h2>The Five Axes</h2>
        <p>Each experiment varies one or two axes while holding the others constant,
           isolating the effect of each design decision.</p>

        <h3>Chunker</h3>
        <p>How documents are split into passages for retrieval. Options: fixed-size (512 tokens),
           recursive (500/100 overlap), sentence-boundary, and semantic (embedding-aware splits).</p>

        <h3>Embedder</h3>
        <p>How text chunks become vector representations for similarity search.
           Embedder choice is <em>held constant within each experiment</em> but
           changed once across the project timeline, so different experiments use
           different embedders.</p>
        <ul>
          <li><strong>Experiment 2</strong> (run before the 2026-05-13 switch):
              <code>mxbai-embed-large</code> via Ollama — 512-token context.</li>
          <li><strong>Experiment 1 &amp; Experiment 0 v3</strong> (post-switch):
              <code>embeddinggemma:300m</code> via Ollama — 2048-token context,
              768-dim output, Apache 2.0, ~300 MB on disk.</li>
        </ul>
        <p>The switch was a methodological response to a chunking-vs-embedder
           conflict surfaced during the first Experiment 2 run: the
           <code>FixedSizeChunker(500&nbsp;words)</code> and
           <code>SemanticChunker</code> configurations triggered hundreds of
           HTTP&nbsp;400s because their chunks (~700 tokens and up) overflowed
           mxbai&rsquo;s 512-token window. Setting Ollama&rsquo;s server-side
           <code>truncate=true</code> didn&rsquo;t help on the current Ollama
           version, leaving client-side truncation as the only fix. Truncating
           chunks to fit a small embedder isn&rsquo;t a fair test of chunkers
           &mdash; it silently discards the tail of every oversized chunk before
           similarity is computed, penalising any strategy that prefers larger
           units. The replacement is <strong>EmbeddingGemma 300M</strong>: a
           text-only encoder distilled from Gemma 3 with a 2K-token context,
           released under Apache 2.0 by Google, well-rated on MTEB at the
           sub-1B size class, and small enough that it co-resides comfortably
           with the 9B generation models on a 24&nbsp;GB GPU.</p>
        <p>The earlier <code>mxbai-embed-large</code> data is preserved in
           Experiment 2&rsquo;s archive but no longer drives new generation runs.
           Comparisons that span both experiments should treat the embedder as
           a known difference, not a held-constant variable.</p>

        <h3>Reranker</h3>
        <p>An optional second-pass ranker that re-scores retrieved chunks before they reach
           the LLM, improving precision at the cost of latency. Options: BGE reranker
           (cross-encoder), MiniLM (lightweight), or none. Experiment 0 uses BGE;
           Experiments 1 and 2 run without a reranker to isolate other variables.</p>

        <h3>Strategy</h3>
        <p>How retrieved context is used to generate answers. Five strategies tested:
           NaiveRAG (single retrieval), SelfRAG (self-critique loop), CorrectiveRAG
           (retrieval validation), AdaptiveRAG (complexity routing), and MultiQueryRAG
           (query expansion).</p>

        <h3>Model</h3>
        <p>Which language model generates the final answer. Tested in Experiments
           1 and 2: <strong>Qwen 3.5</strong> small tier (0.8B, 2B, 4B, 9B) plus
           <strong>Gemma 4</strong> e-tier (e2b &asymp; 2.3B effective, e4b
           &asymp; 4.5B effective) for the cross-family comparison in
           Experiment 1. All run locally via Ollama, GGUF
           Q4_K_M / Q8_0 quantisations as Ollama&rsquo;s default tags ship them.</p>

        <h2>Evaluation Approach</h2>
        <p>RAGBench uses dual evaluation to assess answer quality from two perspectives:</p>
        <p><strong>Intrinsic evaluation</strong> (always available): An LLM judge scores each
           answer on faithfulness, relevance, and conciseness against the retrieved context.
           No external reference data needed.</p>
        <p><strong>Extrinsic evaluation</strong> (when gold data exists): Answers are compared
           against known-correct reference answers using BERTScore and token-level F1.
           This enables scorer validation — Experiment 0 exists specifically to validate
           the LLM-as-judge approach against gold-standard metrics.</p>

        <h2>Scorer Selection</h2>
        <p>
            Experiment 0 ran 500 HotpotQA questions through ten LLM judges (3 Gemini, 5
            Claude, 2 OpenAI) and correlated each judge's quality scores against the gold
            metrics (BERTScore F1, token-level F1). The headline result: <strong>GPT-5.4</strong>
            led at r=0.605, with <strong>Claude Sonnet 4.6</strong> (0.575) and
            <strong>GPT-5.4 Mini</strong> (0.553) close behind.
        </p>
        <p>
            The unexpected finding was <strong>model version drift</strong>. Claude
            Sonnet 4 (May 2025) hit r=0.397; Claude Sonnet 4.6 (the same provider, eight
            months later) hit r=0.575 — a ~45% relative improvement on identical answers.
            A spot-check of Claude Opus 4.7 on a 66-row subset showed the same effect:
            Opus 4 scored r=0.452 on those rows, Opus 4.7 scored r=0.698. Within-provider
            version differences turned out to exceed cross-provider differences.
        </p>
        <p>
            <strong>Final scorer panel for Experiments 1 and 2:</strong>
            <strong>Claude Haiku 4.5</strong> (r=0.450, $0.002/call) and
            <strong>GPT-5.4 Mini</strong> (r=0.553, $0.0015/call). Both are the latest
            generation of their respective family lines (no version-drift penalty), both
            sit in the cheap tier, and they sit in different capability tiers across
            different providers — yielding a more genuinely independent cross-validation
            than a top-tier same-tier pair would. Total estimated cost across both full
            experiments (Exp 1 = 5 strategies × 6 models × 200 questions, Exp 2 = 4
            chunkers × 4 models × 200 questions, ~9,200 rows per judge): <strong>~$32</strong>.
            Sonnet 4.6 was considered as the primary (r=0.575) but at 5&times; the per-call
            cost it would have raised the panel total to ~$106 for ~22% better correlation —
            not worth it for relative-ranking research questions where judge consistency
            matters more than absolute gold-correlation.
        </p>

        <h2>Experiment Design</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Research Question</th>
                    <th>Matrix</th>
                    <th>Held Constant</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>0 (Scorer Validation)</td>
                    <td>Which LLM judge is most accurate?</td>
                    <td>500 medium+hard HotpotQA &times; 10 judges (v3)</td>
                    <td>Qwen3-4B, NaiveRAG + BGE reranker</td>
                </tr>
                <tr>
                    <td>1 (Strategy &times; Model)</td>
                    <td>Does strategy compensate for model size?</td>
                    <td>5 strategies &times; 6 models = 30 configs &times; 200 questions</td>
                    <td>Recursive chunker (500&thinsp;/&thinsp;100),
                        embeddinggemma:300m, hybrid retrieval, no reranker</td>
                </tr>
                <tr>
                    <td>2 (Chunking &times; Model)</td>
                    <td>Does chunking strategy interact with model capability?</td>
                    <td>4 chunkers &times; 4 models = 16 configs &times; 200 questions</td>
                    <td>NaiveRAG, mxbai-embed-large, hybrid retrieval, no reranker</td>
                </tr>
            </tbody>
        </table>

        <h2>Dataset</h2>
        <p>The primary corpus is <strong>HotpotQA</strong> — 113K multi-hop Wikipedia
           question-answer pairs with gold-standard answers and difficulty labels (easy,
           medium, hard). Multi-hop questions require reasoning across multiple passages,
           making them a rigorous test of retrieval and generation quality.</p>
        <p>Secondary: SQuAD 2.0 (simple factoid Q&amp;A for easy-baseline calibration).</p>

        <h2 id="logging-gaps">Logging Gaps &amp; What We&rsquo;re Fixing</h2>
        <p>
            Experiments 0 through 2 ran before RAGBench stamped per-row
            LLM provenance onto the dataset &mdash; a real instrumentation
            oversight that matters most for the branchy strategies
            (CorrectiveRAG, AdaptiveRAG, Self-RAG). What we recorded per
            row was the <em>final answer text</em>, the <em>final context
            sent to the LLM</em>, total strategy latency, and gold-vs-RAG
            metrics &mdash; plenty for headline results. What we did
            <em>not</em> record:
        </p>
        <p class="muted-note">
            One thing that <em>isn&rsquo;t</em> on the list below: the
            prompt <em>templates</em>. Those live in
            <code>src/strategies/*.py</code> (e.g.&nbsp;
            <code>RELEVANCE_PROMPT</code>, <code>REPHRASE_PROMPT</code>,
            <code>CLASSIFY_PROMPT</code>, <code>CRITIQUE_PROMPT</code>),
            are version-controlled in git, and have not changed since
            the runs. So the prompt <em>shape</em> for every call is
            recoverable from the source. What&rsquo;s missing per row
            is laid out below.
        </p>
        <ul>
            <li>The <strong>literal substituted prompt strings as actually
                sent</strong> &mdash; including content that came from
                prior LLM outputs (the specific reformulated query
                Multi-Query produced for this row, the specific classifier
                verdict Adaptive returned, the per-chunk relevance ratings
                Corrective applied). For first-call-in-strategy prompts the
                substitution is deterministic given (template, retrieved
                chunks, query) and partially reconstructable; for any
                subsequent call whose input was a prior call&rsquo;s
                <em>output</em>, the substituted string is gone.</li>
            <li>The number of <strong>LLM calls per row</strong> &mdash;
                only the process-wide running total was snapshotted into
                the heartbeat log.</li>
            <li>The <strong>branch path taken</strong> &mdash; for
                Corrective, whether reformulation fired; for Adaptive,
                which sub-path the classifier chose; for Self-RAG,
                whether the critique pass changed the answer.</li>
            <li>The <strong>intermediate LLM outputs</strong> &mdash; the
                &ldquo;relevant&rdquo;/&ldquo;irrelevant&rdquo; ratings,
                the reformulated query text, the classifier&rsquo;s output.
                These drive the branch decisions and become the next
                call&rsquo;s prompt content, which is why item&nbsp;1 lists
                them as the unrecoverable part of the substituted prompts.</li>
            <li>The <strong>code SHA</strong> at generation time, so a
                later analyst can pin down which version of
                <code>src/strategies/</code> produced a given row.</li>
        </ul>
        <p>
            For NaiveRAG those omissions don&rsquo;t much matter
            &mdash; the strategy has one fixed template, one chat call,
            and a fully deterministic substituted prompt. For the other
            four they matter quite a bit. CorrectiveRAG conditions a
            second retrieval pass on the outcome of N intermediate LLM
            ratings; without recording those ratings we can&rsquo;t tell
            from a row whether it took the one-pass branch or the
            two-pass branch, which makes the cell unsuitable for the
            kind of fine-grained error analysis that distinguishes
            &ldquo;strategy didn&rsquo;t help&rdquo; from &ldquo;strategy
            helped on the rows where its second pass fired.&rdquo;
            That&rsquo;s the kind of gap that distinguishes a publishable
            experiment from a working demo, and we should own it openly
            rather than hide it behind a polished dashboard.
        </p>
        <p>
            <strong>Going forward (Phase 1, shipped):</strong> the runner
            now opens a per-row trace before each strategy run; every
            <code>OllamaLLM.generate</code> records its
            <em>intent</em>, full prompt, response, and latency into that
            trace; at row end the trace is written to a sidecar
            <code>traces.jsonl</code> alongside <code>raw_scores.csv</code>
            and a compact summary is written to seven new CSV columns
            (<code>n_llm_calls</code>, <code>llm_call_intents</code>,
            <code>final_prompt</code>, <code>prompts_source</code>,
            <code>code_sha</code>, <code>llm_num_predict</code>,
            <code>llm_keep_alive</code>). New rows produced from this
            commit forward carry <code>prompts_source = "recorded"</code>.
        </p>
        <p>
            <strong>Backfill (shipped):</strong> historical rows have been
            tagged <code>prompts_source =
            "reconstructed_minimal_2026-05-20"</code>. NaiveRAG rows have
            their <code>final_prompt</code> reconstructed deterministically
            from the byte-identical template + the recorded
            <code>context_sent_to_llm</code>. Multi-step strategy rows
            were <em>not</em> filled with fabricated prompts &mdash; the
            backfill stops at what we can prove. Anyone joining historical
            with new data should filter on <code>prompts_source</code>
            to keep the two regimes distinguishable.
        </p>
        <p>
            <strong>Phase 2 (next):</strong> the strategies themselves
            should expose a hyperparameter dict (CorrectiveRAG&rsquo;s
            &ldquo;reformulate if fewer than 2 pass&rdquo; threshold,
            Multi-Query&rsquo;s &ldquo;3 sub-queries&rdquo; count),
            stamped per row. Per-call token counts via Ollama&rsquo;s
            <code>/api/chat</code> stats would let us answer
            cost-vs-quality questions without estimating token counts
            from string length. Embedder calls deserve their own
            counter, separate from chat calls. The full severity-ranked
            audit of uncollected details lives on the
            <a href="writeup.html#audit">Writeup page</a>, with
            project motivation, hypotheses, headline findings, and the
            BSOD post-mortem for context.
        </p>

    </div>
    """
    return _build_page_template(
        "Methodology",
        nav_active="methodology",
        content_html=content,
    )


def _generate_data() -> str:
    """Downloads + column dictionary + example pandas snippets.

    A reference page for anyone who wants to verify or extend the
    analysis. Every CSV referenced here is generated from the live
    pipeline and copied into ``docs/`` at gallery-render time.
    """
    content = """
    <div class="methodology-content">

        <p class="kicker-mono" style="color: var(--ink-muted);">Data &middot; CSV downloads &amp; column dictionary</p>
        <p class="lede">
            Everything the gallery charts is built from raw per-row
            CSVs. They&rsquo;re published here so anyone can verify the
            numbers in the dashboards, reproduce the analyses, or run
            their own. Sidecar JSONL files carry the per-row LLM-call
            traces and the BSOD-resilience heartbeat events. Sizes are
            modest enough for direct download; the larger files
            (Experiments 1 and 2) are 20&ndash;30&nbsp;MB.
        </p>

        <h2 id="downloads">Downloadable datasets</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Rows</th>
                    <th>Size</th>
                    <th>Description</th>
                    <th>File</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Experiment 0 v1</td>
                    <td>50 questions &times; 6 judges = 300 scores</td>
                    <td>~25&nbsp;KB</td>
                    <td>First scorer-validation pass. Proportionally
                        sampled HotpotQA; ceiling effect, no
                        reranker. Superseded by v3 but preserved for
                        the methodological-evolution narrative.</td>
                    <td><a href="raw_scores.csv">raw_scores.csv</a></td>
                </tr>
                <tr>
                    <td>Experiment 0 v2</td>
                    <td>150 medium+hard questions &times; ~7 judges</td>
                    <td>~280&nbsp;KB</td>
                    <td>Corrected ceiling effect (medium+hard only),
                        added BGE reranker, scorer fed actual context
                        sent to LLM. Still single-judge headline.</td>
                    <td><a href="raw_scores_v2.csv">raw_scores_v2.csv</a></td>
                </tr>
                <tr>
                    <td>Experiment 0 v3</td>
                    <td>500 questions &times; up to 12 judges</td>
                    <td>~950&nbsp;KB</td>
                    <td>Definitive scorer validation. Surfaced the
                        model-version-drift result that changed the
                        Exp 1/2 panel to Claude Haiku 4.5 + GPT-5.4
                        Mini.</td>
                    <td><a href="raw_scores_v3.csv">raw_scores_v3.csv</a></td>
                </tr>
                <tr>
                    <td>Experiment 1</td>
                    <td>5 strategies &times; 6 models &times; 200 = 6,000 rows</td>
                    <td>~20&nbsp;MB</td>
                    <td>Strategy &times; Model Size factorial.
                        HotpotQA, RecursiveChunker(500/100),
                        embeddinggemma:300m, hybrid retrieval.</td>
                    <td><a href="experiment_1_raw_scores.csv">experiment_1_raw_scores.csv</a></td>
                </tr>
                <tr>
                    <td>Experiment 2</td>
                    <td>4 chunkers &times; 4 models &times; 200 = 3,200 rows</td>
                    <td>~30&nbsp;MB</td>
                    <td>Chunking &times; Model Size factorial.
                        HotpotQA, NaiveRAG, mxbai-embed-large, hybrid
                        retrieval.</td>
                    <td><a href="experiment_2_raw_scores.csv">experiment_2_raw_scores.csv</a></td>
                </tr>
            </tbody>
        </table>

        <h2 id="sidecars">Sidecar files</h2>
        <p>
            Two JSONL files travel alongside each experiment&rsquo;s
            CSV. They&rsquo;re not in the dashboards but they&rsquo;re
            essential for forensic deep-dives.
        </p>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Sidecar</th>
                    <th>Format</th>
                    <th>What&rsquo;s in it</th>
                    <th>File</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>events.jsonl</code> (Exp 1)</td>
                    <td>One JSON object per line, ~150&nbsp;KB</td>
                    <td>Heartbeat events from the runner:
                        <code>run_start</code>, <code>config_start</code>,
                        <code>row_done</code> (with running call total
                        + last in-flight call), <code>gpu_snapshot</code>,
                        <code>row_rest_start</code>/<code>tick</code>,
                        <code>config_end</code>, <code>run_end</code>.
                        Built to shrink the BSOD blast radius from one
                        snapshot-interval to one row.</td>
                    <td><a href="experiment_1_events.jsonl">experiment_1_events.jsonl</a></td>
                </tr>
                <tr>
                    <td><code>traces.jsonl</code> (Exp 1, going forward)</td>
                    <td>One JSON object per row, variable size</td>
                    <td>Per-row LLM-call trace: list of every
                        <code>OllamaLLM.generate</code> call the
                        strategy made, with intent label, prompt,
                        response, latency. Empty for rows produced
                        before the per-row instrumentation shipped
                        (see <a href="writeup.html#exp12">Writeup
                        &raquo; Exp 1/2 oversights</a>).</td>
                    <td><em>generated on next run</em></td>
                </tr>
            </tbody>
        </table>

        <h2 id="dictionary">Column dictionary</h2>
        <p>
            All columns that appear in the experiment CSVs, grouped
            by purpose. Not every column is present in every file
            &mdash; older runs predate some additions.
        </p>

        <h3>Identifiers</h3>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>strategy</code></td><td>string</td>
                    <td>RAG strategy name: <code>naive</code>,
                        <code>self_rag</code>, <code>multi_query</code>,
                        <code>corrective</code>, <code>adaptive</code>
                        (Exp 1 only).</td></tr>
                <tr><td><code>chunker</code></td><td>string</td>
                    <td>Chunking strategy: <code>fixed</code>,
                        <code>recursive</code>, <code>sentence</code>,
                        <code>semantic</code> (Exp 2 only).</td></tr>
                <tr><td><code>model</code></td><td>string</td>
                    <td>Ollama model tag used for generation.</td></tr>
                <tr><td><code>question</code></td><td>string</td>
                    <td>HotpotQA question text. Joins back to the
                        original dataset via exact match.</td></tr>
                <tr><td><code>dataset_name</code></td><td>string</td>
                    <td>Source dataset (always <code>hotpotqa</code>
                        currently).</td></tr>
                <tr><td><code>dataset_sample_seed</code></td><td>int</td>
                    <td>Random seed used to sample the question
                        subset (always <code>42</code>).</td></tr>
            </tbody>
        </table>

        <h3>Gold-standard metrics</h3>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>gold_answer</code></td><td>string</td>
                    <td>Reference answer from HotpotQA.</td></tr>
                <tr><td><code>rag_answer</code></td><td>string</td>
                    <td>The model&rsquo;s generated answer.</td></tr>
                <tr><td><code>gold_f1</code></td><td>float [0,1]</td>
                    <td>Token-level F1 between rag_answer and
                        gold_answer.</td></tr>
                <tr><td><code>gold_exact_match</code></td><td>bool</td>
                    <td>True if gold_answer appears verbatim in
                        rag_answer (case-insensitive substring).</td></tr>
                <tr><td><code>gold_bertscore</code></td><td>float [0,1]</td>
                    <td>BERTScore F1 between rag_answer and
                        gold_answer (microsoft/deberta-xlarge-mnli).</td></tr>
            </tbody>
        </table>

        <h3>Per-judge scorer metrics</h3>
        <p>
            Each judge in the panel contributes five columns, prefixed
            with a safe-name slug of the judge identifier.
            <code>&lt;judge&gt;</code> below stands for the slug
            (e.g.&nbsp;<code>anthropic_claude_haiku_4_5_20251001</code>,
            <code>openai_gpt_5_4_mini</code>). The number of judges
            varies by experiment &mdash; Exp 1/2 use a 2-judge panel,
            Exp 0 v3 ran up to 12 judges for the validation comparison.
        </p>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>&lt;judge&gt;_faithfulness</code></td>
                    <td>float [1,5]</td>
                    <td>Judge&rsquo;s 1&ndash;5 rating: how well the
                        rag_answer is supported by the retrieved context.</td></tr>
                <tr><td><code>&lt;judge&gt;_relevance</code></td>
                    <td>float [1,5]</td>
                    <td>How well the rag_answer addresses the
                        question.</td></tr>
                <tr><td><code>&lt;judge&gt;_conciseness</code></td>
                    <td>float [1,5]</td>
                    <td>Whether the rag_answer avoids unnecessary
                        verbosity.</td></tr>
                <tr><td><code>&lt;judge&gt;_quality</code></td>
                    <td>float [1,5]</td>
                    <td>Mean of the three sub-metrics &mdash; the
                        judge&rsquo;s composite quality score.</td></tr>
                <tr><td><code>&lt;judge&gt;_scorer_latency_ms</code></td>
                    <td>float</td>
                    <td>Wall-clock latency of this judge&rsquo;s
                        scoring call.</td></tr>
                <tr><td><code>consensus_quality</code></td>
                    <td>float [1,5]</td>
                    <td>Mean of per-judge <code>quality</code> across
                        the active panel &mdash; the primary metric
                        the dashboards use.</td></tr>
            </tbody>
        </table>

        <h3>Latency</h3>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>strategy_latency_ms</code></td><td>float</td>
                    <td>Time from <code>strategy.run()</code> call to
                        return &mdash; covers retrieval + every LLM
                        call the strategy made.</td></tr>
                <tr><td><code>scorer_latency_ms_total</code></td><td>float</td>
                    <td>Sum of per-judge scoring latencies.</td></tr>
                <tr><td><code>total_latency_ms</code></td><td>float</td>
                    <td>Sum of the two above. Useful for end-to-end
                        cost.</td></tr>
            </tbody>
        </table>

        <h3>Failure diagnostics</h3>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>context_sent_to_llm</code></td><td>string</td>
                    <td>The exact context (retrieved chunks
                        concatenated, post-filter) the answer-generation
                        call saw. Can be tens of KB &mdash; the bulk of
                        the row size.</td></tr>
                <tr><td><code>failure_stage</code></td><td>string</td>
                    <td>Substring-attribution: which pipeline stage
                        is most likely responsible for a wrong answer.
                        One of <code>none</code>, <code>chunker</code>,
                        <code>retrieval</code>, <code>filtering</code>,
                        <code>generation</code>, <code>unknown</code>.</td></tr>
                <tr><td><code>failure_stage_confidence</code></td>
                    <td>string</td>
                    <td>Confidence label for the stage attribution
                        (<code>high</code>, <code>medium</code>,
                        <code>low</code>, <code>n/a</code>).</td></tr>
                <tr><td><code>failure_stage_method</code></td>
                    <td>string</td>
                    <td>Detection method used &mdash; currently
                        <code>substring</code>.</td></tr>
                <tr><td><code>gold_in_chunks</code></td><td>bool</td>
                    <td>True if the gold answer appears in any chunk
                        the chunker produced.</td></tr>
                <tr><td><code>gold_in_retrieved</code></td><td>bool</td>
                    <td>True if the gold answer appears in any chunk
                        that came back from retrieval.</td></tr>
                <tr><td><code>gold_in_context</code></td><td>bool</td>
                    <td>True if the gold answer appears in the
                        post-filter context sent to the model.</td></tr>
            </tbody>
        </table>

        <h3>Pipeline metadata (held constant within an experiment)</h3>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>chunk_type</code></td><td>string</td>
                    <td>Chunker family: <code>recursive</code>,
                        <code>fixed</code>, <code>sentence</code>,
                        <code>semantic</code>.</td></tr>
                <tr><td><code>chunk_size</code></td><td>int</td>
                    <td>Target chunk size in tokens (where applicable).</td></tr>
                <tr><td><code>chunk_overlap</code></td><td>int</td>
                    <td>Overlap between adjacent chunks in tokens.</td></tr>
                <tr><td><code>num_chunks</code></td><td>int</td>
                    <td>Total chunks the chunker produced for the
                        source document.</td></tr>
                <tr><td><code>embed_provider</code></td><td>string</td>
                    <td>Embedder provider, e.g.&nbsp;<code>ollama</code>.</td></tr>
                <tr><td><code>embed_model</code></td><td>string</td>
                    <td>Embedder model tag.</td></tr>
                <tr><td><code>embed_dimension</code></td><td>int</td>
                    <td>Output dimension of the embedder.</td></tr>
                <tr><td><code>retrieval_mode</code></td><td>string</td>
                    <td>Retrieval mode: <code>hybrid</code>,
                        <code>dense</code>, or <code>sparse</code>.</td></tr>
                <tr><td><code>retrieval_top_k</code></td><td>int</td>
                    <td>How many chunks the retriever returned.</td></tr>
                <tr><td><code>num_chunks_retrieved</code></td><td>int</td>
                    <td>Actual number of chunks in
                        <code>context_sent_to_llm</code>
                        (may differ from <code>retrieval_top_k</code>
                        for strategies that filter).</td></tr>
                <tr><td><code>context_char_length</code></td><td>int</td>
                    <td>Character length of
                        <code>context_sent_to_llm</code>.</td></tr>
                <tr><td><code>reranker_model</code></td><td>string|null</td>
                    <td>Reranker model name (null when no reranker
                        was used &mdash; the default for Exp 1 and 2).</td></tr>
                <tr><td><code>reranker_top_k</code></td><td>int|null</td>
                    <td>Top-k after reranking.</td></tr>
                <tr><td><code>llm_provider</code></td><td>string</td>
                    <td>LLM provider (always <code>ollama</code>).</td></tr>
                <tr><td><code>llm_host</code></td><td>string</td>
                    <td>Ollama host URL or <code>local</code>.</td></tr>
                <tr><td><code>llm_model</code></td><td>string</td>
                    <td>Ollama tag used for generation.</td></tr>
                <tr><td><code>llm_quantization</code></td><td>string</td>
                    <td>Quantization level reported by Ollama
                        <code>/api/show</code> at runtime (e.g.&nbsp;<code>Q4_K_M</code>,
                        <code>Q8_0</code>), or <code>unknown</code>.</td></tr>
            </tbody>
        </table>

        <h3>LLM-call provenance (Phase 1 instrumentation, going forward)</h3>
        <p>
            Added with the per-row trace work documented in the
            <a href="writeup.html#exp12">Writeup&rsquo;s Exp 1/2
            oversights section</a>. Rows produced before that
            instrumentation shipped have <code>prompts_source</code>
            set to <code>reconstructed_minimal_2026-05-20</code>;
            new rows are tagged <code>recorded</code>.
        </p>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Type</th><th>Meaning</th></tr>
            </thead>
            <tbody>
                <tr><td><code>n_llm_calls</code></td><td>int</td>
                    <td>Number of LLM chat completions the strategy
                        made for this row. For NaiveRAG always 1;
                        variable for branchy strategies.</td></tr>
                <tr><td><code>llm_call_intents</code></td><td>string</td>
                    <td>Pipe-separated intent labels in call order,
                        e.g.&nbsp;<code>rate_relevance|rate_relevance|generate_answer</code>.
                        See <a href="writeup.html#exp12">Writeup</a> for
                        the label vocabulary per strategy.</td></tr>
                <tr><td><code>final_prompt</code></td><td>string</td>
                    <td>The literal substituted prompt sent to the
                        answer-generation call (the last call in the
                        trace). Reconstructed for NaiveRAG backfill
                        rows; recorded verbatim for new rows.</td></tr>
                <tr><td><code>prompts_source</code></td><td>string</td>
                    <td><code>recorded</code> for rows produced by the
                        instrumented runner;
                        <code>reconstructed_minimal_2026-05-20</code>
                        for historical backfilled rows.</td></tr>
                <tr><td><code>code_sha</code></td><td>string</td>
                    <td>7-char git HEAD SHA at generation time, so an
                        analyst can pin which version of
                        <code>src/strategies/</code> produced the row.
                        Empty on backfilled rows.</td></tr>
                <tr><td><code>llm_num_predict</code></td><td>int</td>
                    <td>Token-generation cap passed to Ollama
                        (<code>OllamaLLM.DEFAULT_NUM_PREDICT</code>,
                        currently 1024).</td></tr>
                <tr><td><code>llm_keep_alive</code></td><td>string</td>
                    <td>Model-residency hint passed to Ollama
                        (<code>30m</code>).</td></tr>
                <tr><td><code>n_total_calls_inc_embed</code></td>
                    <td>int</td>
                    <td>Backfill-only: total Ollama calls (chat +
                        embed) attributable to this row, derived from
                        <code>events.jsonl</code> deltas. Not a
                        substitute for the precise
                        <code>n_llm_calls</code> &mdash; it lumps
                        embedder calls in.</td></tr>
                <tr><td><code>n_token_*</code> &middot; <em>not yet present</em></td>
                    <td>&mdash;</td>
                    <td>Per-call prompt / completion token counts.
                        See the <a href="writeup.html#audit">audit table
                        on Writeup</a> row 4 &mdash; on the Phase 2
                        backlog.</td></tr>
            </tbody>
        </table>

        <h2 id="usage">Example usage</h2>
        <p>
            A few snippets for common analyses. They assume
            <code>pip install pandas</code> and that you&rsquo;ve
            downloaded one of the CSVs above.
        </p>

        <h3>Load Experiment 1 and reproduce the per-strategy ranking</h3>
        <pre><code>import csv, pandas as pd

# raw_scores.csv has rows with multi-KB context fields; raise the limit
# or pandas will choke on the largest cells.
csv.field_size_limit(2**31 - 1)

df = pd.read_csv("experiment_1_raw_scores.csv", engine="python")

ranking = (
    df.groupby("strategy")["consensus_quality"]
      .agg(["mean", "std", "count"])
      .round(3)
      .sort_values("mean", ascending=False)
)
print(ranking)
# strategy      mean    std  count
# naive        4.340  0.840   1199
# multi_query  4.334  0.818   1200
# corrective   4.268  0.881   1200
# adaptive     3.594  1.252   1200
# self_rag     2.992  1.073   1196</code></pre>

        <h3>Filter to rows produced by the new instrumentation</h3>
        <pre><code># Recorded rows have the full per-call trace; reconstructed rows are
# schema-aligned but don't carry intermediate LLM outputs.
recorded = df[df["prompts_source"] == "recorded"]
backfilled = df[df["prompts_source"].str.startswith("reconstructed", na=False)]
print(f"recorded: {len(recorded)}  backfilled: {len(backfilled)}")</code></pre>

        <h3>Join the per-row LLM-call trace back to the CSV</h3>
        <pre><code>import json

traces = {}
with open("experiment_1_traces.jsonl") as fh:
    for line in fh:
        rec = json.loads(line)
        traces[(rec["strategy"], rec["model"], rec["question"])] = rec["trace"]

def get_trace(row):
    return traces.get((row["strategy"], row["model"], row["question"]), [])

df["trace"] = df.apply(get_trace, axis=1)
df["n_chat_calls_recorded"] = df["trace"].str.len()
# For recorded rows this should equal n_llm_calls; for backfilled rows
# trace will be [] and n_chat_calls_recorded will be 0.</code></pre>

        <h3>Compare model size on a fixed strategy</h3>
        <pre><code>fixed_strategy = df[df["strategy"] == "naive"]
by_model = (
    fixed_strategy.groupby("model")[["consensus_quality", "gold_f1"]]
      .mean()
      .round(3)
      .sort_values("consensus_quality", ascending=False)
)
print(by_model)</code></pre>

        <h2 id="licensing">Licensing &amp; attribution</h2>
        <p>
            The CSVs are derivatives of HotpotQA, which is
            <a href="https://hotpotqa.github.io/" target="_blank" rel="noopener">CC-BY-SA 4.0</a>;
            the derived data is shared under the same license.
            Generated answers are model outputs &mdash; treat them as
            you would any LLM output. The pipeline code that produced
            them is on
            <a href="https://github.com/nono638/2026Project" target="_blank" rel="noopener">GitHub</a>
            under that repository&rsquo;s license.
        </p>

    </div>
    """
    return _build_page_template(
        "Data",
        nav_active="data",
        content_html=content,
    )


def _generate_engineering() -> str:
    """Full BSOD post-mortem + stability engineering notes.

    Linked from the Writeup&rsquo;s plain-English summary; not in the
    global nav (would be jargon-heavy for casual visitors).
    """
    content = """
    <div class="methodology-content">

        <p class="kicker-mono" style="color: var(--ink-muted);">Engineering notes &middot; the technical detail</p>
        <p class="lede">
            This is the full account of the hardware-stability problems
            that nearly stopped Experiments 1 and 2, and the
            engineering layers that eventually made the matrix
            complete. It&rsquo;s the technical companion to the
            plain-English summary on
            <a href="writeup.html#bsods">Writeup&rsquo;s &ldquo;The
            technical reality&rdquo;</a>. Skip it unless you care
            about Windows kernel bugcheck codes, Ollama internals, or
            unattended-experiment supervision patterns.
        </p>

        <h2 id="bsods">Six BSODs in eleven days</h2>
        <p>
            The hardware was an RTX 5090 Laptop GPU on Windows 11.
            Between 2026-05-17 and 2026-05-19 the machine crashed six
            times under sustained Ollama batch load. Five distinct
            bugcheck codes appeared, all rooted in the NVIDIA display
            driver (<code>nvlddmkm</code>).
        </p>
        <table class="data-table">
            <thead>
                <tr><th>Date / time</th><th>Bugcheck</th><th>Symptom &amp; phase of run</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>2026-05-17 22:32</td>
                    <td><code>0x00000133</code> DPC_WATCHDOG_VIOLATION</td>
                    <td>nvlddmkm DPC stayed in a single CPU for too
                        long. First crash &mdash; surfaced the pattern.</td>
                </tr>
                <tr>
                    <td>2026-05-18 02:14</td>
                    <td><code>0x0000003B</code> SYSTEM_SERVICE_EXCEPTION</td>
                    <td>Resumed Exp 1 ~32 min after restart.
                        nvlddmkm stack again.</td>
                </tr>
                <tr>
                    <td>2026-05-18 09:21</td>
                    <td><code>0x0000000A</code> IRQL_NOT_LESS_OR_EQUAL</td>
                    <td>Standard hardware-driver fault. Resumed run.</td>
                </tr>
                <tr>
                    <td>2026-05-18 11:53</td>
                    <td><code>0x00000116</code> VIDEO_TDR_ERROR</td>
                    <td>Windows&rsquo; Timeout Detection &amp; Recovery
                        fired when the GPU stopped responding inside an
                        LLM kernel. Config 18/30
                        (multi_query &times; qwen3.5:9b @ query 69).</td>
                </tr>
                <tr>
                    <td>2026-05-18 ??:??</td>
                    <td><em>bugcheck not preserved</em></td>
                    <td>BSOD #5 &mdash; minidump was rolled out before
                        triage.</td>
                </tr>
                <tr>
                    <td>2026-05-19 03:29</td>
                    <td><em>silent kill</em></td>
                    <td>No Python traceback, log just stops mid-row_rest
                        sleep. Config 29/30
                        (corrective &times; gemma4:e4b @ query 100).
                        Classic BSOD silhouette.</td>
                </tr>
            </tbody>
        </table>
        <p>
            The NVIDIA developer forum has a long thread documenting
            similar behaviour on the 5090 family under Ollama load:
            <a href="https://forums.developer.nvidia.com/t/rtx-5090-total-failure-hang-unload-in-ollama-after-couple-of-minutes/341659"
               target="_blank" rel="noopener">RTX 5090 total failure /
            hang / unload in Ollama after a couple of minutes</a>. The
            consensus across that thread is that this is a driver
            and firmware-level instability cluster on the 5090 SKU,
            triggered (not caused) by sustained mixed embed / chat
            workloads.
        </p>

        <h2 id="software-pacing">Layer 1: software pacing inside the experiment runner</h2>
        <p>
            None of these eliminated the BSODs but they cut frequency
            enough to make progress between crashes. All live in
            <code>scripts/run_experiment_1.py</code>.
        </p>
        <ul>
            <li><strong>Row pace:</strong> 0.5&nbsp;s sleep between
                every completed (generate&nbsp;+&nbsp;score) row,
                configurable via <code>--row-pace-s</code>. Inserted
                between bursts so the GPU could drain its DPC queue
                before the next embed/chat hit.</li>
            <li><strong>Periodic mid-config rest:</strong> originally
                a single 10&nbsp;s sleep every 50 rows. Cut to
                3&nbsp;&times;&nbsp;1&nbsp;s ticks (with an fsynced
                <code>row_rest_tick</code> heartbeat each second) after
                BSOD #6 hit mid-sleep &mdash; the 10&nbsp;s monolithic
                idle was a fat window for ASPM downshift to occur and
                wasn&rsquo;t earning anything (GPU sat at 51&ndash;56&nbsp;&deg;C
                throughout).</li>
            <li><strong>Inter-config cooldown:</strong> 30&nbsp;s
                between configs. Gave the driver clean unload time
                before the next model loaded.</li>
            <li><strong>Per-row CSV flush + fsync:</strong> the runner
                writes a single row to <code>raw_scores.csv</code>
                and forces it to disk before moving on. A power loss
                costs at most one row instead of one config (~25
                minutes of work).</li>
            <li><strong>Per-row JSONL heartbeat:</strong>
                <code>events.jsonl</code> gets a <code>row_done</code>
                record with the running <code>calls_total</code> and
                the last in-flight call descriptor. Post-mortem can
                identify the row and call type the BSOD interrupted.</li>
            <li><strong>Periodic GPU snapshot:</strong> every 10 rows
                a <code>gpu_snapshot</code> event records temp /
                utilization / VRAM / power so thermal-correlated
                crashes can be reconstructed from the trail.</li>
        </ul>

        <h2 id="ollama-tuning">Layer 2: Ollama server tuning</h2>
        <p>
            The default Ollama install on Windows runs as a tray app
            with <code>OLLAMA_MAX_LOADED_MODELS=3</code> and
            <code>OLLAMA_NUM_PARALLEL</code> set by the server&rsquo;s
            heuristics. The concurrent embedder/chat residency was the
            allocator-churn regime that correlated with the worst
            BSOD cluster. The fix is to stop the tray-spawned server
            and re-launch <code>ollama serve</code> with explicit env
            vars &mdash; the tray ignores env vars set after launch.
        </p>
        <p>
            <code>scripts/start_ollama_safe.ps1</code> does this. The
            vars it sets:
        </p>
        <ul>
            <li><code>OLLAMA_MAX_LOADED_MODELS=1</code> &mdash; forces
                a clean unload/reload between embedder and chat
                instead of keeping both resident.</li>
            <li><code>OLLAMA_NUM_PARALLEL=1</code> &mdash; our
                pipeline is serial; this disables Ollama&rsquo;s
                per-slot KV duplication.</li>
            <li><code>OLLAMA_KEEP_ALIVE=24h</code> &mdash; well past
                any single experiment config; eviction still happens
                on model-switch (Ollama&rsquo;s
                <code>MAX_LOADED_MODELS=1</code> rule supersedes
                keep_alive when slots are full).</li>
            <li><code>OLLAMA_MAX_QUEUE=512</code> &mdash; explicit so
                it survives a default change.</li>
            <li><code>OLLAMA_MODELS=D:\\OllamaModels</code> &mdash;
                the tray app reads this from its own config, not from
                env vars; passing it explicitly to the replacement
                <code>ollama serve</code> avoids the fallback to
                <code>%USERPROFILE%\\.ollama\\models</code> that
                otherwise reports every model as 404.</li>
        </ul>
        <p>
            Two settings deliberately <em>not</em> set:
            <code>OLLAMA_FLASH_ATTENTION</code> (auto-disabled when an
            embed model is loaded, so a mixed pipeline ignores it
            anyway) and <code>OLLAMA_KV_CACHE_TYPE</code> (q8_0 KV
            silently falls back to f16 on archs that don&rsquo;t
            support it, masking what&rsquo;s actually running).
        </p>

        <h2 id="client-tuning">Layer 3: Ollama client tuning</h2>
        <p>
            In <code>src/llms/ollama.py</code>:
        </p>
        <ul>
            <li><strong><code>keep_alive=&quot;30m&quot;</code> per
                call.</strong> Ollama&rsquo;s default of 5 minutes
                let the timer expire mid-config when a slow
                generation (or judge-side stall) ran long; that
                forced a full reload on the next call. 30&nbsp;m
                safely spans worst-case config duration without
                locking the GPU when the experiment moves to the
                next model.</li>
            <li><strong><code>num_predict=1024</code></strong> cap on
                tokens per generation. Without this, qwen3.5:9b was
                observed generating runaway responses of 8&ndash;17+
                minutes on single queries (the regime that drove
                BSOD #3). Sized from analysis of completed Exp 1 rows
                across all 5 strategies &mdash; 1024 covers
                &gt;95% of non-degenerate answers while keeping
                worst-case generation well under the 180&nbsp;s
                client timeout.</li>
            <li><strong>180&nbsp;s per-call httpx timeout.</strong>
                The Ollama Python client defaults to no timeout,
                which is how an 8&ndash;17-minute hung
                <code>chat()</code> happened before BSOD #3. 180&nbsp;s
                is 3&times; the expected worst-case for a 9B model
                on the 5090 Laptop.</li>
            <li><strong>Runner-crash recovery wait.</strong>
                <code>experiment_utils.py</code> recognises Ollama&rsquo;s
                &ldquo;model runner has unexpectedly stopped&rdquo;
                error and waits 60&nbsp;s before retrying (the runner
                process needs ~30&ndash;90&nbsp;s to come back up
                cleanly after a crash). Normal 2&ndash;8&nbsp;s
                backoff hits a dead runner and hangs again.</li>
        </ul>

        <h2 id="system-tuning">Layer 4: Windows / NVIDIA driver settings</h2>
        <p>
            <code>scripts/apply_5090_stability.ps1</code> (run as
            Administrator, then reboot):
        </p>
        <ul>
            <li><strong>TDR registry keys raised.</strong>
                <code>TdrDelay=60</code>,
                <code>TdrDdiDelay=60</code>,
                <code>TdrLevel=3</code> under
                <code>HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers</code>.
                Default is 2&nbsp;s / 5&nbsp;s, which is too short for
                LLM kernels that legitimately hold the GPU through
                long generations &mdash; was triggering 0x116
                VIDEO_TDR_ERROR (BSOD #4). Raised to 60&nbsp;s with
                full recovery enabled (don&rsquo;t set
                <code>TdrLevel=0</code>; that masks bugs instead of
                surviving them).</li>
            <li><strong>Hardware-Accelerated GPU Scheduling off.</strong>
                <code>HwSchMode=1</code>. Implicated in 5090
                nvlddmkm DPC instability and in the NVIDIA dev forum
                thread cited above. NVIDIA driver re-installs
                sometimes flip this back on &mdash; check after any
                driver update.</li>
            <li><strong>PCIe ASPM disabled.</strong>
                <code>powercfg /setacvalueindex SCHEME_CURRENT
                501a4d13-... ee12f906-... 0</code>. ASPM transitions
                on dGPU laptops are a documented trigger for
                nvlddmkm stalls, and our per-row pacing was creating
                idle windows where ASPM could downshift. Plus a
                forced high-performance power plan so the GPU
                isn&rsquo;t throttled mid-call.</li>
        </ul>
        <p>
            Things that <em>didn&rsquo;t</em> help: a clean NVIDIA
            driver reinstall (the issue is cross-driver-version), and
            chasing &ldquo;cooler temperatures&rdquo; (the GPU sat at
            51&ndash;56&nbsp;&deg;C through the crashed runs &mdash;
            thermal headroom wasn&rsquo;t the issue).
        </p>

        <h2 id="watchdog">Layer 5: unattended supervision</h2>
        <p>
            The lower four layers reduced BSOD <em>frequency</em>.
            Layer 5 reduced BSOD <em>cost</em>: when a crash did
            happen, the run resumed itself rather than waiting for a
            human to notice and re-launch.
        </p>
        <ul>
            <li><strong>Watchdog wrapper.</strong>
                <code>scripts/run_experiment_1_watchdog.py</code>
                supervises the experiment as a subprocess. On any
                non-zero exit it re-launches with
                <code>--resume</code>; the runner&rsquo;s row-level
                CSV checkpoint means at most one row is repeated.
                Exponential backoff caps at 120&nbsp;s. A zero-progress
                abort guard stops the loop if two consecutive
                attempts add no rows &mdash; protects against a
                deterministic Python bug burning through the restart
                budget. Clean exit at 6,000 rows.</li>
            <li><strong>Auto-resume after reboot.</strong>
                <code>scripts/install_auto_resume_task.ps1</code>
                registers a Windows Task Scheduler entry
                (<code>RagBench-Experiment1-AutoResume</code>) that
                fires the watchdog at logon with a 30&nbsp;s delay.
                Closes the human-in-loop gap entirely: after a BSOD,
                log back in, wait ~30&nbsp;s, the experiment resumes
                from the last checkpointed row.</li>
        </ul>
        <p>
            With all five layers in place, the final 300 rows of
            Experiment 1 completed in 46 minutes on a single
            watchdog attempt, no BSOD, no human intervention &mdash;
            after weeks of fighting that section of the matrix.
        </p>

        <h2 id="lessons">Lessons that generalise</h2>
        <ol>
            <li><strong>Own the unattended-execution loop.</strong>
                Don&rsquo;t assume the machine will be reliable for the
                wall-clock duration of an experiment. For multi-day
                workloads, a watchdog plus auto-resume after reboot
                is the difference between &ldquo;completes overnight&rdquo;
                and &ldquo;needs daily intervention.&rdquo;</li>
            <li><strong>Shrink the BSOD blast radius.</strong> Per-row
                fsynced CSV checkpoints + per-row JSONL heartbeats
                let any post-mortem answer &ldquo;what was the process
                doing at the second of death?&rdquo; This is cheap to
                add and pays back hugely on every crash.</li>
            <li><strong>System-level fixes and software pacing are
                complements, not substitutes.</strong> Either alone
                was insufficient on this hardware; together they
                made the run viable.</li>
            <li><strong>Default settings on consumer hardware are
                tuned for desktop interactivity, not batch ML.</strong>
                Windows TDR, GPU scheduling, PCIe ASPM, and Ollama&rsquo;s
                tray-app defaults all needed override before a
                multi-hour Ollama batch could finish reliably.</li>
            <li><strong>Forum threads are diagnostic data.</strong>
                The NVIDIA dev forum thread on the 5090 + Ollama
                pattern was more useful for narrowing the bug
                hypothesis than any single official source. When
                bugcheck codes look unfamiliar, search the
                workload + hardware combination on the appropriate
                vendor forum before deeper triage.</li>
        </ol>

        <p class="muted-note" style="margin-top: 28px;">
            All five layers above are in the source tree and can be
            re-applied to a future run: pacing constants in
            <code>scripts/run_experiment_1.py</code>; Ollama wrapper
            in <code>src/llms/ollama.py</code>; system mitigations in
            <code>scripts/apply_5090_stability.ps1</code> and
            <code>scripts/start_ollama_safe.ps1</code>; supervisor in
            <code>scripts/run_experiment_1_watchdog.py</code> +
            <code>scripts/install_auto_resume_task.ps1</code>. Back
            to <a href="writeup.html#bsods">the Writeup summary</a> or
            on to the rest of the
            <a href="writeup.html">project writeup</a>.
        </p>

    </div>
    """
    return _build_page_template(
        "Engineering Notes",
        nav_active="",  # not in nav; deep-link target only
        content_html=content,
    )


def _generate_writeup() -> str:
    """Long-form project writeup — motivation, findings, BSODs, oversights.

    Lives separately from Methodology because it's a reflective essay,
    not a reference doc. The audience is a recruiter or faculty reader
    who wants to know what RAGBench is, what we learned (good and bad),
    and how honestly to read the dashboards.
    """
    content = """
    <div class="methodology-content">

        <p class="kicker-mono" style="color: var(--ink-muted);">Project writeup &middot; CUNY SPS, Spring 2026</p>
        <p class="lede">
            RAGBench started as a question that seemed easy enough to
            measure: <em>does a smarter Retrieval-Augmented Generation
            strategy actually compensate for using a smaller language
            model?</em> Six BSODs, three revisions of the scorer
            validation, two full strategy &times; model factorials, and
            one painful realisation about uncollected provenance
            later, the answer is &ldquo;mostly no &mdash; but
            the experiment that taught us that needed more honesty about
            its own limits than the first dashboard suggested.&rdquo;
        </p>

        <h2 id="motivation">Why RAGBench</h2>
        <p>
            The RAG literature is loud about strategy choice. Every few
            weeks a new paper or blog post argues that some flavour of
            self-critique, query reformulation, or adaptive routing
            unlocks meaningful quality on a smaller, cheaper model. The
            claims are plausible &mdash; small models hallucinate more,
            and an extra LLM pass to verify or refine should help. But
            the numbers cited are usually on different datasets, with
            different models, against different retrievers, scored by
            different judges. There&rsquo;s nowhere obvious to look and
            ask: <strong>under controlled conditions, on a shared
            question set, with everything else held constant, how big
            is the strategy effect actually?</strong>
        </p>
        <p>
            RAGBench is a configurable harness that tries to answer that
            question for one shared question set (HotpotQA), one held-
            constant embedder per experiment, one chunker per
            experiment, and a panel of LLM judges that we first
            validated against gold-standard metrics. It runs the full
            cartesian product of (chunker &times; embedder &times;
            strategy &times; LLM) configurations specified on the
            command line and writes per-row results to a CSV that
            downstream analysis reads. The codebase is on
            <a href="https://github.com/nono638/2026Project" target="_blank" rel="noopener">GitHub</a>;
            it&rsquo;s very much a personal research project, not yet
            something I&rsquo;d ask a stranger to install and trust.
        </p>

        <h2 id="hypotheses">What we set out to test</h2>
        <p>
            Three working hypotheses going in. I&rsquo;ll mark each
            with what the data eventually said.
        </p>
        <ol>
            <li><strong>H1.</strong> A clever RAG strategy on a small
                language model will frequently beat plain NaiveRAG on
                a larger model. <span class="muted-note">&mdash; Result:
                <em>mostly false</em>. The &ldquo;Strategy Beats Size&rdquo;
                chart in Experiment 1 finds upset cases, but the
                average delta is +0.08 to +0.10 quality points and
                the strategies that produce them (Corrective,
                Multi-Query) sit essentially flat with Naive on
                average. Self-RAG and Adaptive actively hurt every
                model they touched.</span></li>
            <li><strong>H2.</strong> Quality will scale roughly
                monotonically with model size. <span class="muted-note">
                &mdash; Result: <em>directionally true but flat in the
                middle</em>. Going from 0.8B to 9B parameters
                moved mean quality 3.27 &rarr; 4.03 &mdash; meaningful
                at the bottom, but qwen3.5:4B (4.05) edged out
                qwen3.5:9B (4.03) in both Experiment 1 and Experiment 2.
                The 4B mark is the quality sweet spot for this
                workload.</span></li>
            <li><strong>H3.</strong> Chunker choice will interact
                meaningfully with model size &mdash; better chunking
                will rescue smaller models. <span class="muted-note">
                &mdash; Result: <em>false</em>. Within any one model the
                chunker spread is 0.16&ndash;0.27 quality points,
                roughly at the noise floor of stochastic generation
                (see below). Across chunkers the headline differences
                are smaller than the smallest model-size step.</span></li>
        </ol>

        <h2 id="headline">Headline findings, cross-experiment</h2>
        <p>
            Pulled together from the two experiment dashboards:
        </p>
        <ul>
            <li><strong>Strategy effects are small at best, hurtful at
                worst.</strong> The three strategies that didn&rsquo;t
                fight the model (Naive, Multi-Query, Corrective)
                cluster within 0.07 quality points across 1,200 rows
                each. The two that tried to be clever in
                under-instrumented ways (Self-RAG&rsquo;s
                self-critique loop, Adaptive&rsquo;s complexity
                router) cost 0.7&ndash;1.4 quality points on every
                model. Those negative results are the most
                publishable thing in the dataset.</li>
            <li><strong>Model size matters less than expected, and
                non-monotonically.</strong> qwen3.5:4B beat
                qwen3.5:9B in <em>both</em> experiments. Either the
                9B variant is poorly calibrated for HotpotQA, or
                we&rsquo;re into the regime where more parameters
                hurt as much as they help on simple factoid Q&amp;A.</li>
            <li><strong>Chunker choice barely moves the needle on
                HotpotQA.</strong> The best chunker (fixed) and the
                worst (sentence) differ by 0.20 quality points
                aggregate. Spending engineering effort on chunker
                tuning is unlikely to pay back vs spending it on
                model selection or, frankly, on writing better
                prompts.</li>
            <li><strong>LLM judges agree when they&rsquo;re
                contemporaneous.</strong> Experiment 0 v3 ran 9 judges
                on 500 questions; cross-provider Pearson against gold
                F1 was 0.55&ndash;0.60 for current-generation models
                (GPT-5.4, Claude Sonnet 4.6) and dropped 30&ndash;45%
                for older variants of the same families. <em>Model
                version drift dominates provider differences</em>
                &mdash; an under-appreciated property of the
                LLM-as-judge approach.</li>
            <li><strong>The winning configurations are boring.</strong>
                Experiment 1 winner: NaiveRAG &times; qwen3.5:4B
                (4.59 quality, 59% exact match). Experiment 2
                winner: fixed-size chunker &times; qwen3.5:4B
                (4.65 quality, 74% exact match). No clever strategy,
                no clever chunker, mid-size model. The interesting
                content is in the absence of the expected effects,
                not in the rankings themselves.</li>
        </ul>

        <h2 id="bsods">The technical reality: the laptop kept crashing</h2>
        <p>
            The biggest non-research problem in the whole project was
            that the laptop running everything kept crashing. Over
            three days in mid-May the machine blue-screened
            <strong>six times</strong> while in the middle of long
            batch experiments. Each crash forced a reboot and lost
            whatever the experiment was working on at that second.
            The cause turned out to be a known stability problem with
            the laptop&rsquo;s graphics card and its driver when
            running language models for long stretches &mdash; not a
            bug in RAGBench&rsquo;s code, but very much our problem to
            work around.
        </p>
        <p>
            Solving it took five layers of fixes stacked on top of
            each other: slowing down the rate at which RAGBench called
            the language model, tuning Ollama (the local model server)
            to use less aggressive defaults, raising some Windows
            timeout settings that were too short for long
            model-generation kernels, saving progress to disk after
            every single row so a crash cost at most one row of work,
            and finally wrapping the whole experiment in a supervisor
            program that automatically restarts and resumes the run
            after the laptop reboots. With all five in place, the last
            300 rows of Experiment 1 finished in 46 minutes on a
            single uninterrupted attempt &mdash; after weeks of
            fighting the same section of the matrix.
        </p>
        <p>
            The broader lesson: on consumer hardware, don&rsquo;t
            trust that your machine will stay up for the whole
            duration of a long ML experiment. Write the experiment as
            if it will be interrupted, because eventually it will.
        </p>
        <p style="margin-top: 18px;">
            <a href="engineering.html" class="cta-btn">
                Click here for more technical details &rarr;
            </a>
        </p>
        <p class="muted-note">
            The detail page covers the specific bugcheck codes,
            Ollama environment vars, Windows registry settings,
            client-side timeouts, watchdog architecture, and the
            scheduled-task auto-resume pattern. Read it if you care
            about kernel-level diagnostics or unattended-experiment
            engineering; skip it otherwise.
        </p>

        <h2 id="exp0">What we got wrong: Experiment 0&rsquo;s three revisions</h2>
        <p>
            Experiment 0&rsquo;s job was to pick the LLM judge for
            Experiments 1 and 2. The first version was sound enough
            to <em>answer the literal question asked</em> but had five
            methodological oversights that only surfaced under
            post-analysis. The reason there are three versions of
            Experiment 0 on this site is that we kept catching them and
            fixing them.
        </p>
        <h3>v1: the first end-to-end run</h3>
        <p>
            50 HotpotQA questions sampled proportionally across
            difficulties, scored by 6 LLM judges. Headline conclusion:
            use Claude Sonnet as the primary scorer. The five
            oversights:
        </p>
        <ol>
            <li><strong>Pipeline observability was absent.</strong> We
                generated answers but didn&rsquo;t record which chunks
                were retrieved, what context was assembled, or what
                the LLM actually saw. When a row failed, &ldquo;why?&rdquo;
                was guesswork.</li>
            <li><strong>The scorer judged against information the LLM
                never received.</strong> We passed the full source
                document to the judge for the faithfulness rating, but
                the answering model only saw the small subset of
                retrieved chunks. A hallucinated detail that happened
                to appear elsewhere in the document scored as
                &ldquo;faithful.&rdquo;</li>
            <li><strong>No reranker.</strong> v1 used raw hybrid
                retrieval (BM25 + dense with RRF fusion) with no
                cross-encoder reranker, which meant we validated our
                scorers on a different pipeline configuration than the
                one Experiments 1 and 2 would use.</li>
            <li><strong>Ceiling effect.</strong> 50 questions
                proportionally sampled meant ~74% exact match. With
                ~13 wrong answers there wasn&rsquo;t enough signal
                to discriminate between judges.</li>
            <li><strong>No composite quality metric.</strong> We had
                BERTScore, F1, and judge ratings as independent
                signals but no way to ask &ldquo;do all three agree
                this answer is good?&rdquo; A judge that gave 5/5 to
                an answer with low BERTScore and low F1 had a blind
                spot we couldn&rsquo;t see.</li>
        </ol>
        <h3>v2: corrections, but still single-judge thinking</h3>
        <p>
            150 medium+hard questions (eliminating the ceiling effect),
            BGE reranker added, scorer fed the actual context the LLM
            received, pipeline diagnostics captured. Conclusion stood
            (Sonnet wins) but with the caveat that we were still
            evaluating one judge at a time against gold metrics. The
            insight that <em>judges should be evaluated as a panel,
            cross-validating each other</em>, hadn&rsquo;t landed yet.
        </p>
        <h3>v3: the version-drift discovery</h3>
        <p>
            500 questions, 9 judges spanning 3 providers and 4 Claude
            generations. Two things emerged that weren&rsquo;t in the
            original plan:
        </p>
        <ul>
            <li><strong>Model version drift dominates provider
                differences.</strong> Claude Sonnet 4 hit r=0.40
                against gold F1; Claude Sonnet 4.6, same provider
                eight months later, hit r=0.58 &mdash; a 45%
                relative improvement on identical answers. A
                spot-check of Opus 4 vs Opus 4.7 showed the same
                pattern. Within-provider version differences turned
                out to <em>exceed</em> cross-provider differences,
                which has consequences for anyone using
                LLM-as-judge: you can&rsquo;t cite an old judge
                model&rsquo;s correlation number and assume the
                current version of the same family behaves the same.</li>
            <li><strong>GPT-5.4 Mini is the best value judge.</strong>
                r=0.55 against gold F1, $0.0015 per call, no
                version-drift penalty. The final v3 conclusion changed
                the Experiment 1 / 2 panel from &ldquo;Claude
                Sonnet&rdquo; to &ldquo;Claude Haiku 4.5 + GPT-5.4
                Mini&rdquo;: cross-provider, both current-generation,
                ~$32 total budget across both full experiments.</li>
        </ul>
        <p>
            The v1 conclusion on the website is annotated as
            superseded by v3 rather than rewritten, so the
            methodological evolution is visible to a reader.
        </p>

        <h2 id="exp12">What we got wrong: Experiments 1 and 2</h2>
        <p>
            By the time Experiments 1 and 2 ran, we&rsquo;d learned
            quite a bit from Experiment 0&rsquo;s revisions: we were
            now capturing what the LLM actually saw, scoring against
            that, using a validated cross-provider judge panel, and
            running on a real factorial design. The dashboards look
            polished. They are <em>not</em> as scientifically robust
            as they look.
        </p>
        <p>
            The big oversight wasn&rsquo;t about <em>what</em> we
            measured but <em>what we wrote down per row</em>. The CSV
            captured the final answer text, the final context sent to
            the LLM, total strategy latency, and gold-vs-RAG metrics
            &mdash; plenty for headline rankings. What it didn&rsquo;t
            capture:
        </p>
        <ul>
            <li>The number of LLM calls per row &mdash; only a
                process-wide running total in the heartbeat log.</li>
            <li>The branch path taken by Corrective (did reformulation
                fire?) or Adaptive (which sub-path did the classifier
                pick?).</li>
            <li>The intermediate LLM outputs (the
                &ldquo;relevant&rdquo;/&ldquo;irrelevant&rdquo;
                ratings inside Corrective, the reformulated query
                Multi-Query produced, the classifier verdict
                Adaptive returned). These drive branch decisions and
                become inputs to later prompts &mdash; missing them
                means missing the per-row substituted prompts for
                every call after the first.</li>
            <li>The code SHA at generation time, so an analyst could
                pin which version of <code>src/strategies/</code>
                produced a given row.</li>
        </ul>
        <p>
            The prompt <em>templates</em> are version-controlled in
            <code>src/strategies/*.py</code>, so the prompt
            <em>shape</em> for every call is recoverable from source.
            What isn&rsquo;t recoverable per row is the substituted
            content for any call whose input came from a prior LLM
            output. For NaiveRAG that&rsquo;s fully recoverable
            (one fixed template, one chat call); for the other four
            strategies it isn&rsquo;t.
        </p>
        <p>
            That gap matters most for CorrectiveRAG and AdaptiveRAG,
            which are precisely the strategies whose value
            <em>depends</em> on conditional behaviour. Without
            recording the conditional outcomes we can&rsquo;t do the
            fine-grained error analysis that distinguishes
            &ldquo;strategy didn&rsquo;t help&rdquo; from &ldquo;strategy
            helped on the rows where its second pass actually
            fired.&rdquo; That&rsquo;s the difference between a
            publishable experiment and a working demo, and we
            should own it openly.
        </p>
        <p>
            Phase 1 of the fix shipped: a per-row trace
            (<code>begin_row</code> / <code>end_row</code> /
            <code>record_complete</code>) wraps every
            <code>OllamaLLM.generate</code> call, every strategy now
            tags its calls with an <em>intent</em> label
            (<code>rate_relevance</code>, <code>reformulate_query</code>,
            <code>classify_complexity</code>, etc.), and the runner
            writes the full per-call trace to a sidecar
            <code>traces.jsonl</code> alongside <code>events.jsonl</code>.
            Seven new CSV columns (<code>n_llm_calls</code>,
            <code>llm_call_intents</code>, <code>final_prompt</code>,
            <code>prompts_source</code>, <code>code_sha</code>,
            <code>llm_num_predict</code>, <code>llm_keep_alive</code>)
            carry the summary. Historical rows have been
            schema-aligned via a backfill script; NaiveRAG rows have
            their <code>final_prompt</code> reconstructed
            byte-deterministically, multi-step strategy rows are
            tagged but not fabricated.
        </p>

        <h2 id="reproducibility">Stochastic generation: the reproducibility cost</h2>
        <p>
            The other reason Experiments 0&ndash;2 are less
            reproducible than they should be: the runner used
            Ollama&rsquo;s default generation parameters, which
            include <code>temperature&nbsp;&asymp;&nbsp;0.8</code> and
            no seed. The same row, if re-run today, would produce a
            different answer &mdash; and possibly a different LLM
            judgement of it, and possibly a different branch path in
            Corrective or Adaptive.
        </p>
        <p>
            That means row-level reproducibility is gone. Aggregate
            statistics (mean F1 over 200 questions, strategy &times;
            model rankings) remain meaningful <em>in expectation</em>,
            but their <em>variance</em> carries unnecessary
            stochastic-generation noise. The 0.07-point gap between
            Naive and Corrective in Experiment 1, and the 0.20-point
            chunker spread in Experiment 2, are both within the noise
            band that a <code>temperature&nbsp;=&nbsp;0</code> +
            fixed-seed regime would eliminate. We don&rsquo;t know how
            much of that &ldquo;within noise&rdquo; is real
            indistinguishability vs sampling variance until we re-run
            with deterministic generation.
        </p>
        <p>
            This is one line of options in
            <code>OllamaLLM.generate</code> and a few CSV columns to
            stamp it per row; the new per-row instrumentation can
            carry it without further changes. Phase&nbsp;2 work.
        </p>

        <h2 id="audit">The full audit: uncollected details, severity-ranked</h2>
        <p>
            The prose above sketched the gaps in narrative form. This is
            the comprehensive list, ordered by how much each gap limits
            interpretability of Experiments 0&ndash;2. None of these
            invalidate the headline findings; they constrain how finely
            the data can be sliced, audited, or reproduced. Items 1, 2,
            and 3 are the practical limit to how scientific the existing
            data can be made retroactively; for the next experiment, items
            1, 2, 4, 6, and 7 are the right batch to add together.
        </p>

        <div class="callout callout--critical">
            <div class="callout__label">Most severe &middot; affects every row in every experiment</div>
            <h3 class="callout__title">Stochastic generation without a fixed seed</h3>
            <p>
                Generation parameters were not set explicitly &mdash; the
                runner used Ollama&rsquo;s defaults, which include
                <code>temperature&nbsp;&asymp;&nbsp;0.8</code> and no
                <code>seed</code>. That means a given row, if re-run
                today, would produce a <em>different</em> answer (and
                possibly a different LLM judgement, and possibly a
                different branch path in CorrectiveRAG / AdaptiveRAG).
                Row-level reproducibility is therefore not what a
                research benchmark should provide. Aggregate statistics
                (mean F1 over 200 questions, strategy &times; model
                rankings) remain meaningful in expectation, but their
                <em>variance</em> is partly stochastic-generation noise
                that a <code>temperature&nbsp;=&nbsp;0</code> +
                fixed-seed regime would eliminate. The fix is one line
                of options in <code>OllamaLLM.generate</code>; a
                principled fix also stamps <code>temperature</code>,
                <code>top_p</code>, <code>top_k</code>, and
                <code>seed</code> per row, which the new
                instrumentation can carry without further changes.
            </p>
        </div>

        <table class="data-table audit-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>What was not recorded</th>
                    <th>Why it matters for Exp 0&ndash;2</th>
                    <th>Severity</th>
                    <th>Difficulty to add</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td><strong>Generation params per row</strong>
                        (temperature, top_p, top_k, seed)</td>
                    <td>Same row re-run today produces a different
                        answer. Eliminates row-level reproducibility;
                        leaves aggregate statistics intact in expectation.</td>
                    <td><span class="sev sev--critical">Critical</span></td>
                    <td>Low &mdash; one options dict + CSV column</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td><strong>Strategy hyperparameters per row</strong>
                        (e.g.&nbsp;CorrectiveRAG&rsquo;s &ldquo;reformulate
                        if &lt;&nbsp;2 chunks pass&rdquo; threshold,
                        MultiQueryRAG&rsquo;s 3 sub-queries)</td>
                    <td>If any threshold is tweaked between runs, two
                        cells of the same (strategy,&nbsp;model) become
                        silently incomparable. No versioning protects
                        against this today.</td>
                    <td><span class="sev sev--high">High</span></td>
                    <td>Medium &mdash; <code>.hyperparameters</code>
                        property per strategy + CSV stamp</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td><strong>Branch path / decisions taken</strong>
                        (Corrective: did reformulation fire? Adaptive:
                        which sub-path? Self-RAG: did the critique change
                        the answer?)</td>
                    <td>For branchy strategies, this is half the
                        information about what the strategy <em>did</em>
                        on a row. Phase 1 traces capture intent labels
                        but not boolean outcomes.</td>
                    <td><span class="sev sev--high">High</span></td>
                    <td>Medium &mdash; strategies populate
                        <code>diagnostics["branch"]</code></td>
                </tr>
                <tr>
                    <td>4</td>
                    <td><strong>Per-call token counts</strong>
                        (prompt_tokens, completion_tokens,
                        eval_duration)</td>
                    <td>Cost is currently estimated from string length
                        &times;&nbsp;~4. With Ollama&rsquo;s
                        <code>eval_count</code>/<code>prompt_eval_count</code>
                        in <code>/api/chat</code> we&rsquo;d measure
                        it. Also surfaces &ldquo;did this row hit the
                        <code>num_predict</code> cap?&rdquo;</td>
                    <td><span class="sev sev--medium">Medium</span></td>
                    <td>Low &mdash; already in the response object</td>
                </tr>
                <tr>
                    <td>5</td>
                    <td><strong>Per-chunk retrieval scores</strong></td>
                    <td>CSV shows <em>which</em> chunks were retrieved,
                        not <em>with what similarity score</em>. Gold
                        chunk at rank 1 score 0.89 is a very different
                        signal from rank 5 score 0.41 &mdash; currently
                        indistinguishable in raw_scores.csv.</td>
                    <td><span class="sev sev--medium">Medium</span></td>
                    <td>Low &mdash; already in the retriever&rsquo;s
                        return dict, just not surfaced</td>
                </tr>
                <tr>
                    <td>6</td>
                    <td><strong>Intermediate LLM outputs</strong>
                        (relevance ratings, reformulated query text,
                        classifier outputs)</td>
                    <td>The intermediate strings that drive branch
                        decisions and final-prompt content. Phase 1
                        captures these in <code>traces.jsonl</code>
                        going forward; not in the CSV.</td>
                    <td><span class="sev sev--medium">Medium</span></td>
                    <td>Trivial now (already in trace; needs a
                        flattener)</td>
                </tr>
                <tr>
                    <td>7</td>
                    <td><strong>Embedder calls per row</strong>
                        (separate from chat calls)</td>
                    <td>Lumped into <code>n_total_calls_inc_embed</code>
                        currently. Separating lets us measure retrieval
                        cost independently of generation cost.</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Low &mdash; wire
                        <code>record_complete("embed", &hellip;)</code>
                        into OllamaEmbedder</td>
                </tr>
                <tr>
                    <td>8</td>
                    <td><strong>Library versions</strong> at gen time
                        (ollama-python, numpy, langchain-text-splitters,
                        huggingface-hub)</td>
                    <td>A retrieval bug fixed in
                        <code>langchain-text-splitters&nbsp;0.4.5</code>
                        would silently change chunk boundaries between
                        runs. No versioning catches this today.</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Low &mdash; once-per-run
                        <code>pkg_resources</code> snapshot</td>
                </tr>
                <tr>
                    <td>9</td>
                    <td><strong>Ollama model digest</strong> (not just
                        the tag)</td>
                    <td>An <code>ollama pull qwen3.5:4b</code> can
                        replace a tag&rsquo;s underlying weights without
                        warning. We capture <code>quantization_level</code>
                        but not the SHA of the model artifact actually
                        loaded.</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Low &mdash; <code>/api/show</code> returns
                        <code>digest</code>; we already call this for
                        quantization</td>
                </tr>
                <tr>
                    <td>10</td>
                    <td><strong>Per-chunk document IDs &amp; offsets</strong></td>
                    <td>The CSV has <code>context_sent_to_llm</code>
                        (text) but no
                        <code>chunk_doc_id</code>&nbsp;/&nbsp;<code>chunk_offset</code>
                        per retrieved chunk. &ldquo;Did we retrieve from
                        the right source document?&rdquo; is text-matching
                        rather than lookup.</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Medium &mdash; the chunker/Document model has
                        to carry doc_id end-to-end</td>
                </tr>
                <tr>
                    <td>11</td>
                    <td><strong>Hardware state per row</strong> (GPU
                        temperature, power draw, clocks)</td>
                    <td>Already captured every 10 rows in
                        <code>events.jsonl</code>, but never joined onto
                        rows. Thermal-correlated quality drift would be
                        invisible without that join.</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Trivial &mdash; post-hoc join from
                        events.jsonl</td>
                </tr>
                <tr>
                    <td>12</td>
                    <td><strong>API judge network-vs-model timing</strong>
                        breakdown (Anthropic / OpenAI)</td>
                    <td>Scorer latency includes variable network time.
                        Can&rsquo;t tell &ldquo;judge was slow today
                        because of network&rdquo; from &ldquo;judge was
                        slow because the model is slow on long
                        contexts.&rdquo;</td>
                    <td><span class="sev sev--low">Low</span></td>
                    <td>Hard &mdash; providers don&rsquo;t expose it
                        cleanly</td>
                </tr>
            </tbody>
        </table>

        <h2 id="public-ready">How far is this from public-ready?</h2>
        <p>
            Honestly, a fair amount. RAGBench is a personal research
            project I&rsquo;d be comfortable showing to a hiring
            manager or to a faculty advisor; I&rsquo;d <em>not</em> be
            comfortable having a stranger cite the numbers as if they
            were a benchmark on the order of MTEB or HELM. The
            distance from here to &ldquo;cite-able benchmark&rdquo;
            is roughly:
        </p>
        <ul>
            <li><strong>Strict reproducibility.</strong> Fix
                temperature=0 + seed, pin every dependency version,
                stamp library + Ollama versions per row. Without this
                no number can be exactly replicated.</li>
            <li><strong>Confidence intervals on the headline numbers.</strong>
                The dashboards show point estimates of means. There
                are no error bars or significance tests. With 200
                questions per cell that&rsquo;s perfectly tractable
                bootstrap territory; just not done yet.</li>
            <li><strong>More than one dataset.</strong> Everything
                here is HotpotQA. The conclusions need to be
                replicated on at least one factoid dataset (SQuAD 2.0
                is wired up but unused) and one long-form / domain-
                specific corpus before they generalise.</li>
            <li><strong>More than one hardware platform.</strong> All
                the timing data is from one 5090 Laptop on one
                Windows install. Latency comparisons would shift on
                a server-class GPU, and the BSOD story is specifically
                a 5090 driver issue.</li>
            <li><strong>Cost measurement, not estimation.</strong>
                Currently cost is estimated from string length &times;
                ~4. Per-call <code>prompt_tokens</code>/<code>completion_tokens</code>
                from Ollama&rsquo;s <code>/api/chat</code> response
                turns that into measurement; on the API side we have
                the actual usage figures from the SDKs already.</li>
            <li><strong>Per-row provenance for everything in the
                <a href="methodology.html#logging-gaps">audit table</a>.</strong>
                Phase 1 fixed prompts + call counts; Phase 2 needs
                generation params, strategy hyperparameters, branch
                outcomes, token counts, per-chunk retrieval scores.</li>
            <li><strong>Documentation for outside users.</strong> The
                README is honest about what works on the author&rsquo;s
                machine. It&rsquo;s not a Quickstart that would land
                cleanly on a fresh install elsewhere &mdash; the
                Ollama setup, the model downloads, the API keys, the
                Windows-specific stability scripts, and the assumption
                of an RTX-class GPU would all need spelling out.</li>
            <li><strong>Statistical analysis beyond the dashboards.</strong>
                The visualisations summarise; they don&rsquo;t prove
                anything. A proper writeup would include an ANOVA on
                the strategy &times; model design, a regression
                isolating the main effects, and a discussion of
                where the negative results are bounded by sample
                size.</li>
            <li><strong>Test coverage for edge cases.</strong> 691
                tests pass today, but they cover happy paths plus
                the failures we&rsquo;ve already debugged. Network
                partition during retrieval, Ollama runner mid-call
                crash, malformed strategy output, judge API rate
                limit during a long run &mdash; some of these are
                handled in code, none of them are systematically
                tested.</li>
        </ul>
        <p>
            That&rsquo;s probably another month of focused work to
            close. The good news: most of it is mechanical, not
            conceptual. The hard part &mdash; designing the
            experiments, picking the judges, surviving the
            5090 &mdash; is largely behind us.
        </p>

        <h2 id="next">What&rsquo;s next</h2>
        <ul>
            <li>Set <code>temperature=0</code> and a per-row seed in
                <code>OllamaLLM.generate</code>, stamp both per row.
                Re-run a subset (say, 10 questions &times; full matrix)
                to estimate how much of the inter-strategy noise is
                stochastic vs systematic.</li>
            <li>Expose strategy hyperparameters
                (<code>min_relevant_chunks</code>,
                <code>n_alt_queries</code>) as Python properties
                instead of hardcoded constants, and stamp them per row.</li>
            <li>Pull Ollama&rsquo;s <code>eval_count</code> /
                <code>prompt_eval_count</code> into the per-call
                trace records; surface per-row token totals in the CSV.</li>
            <li>Run Experiment 1 again with the new instrumentation,
                so a clean apples-to-apples comparison exists between
                &ldquo;reconstructed-minimal&rdquo; and
                &ldquo;recorded&rdquo; regimes.</li>
            <li>Add bootstrap confidence intervals to the headline
                charts. Probably the highest-leverage analysis
                upgrade.</li>
            <li>Decide whether to publish at all, or to leave RAGBench
                as a personal infrastructure project and use the
                lessons (logging-first instrumentation, BSOD-resilient
                runners, transparent retrospective) on the next thing.</li>
        </ul>

        <p class="muted-note" style="margin-top: 28px;">
            The full <a href="#audit">audit table</a> of uncollected
            granular details is above on this page. The schema-level
            &ldquo;Logging Gaps&rdquo; reference (new CSV columns,
            sidecar paths, backfill tag) lives on
            <a href="methodology.html#logging-gaps">Methodology</a>.
            Source code on
            <a href="https://github.com/nono638/2026Project" target="_blank" rel="noopener">GitHub</a>.
        </p>

    </div>
    """
    return _build_page_template(
        "Writeup",
        nav_active="writeup",
        content_html=content,
    )


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
            Defaults to ``docs/`` (the GitHub Pages source).
        experiments: List of experiment numbers to generate.
            Defaults to ``[0, 1, 2]``.
    """
    if results_dir is None:
        results_dir = _PROJECT_ROOT / "results"
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = _PROJECT_ROOT / "docs"
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
        exp0_v3_csv = results_dir / "experiment_0_v3" / "raw_scores.csv"
        has_v1 = exp0_v1_csv.exists() and exp0_v1_csv.stat().st_size > 0
        has_v2 = exp0_v2_csv.exists() and exp0_v2_csv.stat().st_size > 0
        has_v3 = exp0_v3_csv.exists() and exp0_v3_csv.stat().st_size > 0

        if has_v1 or has_v2 or has_v3:
            desc_parts = []
            if has_v1:
                desc_parts.append("v1: 50 HotpotQA × NaiveRAG × Qwen3 4B")
            if has_v2:
                desc_parts.append("v2: 150 medium+hard × BGE reranker × diagnostics")
            if has_v3:
                desc_parts.append("v3: 500 HotpotQA × BGE reranker × 6 judges (3 Gemini + 3 Claude)")
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
                    # v2 no longer gets its own card on the home page — it's
                    # reachable via the v3 narrative section instead
                except Exception as exc:
                    logger.warning("Experiment 0 v2 dashboard generation failed: %s", exc)

            if has_v3:
                logger.info("Generating Experiment 0 v3 dashboard from %s", exp0_v3_csv)
                try:
                    # v3 uses its own generator with narrative + charts
                    exp0_v3_html = _generate_experiment_0_v3(exp0_v3_csv)
                    (output_dir / "experiment_0_v3.html").write_text(
                        exp0_v3_html, encoding="utf-8",
                    )
                    shutil.copy2(exp0_v3_csv, output_dir / "raw_scores_v3.csv")
                    # v3 is the hero on the home page — no separate card needed
                except Exception as exc:
                    logger.warning("Experiment 0 v3 dashboard generation failed: %s", exc)

            if not has_v1:
                # Only v2/v3 exists — make the latest the main page
                if has_v3:
                    shutil.copy2(
                        output_dir / "experiment_0_v3.html",
                        output_dir / "experiment_0.html",
                    )
                elif has_v2:
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

    # Always generate methodology page regardless of --experiments flag
    methodology_html = _generate_methodology()
    (output_dir / "methodology.html").write_text(methodology_html, encoding="utf-8")

    # Long-form project writeup — motivation, findings, postmortem.
    writeup_html = _generate_writeup()
    (output_dir / "writeup.html").write_text(writeup_html, encoding="utf-8")

    # Engineering Notes — deep-dive technical companion to the Writeup's
    # plain-English BSOD summary. Not in the nav; linked from Writeup.
    engineering_html = _generate_engineering()
    (output_dir / "engineering.html").write_text(engineering_html, encoding="utf-8")

    # Data / Downloads page + copy large CSVs so they're directly
    # downloadable from the live site. Exp 0 v1/v2/v3 CSVs are copied
    # in the experiment-0 block above; here we mirror Exp 1 and Exp 2
    # plus the events.jsonl heartbeat sidecar so the Data page links
    # resolve. Files are tens of MB but well under Pages's per-file
    # 100 MB and per-site 1 GB limits.
    import shutil as _shutil
    for exp_num in (1, 2):
        src_csv = results_dir / f"experiment_{exp_num}" / "raw_scores.csv"
        if src_csv.exists() and src_csv.stat().st_size > 0:
            try:
                _shutil.copy2(
                    src_csv,
                    output_dir / f"experiment_{exp_num}_raw_scores.csv",
                )
            except OSError as exc:
                logger.warning("Could not copy %s into docs/: %s", src_csv, exc)
    exp1_events = results_dir / "experiment_1" / "events.jsonl"
    if exp1_events.exists():
        try:
            _shutil.copy2(exp1_events, output_dir / "experiment_1_events.jsonl")
        except OSError as exc:
            logger.warning("Could not copy events.jsonl into docs/: %s", exc)

    data_html = _generate_data()
    (output_dir / "data.html").write_text(data_html, encoding="utf-8")

    # Generate index page
    index_html = _build_page_template(
        "RAGBench Findings Gallery",
        nav_active="home",
        content_html=_generate_index(experiments_info),
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    # Count actually-rendered HTML files (experiments + index +
    # methodology + writeup + engineering + data + any version pages).
    page_count = len(list(output_dir.glob("*.html")))
    logger.info("Gallery generated: %d HTML pages in %s", page_count, output_dir)
    print(f"Gallery generated in {output_dir}/ ({page_count} pages)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAGBench findings gallery")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: docs/, the GitHub Pages source)",
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
