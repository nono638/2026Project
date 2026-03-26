# task-044: Gallery MVP Polish

## What

Upgrade the findings gallery (`scripts/generate_gallery.py` + `site/`) from a functional
prototype to an MVP-ready demo site for the April 11 presentation. Focus on three areas:
landing page, methodology page, and navigation/polish.

## Why

The current gallery has great Plotly dashboards for Experiment 0 but the landing page is
minimal — just experiment cards. For the MVP demo, visitors need to understand what RAGBench
is, how it works, and what we've found, without clicking into individual dashboards first.
This is the "Explorer" audience from the product vision.

## Files to Modify

- `scripts/generate_gallery.py` — all changes go here (the gallery generator)

## Files NOT to Touch

- `scripts/generate_experiment0_dashboard.py` — leave as-is
- `scripts/generate_experiment1_dashboard.py` — leave as-is
- `scripts/generate_experiment2_dashboard.py` — leave as-is
- `scripts/generate_visuals.py` — legacy, ignore
- Any files in `src/` — this is a presentation-layer task only

## Output

Running `python scripts/generate_gallery.py` should produce the following in `site/`:
- `index.html` — upgraded landing page (see below)
- `methodology.html` — new page explaining the pipeline and experiments
- `experiment_0.html` — unchanged (existing Plotly dashboard)
- `experiment_1.html` — unchanged (placeholder or dashboard)
- `experiment_2.html` — unchanged (placeholder or dashboard)

## Detailed Requirements

### 1. Landing Page Upgrade (`index.html`)

**Hero section** at top:
- Project title: "RAGBench"
- Tagline: "A configurable evaluation pipeline for Retrieval-Augmented Generation"
- One-paragraph description: RAGBench runs the full cartesian product of RAG configurations
  (chunker x embedder x strategy x language model), scores the results with LLM judges, and
  identifies optimal configurations for different constraints. It answers: when does a small
  model with a smart strategy outperform a larger model with naive RAG?
- A "View Experiments" button/link that scrolls to the experiment cards section

**Key Findings section** (below hero, above experiment cards):
- Show 3-4 headline findings from Experiment 0v2 as styled cards/callouts. Use these findings
  (hardcoded text is fine — these are established results from v2, 150 medium+hard questions):
  1. "Claude Haiku is the most accurate LLM judge (r=0.640 with gold BERTScore) — and the cheapest Anthropic option"
  2. "Gemini 2.5 Pro is the best free scorer (r=0.518 with BERTScore) — no API cost via Google AI Studio"
  3. "Inter-judge agreement is strong among top judges (Sonnet-Opus r=0.884, Flash-Pro r=0.841)"
  4. "74% exact match on medium+hard HotpotQA — retrieval and generation failures split evenly (13% each)"
- Each finding card: icon/emoji-free, just a bold headline and one sentence of context

**Experiment Cards section** (existing, but improved):
- Keep the current experiment card grid
- Add a "View Methodology" link/card alongside experiment cards

**Footer**:
- "Built by Noah | CUNY SPS Spring 2026 | Powered by RAGBench"
- Link to methodology page

### 2. Methodology Page (`methodology.html`)

A new page explaining how RAGBench works. This is for visitors who want to understand
the approach before diving into results.

**Sections:**

**Pipeline Overview** — Text description of the RAG evaluation pipeline with a simple
ASCII/text flow diagram rendered as a styled `<pre>` block:
```
Documents --> QueryGenerator --> Queries --> QueryFilter --> Validated Queries
                                                                |
Validated Queries x (Chunker x Embedder x Strategy x Model) --> Answers
                                                                |
                                                     Scorer --> Scores
                                                                |
                                                  ExperimentResult --> Analysis
```

**The Four Axes** — Brief explanation of what each axis tests:
- Chunker: how documents are split (fixed-size, recursive, sentence, semantic)
- Embedder: how chunks become vectors (mxbai-embed-large via Ollama)
- Strategy: how retrieved context is used (NaiveRAG, SelfRAG, CorrectiveRAG, AdaptiveRAG, MultiQueryRAG)
- Model: which LLM generates answers (Qwen3 0.6B/1.7B/4B/8B, Gemma 3 1B/4B)

**Evaluation Approach** — Explain dual evaluation:
- Intrinsic: LLM judge scores faithfulness, relevance, conciseness against retrieved context
- Extrinsic: compare against gold-standard answers (BERTScore, F1) when available
- Scorer validation: Experiment 0 exists specifically to validate the LLM-as-judge approach

**Experiment Design** — Table or cards describing each experiment:
| Experiment | Tests | Matrix | Held Constant |
|---|---|---|---|
| 0 (Scorer Validation) | Which LLM judge is most accurate? | 150 medium+hard HotpotQA x 7 judges | Qwen3-4B, NaiveRAG + BGE reranker |
| 1 (Strategy x Model) | Does strategy compensate for model size? | 5 strategies x 6 models | Recursive chunker, mxbai-embed-large |
| 2 (Chunking x Model) | Does chunking strategy interact with model capability? | 4 chunkers x 4 models | NaiveRAG, mxbai-embed-large |

**Dataset** — Brief note: primary corpus is HotpotQA (113K multi-hop Wikipedia Q&A pairs
with gold answers and difficulty labels).

### 3. Navigation and Polish

**Navigation bar** — update to include:
- RAGBench (home link, bold/logo style)
- Experiments dropdown or flat links: Exp 0, Exp 1, Exp 2
- Methodology (link to new page)
- Active page indicator (highlight current page in nav)

**CSS polish:**
- Hero section: large text, centered, subtle background gradient (#1a1a2e to #2d2d44)
  with white text
- Finding cards: white background, subtle shadow, left border accent using IBM palette colors
- Methodology page: clean readable prose, max-width ~800px for text content
- Ensure the nav "active" state is visually distinct (brighter text or underline)

**Responsive:**
- Experiment card grid should stack to single column on narrow screens (max-width: 768px)
- Hero text should scale down on mobile
- Nav should remain functional on mobile (can stay horizontal, just smaller)

### 4. Implementation Notes

- All changes are in `generate_gallery.py`. The gallery is a Python script that writes
  HTML files — there are no separate template files.
- Follow the existing pattern: HTML is built as Python string templates with f-strings
  and `.format()` calls.
- The shared CSS is in the `_GALLERY_CSS` constant — extend it, don't replace it.
- The shared nav is generated by a function — update it to include the new pages.
- The `_page_template()` function wraps content in the shared layout — use it for the
  methodology page too.
- Plotly CDN is already loaded per page — no changes needed for chart pages.
- The `main()` function orchestrates page generation — add methodology page generation there.

## Edge Cases

- If `results/experiment_0/` doesn't exist, the Key Findings section should still render
  (findings are hardcoded text, not computed from data).
- The methodology page is entirely static content — no data dependencies.
- Don't break the existing `--experiments` flag behavior. Methodology page should always
  generate regardless of which experiments are selected.

## What NOT to Do

- Don't modify the experiment dashboard generators (exp0/1/2 dashboard scripts)
- Don't add JavaScript beyond what Plotly needs — keep it a static site
- Don't add a build system, bundler, or static site generator dependency
- Don't fetch external resources beyond the existing Plotly CDN
- Don't add dark mode, theme switching, or other scope-expanding features
