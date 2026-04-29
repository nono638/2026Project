# task-051: Figure captions for gallery dashboards (Exp 0 v1/v2/v3)

## Goal

Add a 1-3 sentence caption beneath each figure in the Exp 0 v1, v2, and v3 Plotly
dashboards explaining what the chart shows and what to take away from it. Professor
feedback (2026-04-29) said the figures look impressive but lack interpretation —
specifically called out the answer-length distribution as one that needs context.

## Why

Right now charts are presented bare. A reader who isn't deep in RAG eval doesn't know
why a histogram of answer lengths matters or what conclusion to draw. Captions turn the
gallery from a data dump into a narrative. This is a low-effort, high-impact change before
final demo (2026-05-11).

## Files to modify

Find the dashboard generators:
- `scripts/generate_experiment0_dashboard.py` (v1)
- `scripts/generate_experiment0_v2_dashboard.py` (or wherever v2 lives — find via grep)
- `scripts/generate_experiment0_v3_dashboard.py` (or v3 equivalent)

Use grep to locate them: `grep -rn "build_experiment0_figures\|fig.update_layout" scripts/`

## Pattern

After each `fig` is added to the dashboard layout, render a caption directly below it.
In Plotly HTML output, the simplest way is to insert an HTML `<p class="caption">...</p>`
between figures in the assembled HTML (not inside the Plotly figure object itself).

If the dashboard assembles HTML via string concatenation or a template (likely — task-030
built it as a single self-contained HTML), add a `CAPTIONS: dict[str, str]` mapping
figure-id → caption text near the top of each generator file, and inject `<p class="caption">{CAPTIONS[fig_id]}</p>`
after each figure's div.

Add CSS for `.caption` to the gallery's stylesheet (`docs/style.css` or wherever
generate_gallery.py writes styles — find via grep):
```css
.caption {
  font-size: 0.95em;
  color: #4a4a4a;
  font-style: italic;
  max-width: 800px;
  margin: 0.5em auto 2em auto;
  line-height: 1.4;
  padding: 0 1em;
}
```

## Caption content

Write captions yourself based on what each chart shows. Each caption should answer:
1. **What the chart shows** (one short sentence)
2. **The takeaway** (one sentence — what should the reader conclude?)
3. **Optional**: a caveat or interesting nuance (one sentence, only if non-obvious)

### Specific captions to write

The dashboards have ~12-18 figures each. For each figure, write a caption following the
pattern above. Below are required captions for the figures the professor flagged or that
are most central to the methodology — write good captions for these specifically. For
the rest, write reasonable captions following the same pattern.

**Required: answer-length distribution (the figure professor singled out):**
> "Distribution of answer lengths (in tokens) for the model's RAG answers versus the
> HotpotQA gold answers. The model produces substantially longer answers because it isn't
> prompted to be concise — it writes natural-language sentences, while HotpotQA gold
> answers are short extractive spans (often 1-3 words). This explains why conciseness
> scores diverge from gold F1: the model isn't wrong, it's verbose. A future experiment
> could test whether a 'be concise' instruction closes the gap without hurting faithfulness."

**Required: judge-vs-gold scatter (the primary scorer-validation chart):**
> "Each point is one question; the x-axis is the gold-standard quality signal (BERTScore
> against the HotpotQA reference answer), the y-axis is the LLM judge's quality score.
> A judge that tracks gold well will show a tight upward trend. The Pearson correlation
> shown in the title is our primary metric for picking a scorer."

**Required: inter-judge correlation heatmap:**
> "Pairwise Pearson correlation between every pair of LLM judges. High off-diagonal
> values (warm colors) mean two judges agree; low values mean they disagree. Judges
> from the same provider family (e.g., all Gemini models) tend to correlate more
> strongly with each other than across families — evidence that family-level bias is
> real and that using judges from multiple providers is methodologically sound."

**Required: BERTScore histogram:**
> "Distribution of BERTScore F1 between model answers and HotpotQA gold answers across
> the n questions in this run. A right-skewed distribution (most mass near 1.0) means
> the model is mostly producing semantically-equivalent answers; a wider spread means
> more variance in answer quality. This is our objective-truth signal and the anchor
> for judge validation."

**For all other figures**: write captions in the same style. Be specific, not generic
("This chart shows the data" is useless). State the takeaway. Avoid jargon the gallery
landing page hasn't introduced.

## Where the captions live

For v3 specifically, also add a brief paragraph at the top of the dashboard above the
first chart summarizing the v3 finding (which scorer won the tiebreaker). Read
`results/experiment_0_v3/report.md` for the actual finding — do not invent numbers.

## What NOT to touch

- Do not change any chart's data, axes, colors, or layout
- Do not refactor the dashboard generators beyond what's needed to inject captions
- Do not add captions to Exp 1 / Exp 2 dashboards — those experiments haven't been run
  yet and figures will change
- Do not change the gallery landing page or methodology page (task-044 covers that)

## Quality checklist

- [ ] Every figure in v1, v2, v3 dashboards has a caption below it
- [ ] Captions follow the what/takeaway pattern
- [ ] Required figures (answer-length, judge-vs-gold, inter-judge heatmap, BERTScore
      histogram) use the captions in this spec verbatim
- [ ] No numbers invented — anything specific (correlations, n, etc.) read from CSVs/reports
- [ ] CSS `.caption` style added and renders correctly in browser
- [ ] Gallery rebuilds without errors: `python scripts/generate_gallery.py`
- [ ] All existing tests still pass
- [ ] V3 dashboard has a top-of-page summary paragraph
