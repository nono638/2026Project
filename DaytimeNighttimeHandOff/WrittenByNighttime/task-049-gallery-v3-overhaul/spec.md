# task-049: Gallery website overhaul — v3 front and center, v1/v2 as history

## Summary

The gallery website currently treats v1, v2, and v3 as equal peers in a card grid. It
should be restructured so Experiment 0 v3 (n=500) is the centerpiece — the definitive
answer to "which LLM judge tracks truth?" — with v1 and v2 presented as the journey
that got us there. The home page should lead directly to v3 with a "third time's a charm"
framing. The v3 page needs a narrative "Journey to v3" section explaining what v1 and v2
got wrong and why v3 was necessary. v1 and v2 pages stay as-is but are linked from v3's
narrative. The broken v2/v3 card links must also be fixed.

## Requirements

1. **Home page redesign**: Replace the current card grid with a single hero focused on
   Experiment 0 v3. The hero should convey: "Which LLM judge tracks truth? (Third time's
   a charm.)" with a tagline about 500 questions, 6 judges, and the answer (Haiku). Include
   a prominent "View Results" link to `experiment_0_v3.html`. Exp 1 and Exp 2 cards should
   still appear below but as secondary items (they remain placeholders). The key findings
   grid should reflect v3 stats (already done — preserve current values).

2. **v3 page narrative**: Add a "Journey to v3" section at the top of the v3 page (before
   the charts). Three short paragraphs:
   - **v1** (n=50): First attempt. Didn't track what the LLM actually saw — judges scored
     against the full source document, not the retrieved chunks. No reranker. Sonnet came
     out on top. Link: `experiment_0.html`.
   - **v2** (n=150): Fixed the tracking, added BGE reranker, filtered to medium+hard
     questions. Haiku beat Sonnet — contradicting v1. But n=150 left room for doubt.
     Link: `experiment_0_v2.html`.
   - **v3** (n=500): Tiebreaker at scale. Haiku confirmed as best (r=0.450 gold F1),
     Sonnet second (0.397), Opus third (0.382). Flash≈Pro (r=0.892). Settled.
   After the narrative, include a key findings summary card (like the v2 page has), then
   all the charts from `_generate_experiment_0_v2()`.

3. **v3 gets its own generator function**: Create `_generate_experiment_0_v3(csv_path)`
   instead of reusing `_generate_experiment_0_v2()` directly. This new function should:
   - Include the Journey to v3 narrative section
   - Include a key findings summary card with v3-specific stats:
     - Best judge: Haiku (r=0.450 gold F1)
     - Best free judge: Gemini 2.5 Pro (r=0.348)
     - Pipeline accuracy: 76.2% exact match, mean F1 0.546
     - Failure stages: 76% none, 14% retrieval, 10% generation
   - Include a download link to `raw_scores_v3.csv`
   - Reuse chart-building logic from `_generate_experiment_0_v2()` (call into its chart
     code, or extract shared chart logic into a helper)
   - Use `nav_active="exp0v3"` (see requirement 5)

4. **Fix broken card links**: The current code uses `num: "0v2"` and `num: "0v3"` which
   generates `experiment_0v2.html` (no underscore) but the actual files are
   `experiment_0_v2.html` (with underscore). Fix the link generation so card links point
   to the correct filenames. The simplest fix: change the `href` in the card template to
   not derive from `num`, or change `num` values to match filenames (e.g., `"0_v2"`).

5. **Navigation update**: Update `_NAV_ITEMS` to make v3 the primary Experiment 0 link.
   Replace:
   ```python
   ("exp0", "Exp 0: Scorer Validation", "experiment_0.html"),
   ```
   With:
   ```python
   ("exp0v3", "Exp 0: Scorer Validation", "experiment_0_v3.html"),
   ```
   The v1 page (`experiment_0.html`) and v2 page (`experiment_0_v2.html`) should still
   exist and be reachable via links in the v3 narrative, but they don't need their own
   nav entries. Update all `nav_active="exp0"` references in v1/v2 generators to still
   highlight the Exp 0 nav item (use `nav_active="exp0v3"` so the nav item highlights
   when viewing any Experiment 0 page).

6. **Backward compatible**: v1 page (`experiment_0.html`) and v2 page
   (`experiment_0_v2.html`) must still generate and be accessible. Their content stays
   unchanged. Only the nav highlighting and the links from v3 change.

## Files to Modify

- `scripts/generate_gallery.py`:
  - `_NAV_ITEMS` (line 284): Change exp0 entry to point to `experiment_0_v3.html`
  - `_generate_index()` (line 346): Rewrite hero section to center on v3. Keep experiment
    cards for Exp 1, 2, and Methodology below. Remove the separate v1/v2/v3 cards —
    Experiment 0 should be represented by the hero, not cards.
  - `_generate_experiment_0_v2()` (line 750): Update `nav_active` from `"exp0"` to
    `"exp0v3"`. No other changes — this function stays as the v2 page generator.
  - `_generate_experiment_0()` (line 984): Update `nav_active` from `"exp0"` to
    `"exp0v3"`. No other changes.
  - New function `_generate_experiment_0_v3(csv_path)`: ~150 lines. Journey to v3
    narrative + key findings card + reuse chart logic from v2 generator. Place it after
    `_generate_experiment_0_v2()` (around line 982).
  - `main()` (line 1968+): Update the v3 generation block to call
    `_generate_experiment_0_v3()` instead of `_generate_experiment_0_v2()`. Fix the
    `experiments_info` entries so card links work (or remove v2/v3 from `experiments_info`
    since they'll be reached through the hero, not cards).

## New Dependencies

None — all required packages are already installed.

## Edge Cases

- **Only v3 data exists (no v1, no v2)**: v3 page generates fine. Journey narrative still
  references v1/v2 with links, but those links go to pages that don't exist. This is
  acceptable — the night instance should NOT add conditional link logic. v1 and v2 data
  will always be present in this repo.
- **v3 data has partial judges**: The partial judge detection from `_generate_experiment_0_v2`
  handles this already (< 50% non-null → excluded from correlations). Reuse that logic.
- **Gallery generated with `--experiments 0` flag**: Should still work and generate all
  three pages (v1, v2, v3) plus the index.
- **No Experiment 0 data at all**: Existing placeholder logic handles this. No change.

## Decisions Made

- **v3 gets its own function, not a parameterized v2 function**: The v3 page has unique
  narrative content (Journey to v3) that doesn't apply to v2. A separate function is
  cleaner than adding v2/v3 branching logic to one function. **Why:** Keeps each page
  generator self-contained and readable.
- **Don't remove v1/v2 experiment cards entirely**: They should not appear as top-level
  cards on the home page (v3 hero handles Experiment 0), but the pages themselves stay.
  **Why:** v1/v2 pages are already built and have value as historical context.
- **Nav points to v3, not v1**: The primary Experiment 0 nav link goes to the v3 page.
  **Why:** v3 is the definitive run. v1 is 50 questions with known methodology flaws.
- **Hero framing: "Third time's a charm"**: Light, honest framing that acknowledges the
  iteration. **Why:** User requested this specific tone.
- **Key findings in hero use v3 stats**: Already done in previous commit. Preserve those
  values (Haiku r=0.450, 76% exact match, etc.).
- **v1/v2 generators keep `nav_active="exp0v3"`**: When viewing v1 or v2, the nav still
  highlights "Exp 0" so the user knows they're in the Experiment 0 family. **Why:**
  Consistent navigation experience.

## What NOT to Touch

- `scripts/generate_experiment0_dashboard.py` — chart builder for v1, works fine
- `scripts/generate_experiment1_dashboard.py` — Exp 1 charts, unrelated
- `scripts/generate_experiment2_dashboard.py` — Exp 2 charts, unrelated
- `_generate_experiment_0()` content — v1 page content stays as-is (only nav_active changes)
- `_generate_experiment_0_v2()` content — v2 page content stays as-is (only nav_active changes)
- `_GALLERY_CSS` — no CSS changes needed
- `_build_page_template()` — template wrapper stays the same
- `_generate_methodology()` — methodology page unchanged
- Any `src/` or experiment runner code

## Exact Prose for v3 Page

The night instance should use this exact text for the Journey to v3 section (inside a
`<div class="card">` block):

```html
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
    <strong>v3</strong> (n=500) is the tiebreaker at scale. With 500 HotpotQA questions
    and 6 LLM judges (3 Gemini, 3 Claude), the results are clear: <strong>Claude Haiku
    is the most accurate judge</strong> (r=0.450 with gold F1), followed by Sonnet
    (0.397) and Opus (0.382). Among the free Gemini judges, Flash and Pro are nearly
    interchangeable (r=0.892). The question is settled.
</p>
```

## Exact Prose for Home Page Hero

```html
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

<div class="card" style="border-left: 4px solid #648FFF; margin-bottom: 32px;">
    <h2>Experiment 0: Which LLM Judge Tracks Truth?</h2>
    <p style="color: #888; font-style: italic; margin-bottom: 12px;">Third time's a charm.</p>
    <p>
        500 HotpotQA questions. 6 LLM judges. Three iterations to get it right.
        The answer: <strong>Claude Haiku at $0.002/call</strong> — the cheapest
        Anthropic model is also the most accurate judge.
    </p>
    <a href="experiment_0_v3.html" class="cta-btn" style="margin-top: 16px; display: inline-block;">
        View Experiment 0 Results →
    </a>
</div>
```

Then the key findings grid (already correct from previous commit), then the remaining
experiment cards (Exp 1, Exp 2, Methodology — NOT Exp 0 cards).

## Testing Approach

Tests verify the generated HTML output contains the expected structure:
1. v3 page contains "The Road to v3" narrative
2. v3 page contains links to v1 and v2 pages
3. v3 page contains key findings card with correct stats
4. Home page contains "Third time's a charm" text
5. Home page links to `experiment_0_v3.html`
6. Home page does NOT contain separate v1/v2/v3 experiment cards
7. Nav bar points to `experiment_0_v3.html` for Exp 0
8. v1 and v2 pages still generate without errors

Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-049-gallery-v3-overhaul/tests/ -v`
