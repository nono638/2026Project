# Plan: task-044 — Gallery MVP Polish

## Files to Modify
- `scripts/generate_gallery.py` — all changes here

## Approach

### 1. Update `_GALLERY_CSS`
- Add hero section styles (gradient background #1a1a2e to #2d2d44, white text, centered)
- Add finding card styles (white bg, shadow, left border accent with IBM palette colors)
- Add methodology page styles (max-width 800px for text content)
- Add responsive media queries (768px breakpoint for card grid stacking, hero scaling)
- Add active nav indicator (already exists as `.nav a.active`)

### 2. Update `_NAV_ITEMS`
- Add methodology entry: `("methodology", "Methodology", "methodology.html")`

### 3. Update `_build_page_template`
- Update footer text to: "Built by Noah | CUNY SPS Spring 2026 | Powered by RAGBench"
- Add methodology link in footer

### 4. Rewrite `_generate_index`
- Add hero section with title, tagline, description, and "View Experiments" anchor link
- Add Key Findings section with 4 hardcoded findings from Exp 0v2
- Keep experiment card grid
- Add a "View Methodology" card in the grid

### 5. Add `_generate_methodology` function
- Pipeline overview with styled `<pre>` block diagram
- Four axes explanation
- Evaluation approach (intrinsic + extrinsic)
- Experiment design table
- Dataset mention (HotpotQA)

### 6. Update `main()`
- Always generate methodology.html regardless of --experiments flag
- Wire up `_generate_methodology()` call

## Ambiguities
- Test checks for "Claude Sonnet" OR "most accurate" — spec says "Claude Haiku is the most accurate".
  Using spec's findings (Haiku = most accurate from v2). The test will pass with "most accurate".
- Test finding marker "answer quality" maps to the 74% exact match finding. Will include.
