# Task 030: Interactive Plotly Visualizations for Experiment 0

## What

Replace the static matplotlib visualizations with an interactive Plotly HTML dashboard for Experiment 0 results. Generate a single self-contained `visuals/experiment_0.html` file containing all charts. The goal is an abundance of visualizations — the user will prune later.

## Why

The static PNGs are underwhelming. The user needs:
1. A pipeline walkthrough that explains "what is RAG?" to the uninitiated
2. Judge vs gold comparison as the primary analysis (not inter-judge correlation)
3. Interactive filtering by question metadata (difficulty, type)
4. Charts that tell a story without requiring domain knowledge

Plotly produces self-contained HTML with hover, zoom, filter — far better for exploration.

## Files to Create

- `scripts/generate_experiment0_dashboard.py` — the main script
- `visuals/experiment_0.html` — output (generated, not committed)

## Files to Read (not modify)

- `results/experiment_0/raw_scores.csv` — all judge scores + gold metrics
- `results/experiment_0/raw_answers.csv` — questions, answers, doc_text

## Dependencies

- `plotly==6.6.0` — already installed in venv and in requirements.txt

## Data Notes

The CSV has 50 rows. Judge columns use prefix format like `google_gemini_2_5_flash_faithfulness`. Available judges with valid data:

| Judge | Valid scores (out of 50) |
|-------|------------------------|
| google:gemini-2.5-flash-lite | 20 |
| google:gemini-2.5-flash | 21 |
| google:gemini-2.5-pro | 0 (skip entirely) |
| google:gemini-3.1-pro-preview | 50 |
| anthropic:claude-haiku-4-5-20251001 | 50 |
| anthropic:claude-sonnet-4-20250514 | 50 |

**Skip gemini-2.5-pro in all visualizations** — it has zero valid scores.

Column prefixes map to display names:
```python
JUDGE_DISPLAY_NAMES = {
    "google_gemini_2_5_flash_lite": "Flash-Lite",
    "google_gemini_2_5_flash": "Flash",
    "google_gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "anthropic_claude_haiku_4_5_20251001": "Claude Haiku",
    "anthropic_claude_sonnet_4_20250514": "Claude Sonnet",
}
```

## Enrichment Step: Add HotpotQA Metadata to CSV

The raw_scores.csv is missing difficulty and question_type. Before generating visuals, enrich it:

1. Load HotpotQA with the same parameters used in Experiment 0: `load_hotpotqa(split="train")`, then `sample_hotpotqa(docs, queries, n=50, seed=42)`
2. Extract `query.metadata["difficulty"]` and `query.metadata["question_type"]` for each example
3. Add `difficulty` and `question_type` columns to raw_scores.csv (save enriched version)
4. These columns enable filtering in the interactive charts

HotpotQA difficulty levels: "easy", "medium", "hard"
HotpotQA question types: "comparison", "bridge"

## Visualizations to Generate

Generate ALL of the following. Output is one HTML file with sections and headings. Use `plotly.io.to_html(fig, full_html=False)` for each figure, then combine into a single HTML page with section headers and basic CSS styling.

### Section 1: Pipeline Walkthrough

**1. Interactive RAG Pipeline Walkthrough**
- Dropdown to select any of the 50 examples
- Shows in sequential panels/boxes:
  - **Question** — the original HotpotQA question
  - **Document** — the full source text (scrollable, truncated preview)
  - **RAG Answer** — what the model generated
  - **Gold Answer** — the reference answer
  - **Gold Metrics** — exact match (yes/no), F1, BERTScore
  - **Judge Scores** — grouped bar chart showing all judges' faithfulness/relevance/conciseness for this example
- This is NOT a Plotly chart — it's an HTML widget with a `<select>` dropdown and JavaScript to show/hide content. Use inline JS, no external deps.

### Section 2: Judge vs Gold (Primary Analysis)

**2. Judge Quality vs BERTScore Scatter**
- X axis: gold BERTScore, Y axis: judge quality score
- One trace per judge (color-coded), with dropdown to show/hide judges
- Hover shows: question text (truncated), gold answer, RAG answer, exact match
- Add regression line per judge (OLS trendline)
- Title: "Do judges agree with semantic similarity to the gold answer?"

**3. Judge Quality vs Gold F1 Scatter**
- Same as above but X axis is word-overlap F1
- Title: "Do judges agree with word-overlap correctness?"

**4. Judge-Gold Correlation Bar Chart**
- Grouped bar: each judge has two bars — BERTScore correlation and F1 correlation
- Sort by BERTScore correlation descending
- Annotation: Pearson r value on each bar
- Title: "Which judge best tracks ground truth?"

**5. Correct vs Incorrect: Judge Score Split**
- Split examples by exact_match (True/False)
- Box plots: each judge's quality scores, split into two boxes (correct / incorrect)
- A good judge should give higher scores to correct answers
- Title: "Can judges distinguish correct from incorrect answers?"

**6. Per-Example Score Heatmap**
- Rows: examples (labeled by truncated question text, 40 chars)
- Columns: judges + gold_bertscore + gold_f1
- Color: score value (1-5 for judges, 0-1 for gold metrics — normalize to same scale or use two colorbars)
- Sort rows by gold_bertscore descending
- Hover shows full question and all scores
- Title: "Score landscape across all judges and examples"

### Section 3: Score Distributions

**7. Score Distribution Violin Plots**
- One violin per judge for the quality metric
- Show individual points overlaid (jittered)
- Title: "How do judges distribute their scores?"

**8. Metric Breakdown per Judge**
- Grouped bar chart: judges on X, three bars each (faithfulness, relevance, conciseness)
- Shows mean ± std error bars
- Title: "Which dimensions drive quality differences?"

**9. Score vs Answer Length**
- X axis: RAG answer word count, Y axis: average quality across judges
- Color by exact_match
- Hover: question, answer
- Title: "Does answer length affect judge scores?"

**10. Score vs Question Length**
- X axis: question word count, Y axis: average quality across judges
- Color by difficulty (if enriched)
- Title: "Does question complexity affect scores?"

### Section 4: Judge Agreement (Secondary)

**11. Inter-Judge Correlation Heatmap**
- Correlation matrix (Pearson) on quality scores
- Annotated with r values
- Use a diverging colorscale (red-white-blue or similar)
- Title: "How much do judges agree with each other?"

**12. Biggest Disagreements Table**
- Find the 10 examples with highest variance in quality across judges
- Display as an interactive table (Plotly table or HTML table): question, each judge's score, gold metrics
- Title: "Where do judges disagree most?"

### Section 5: Gold Answer Analysis

**13. BERTScore Distribution**
- Histogram of gold_bertscore values across all 50 examples
- Add vertical line at mean
- Color bars by exact_match
- Title: "Semantic similarity to gold answers (BERTScore)"

**14. F1 Distribution**
- Histogram of gold_f1 values
- Add vertical line at mean
- Color bars by exact_match
- Title: "Word-overlap F1 with gold answers"

**15. BERTScore vs F1 Scatter**
- X: gold_f1, Y: gold_bertscore
- Color by exact_match
- Hover: question, gold_answer, rag_answer
- Title: "Do semantic and lexical metrics agree?"

### Section 6: Data Overview

**16. Question Length Distribution**
- Histogram of question word counts
- Color by question_type (if enriched)

**17. Answer Length Comparison**
- Side-by-side histograms: gold answer word count vs RAG answer word count
- Title: "RAG answers vs gold answers: length comparison"

**18. Summary Statistics Card**
- HTML card (not a chart) at the top of the page showing:
  - N examples: 50
  - Model: Qwen3 4B via NaiveRAG
  - Exact match rate: 74%
  - Mean BERTScore: 0.931
  - Mean F1: 0.611
  - Number of judges: 5 (with valid data counts)

## HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Experiment 0: Scorer Validation Dashboard</title>
    <style>
        /* Clean, readable CSS — dark text on light background */
        /* Section headers, card styling for summary stats */
        /* Scrollable containers for the pipeline walkthrough */
    </style>
</head>
<body>
    <h1>Experiment 0: Scorer Validation</h1>
    <!-- Summary card (#18) -->
    <!-- Section 1: Pipeline Walkthrough (#1) -->
    <!-- Section 2: Judge vs Gold (#2-6) -->
    <!-- Section 3: Score Distributions (#7-10) -->
    <!-- Section 4: Judge Agreement (#11-12) -->
    <!-- Section 5: Gold Analysis (#13-15) -->
    <!-- Section 6: Data Overview (#16-17) -->
    <script>
        /* Inline JS for pipeline walkthrough dropdown */
    </script>
</body>
</html>
```

Use inline `<style>` and `<script>` — no external CSS/JS files. Plotly's JS is embedded by `to_html(include_plotlyjs="cdn")` — use CDN to keep file size manageable.

## What NOT to Touch

- Don't modify `scripts/generate_visuals.py` (the matplotlib version) — keep it as a fallback
- Don't modify any `src/` files
- Don't modify `results/` CSVs except to add the difficulty/question_type enrichment columns
- Don't install any packages beyond plotly (already installed)

## Edge Cases

- Skip gemini-2.5-pro entirely (all NaN)
- For judges with partial data (Flash-Lite: 20/50, Flash: 21/50), use only valid rows. Note the sample size in hover/annotations.
- If HotpotQA enrichment fails (network issue downloading dataset), generate all charts without difficulty/type filtering and log a warning.

## Tests

Write tests in `tests/test_experiment0_dashboard.py`:

1. **test_dashboard_generates_html** — run the script with test data (tiny CSV with 3 rows, 2 mock judges), verify HTML file is created and contains expected section headers.
2. **test_enrichment_adds_columns** — mock HotpotQA loading, verify difficulty and question_type columns are added to DataFrame.
3. **test_skips_empty_judges** — verify judges with all-NaN scores are excluded from charts.
4. **test_judge_display_names** — verify column prefix → display name mapping works for all judges.
5. **test_pipeline_walkthrough_has_all_examples** — verify the generated HTML contains a select option for each example_id.

Mock all data — no real API calls or HotpotQA downloads in tests.
