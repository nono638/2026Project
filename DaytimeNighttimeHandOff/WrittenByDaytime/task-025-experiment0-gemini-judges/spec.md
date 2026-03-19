# Task 025: Expand Experiment 0 to All-Gemini Judges

## What

Update `scripts/run_experiment_0.py` so that Experiment 0 works well without an
Anthropic API key. The user only has a Google AI Studio key, so we need enough
Gemini judges to produce a meaningful inter-scorer correlation matrix.

## Why

The user doesn't want to pay for Anthropic API calls right now. Google AI Studio
is free/cheap. The experimental design (do cheap judges agree with expensive ones?)
works just as well across Gemini tiers: Flash-Lite vs Flash vs Pro.

## Changes

### `scripts/run_experiment_0.py`

1. **Update `JUDGE_CONFIGS`** to include 4 Gemini judges + 2 Anthropic judges:

```python
JUDGE_CONFIGS = [
    # Gemini judges (free via Google AI Studio)
    {"provider": "google", "model": "gemini-2.5-flash-lite"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "google", "model": "gemini-2.5-pro"},
    # Anthropic judges (optional — skipped if ANTHROPIC_API_KEY not set)
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
]
```

Drop Claude Opus (too expensive for a scoring experiment). Add `gemini-2.5-flash-lite`
as the cheapest baseline.

2. **Update the docstring** at the top of the file:
   - Change "5 LLM judges" to "up to 6 LLM judges (4 Gemini + 2 Claude)"
   - Note that Anthropic judges are optional and skipped if API key is missing
   - Update the usage examples

3. **Update `cost_estimates` dict** in `generate_report()`:
   - Add entry for `"google:gemini-2.5-flash-lite": 0.00005`
   - Remove `"anthropic:claude-opus-4-6"` entry

4. **Update the Recommendation section** of the report to reference Gemini tiers:
   - Change "cheap scorers (Gemini Flash)" to "cheap scorers (Gemini Flash-Lite, Flash)"
   - Change "expensive ones (Claude Opus)" to "expensive ones (Gemini Pro, Claude)"

5. **Add a startup log** after initializing scorers that reports how many were
   successfully initialized and which were skipped, so the user sees clearly
   what's running:
   ```
   Initialized 4/6 judges: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro, ...
   Skipped 2 judges (missing API keys): claude-haiku-4-5-20251001, claude-sonnet-4-20250514
   ```

### `tests/test_experiment_0.py` (create new)

Write tests with all API calls mocked:

1. `test_compute_f1_exact` — "the capital is Paris" vs "Paris" → high F1
2. `test_compute_f1_no_overlap` — completely different strings → 0.0
3. `test_exact_match_contains` — gold substring in prediction → True
4. `test_exact_match_case_insensitive` — different cases → True
5. `test_safe_scorer_name` — "google:gemini-2.5-flash" → "google_gemini_2_5_flash"
6. `test_score_all_answers_skips_missing_keys` — mock LLMScorer to raise on
   anthropic init, verify only google scorers produce columns

## What NOT to touch

- Do not modify `src/scorers/llm.py` — the scorer already handles provider selection
- Do not change the generation pipeline (NaiveRAG, chunker, embedder)
- Do not add new dependencies

## Edge cases

- If ALL scorers fail to initialize, the script already exits with an error (line 244-245).
  This is correct behavior — don't change it.
- Google AI Studio has rate limits on free tier (5-15 RPM). The script already runs
  sequentially, so this shouldn't be an issue for 50 examples. Do NOT add rate limiting
  logic — if we hit limits, we'll handle it at runtime.
