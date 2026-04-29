# Plan: task-051 figure captions

## Files to modify
1. `scripts/generate_experiment0_dashboard.py` — add CAPTIONS dict, inject `<p class="caption">` after each chart, add `.caption` CSS.
2. `scripts/generate_gallery.py` — add CAPTIONS dict, inject after v2/v3 charts (3 v2-specific + 3 v3-specific + standard scorer charts), add `.caption` to `_GALLERY_CSS`.

## Approach
- Create a single shared CAPTIONS mapping keyed by chart title (e.g., "Inter-Judge Correlation", "Answer Length Comparison", "BERTScore Distribution"). Use the spec's verbatim captions for the 4 required ones, write reasonable captions for the rest.
- Helper `_caption_html(key)` returns the `<p class="caption">…</p>` string or empty if no caption registered.
- v1 dashboard: insert `_caption_html(title)` after each `chart-container` div in `generate_dashboard()`.
- v2/v3: same pattern in `_generate_experiment_0_v2` and `_generate_experiment_0_v3`. The standard scorer charts loop already has `title` in scope; v2/v3-specific charts have hardcoded `<h3>` titles to key on.
- Add CSS `.caption {…}` per spec to both `_CSS` and `_GALLERY_CSS`.
- v3: Add 1-paragraph summary above the first v3 chart with the actual finding (Claude Haiku r=0.450, n=500). Existing "Road to v3" already mentions this — add a short distinct summary note above charts.

## Captions
- Verbatim from spec: answer-length, judge-vs-gold scatter, inter-judge heatmap, BERTScore histogram.
- Reasonable for the rest (judge_vs_f1, judge_gold_correlation, score_heatmap, violin_distributions, metric_breakdown, score_vs_answer_length, score_vs_question_length, biggest_disagreements, F1 distribution, BERTScore_vs_F1, question_length_distribution, correct_vs_incorrect, answer_quality, failure_stage, judge_means).

## Tests
- Run `tests/test_gallery.py` and the existing dashboard tests; manual verify HTML by grep that captions appear (no live browser).
