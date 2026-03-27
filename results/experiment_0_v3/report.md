# Experiment 0: Scorer Validation Report

## Per-Judge Mean Scores

| Judge | Faithfulness | Relevance | Conciseness | Quality |
|-------|-------------|-----------|-------------|---------|
| google:gemini-2.5-flash-lite | 4.568 | 4.658 | 4.786 | 4.671 |
| google:gemini-2.5-flash | 4.684 | 4.844 | 4.700 | 4.743 |
| google:gemini-2.5-pro | 4.673 | 4.901 | 4.667 | 4.747 |
| anthropic:claude-haiku-4-5-20251001 | 4.592 | 4.668 | 4.450 | 4.570 |
| anthropic:claude-sonnet-4-20250514 | 4.652 | 4.766 | 4.582 | 4.667 |
| anthropic:claude-opus-4-20250514 | 4.722 | 4.804 | 4.664 | 4.730 |

## Inter-Scorer Correlation (Pearson, Quality)

|                                     |   google:gemini-2.5-flash-lite |   google:gemini-2.5-flash |   google:gemini-2.5-pro |   anthropic:claude-haiku-4-5-20251001 |   anthropic:claude-sonnet-4-20250514 |   anthropic:claude-opus-4-20250514 |
|:------------------------------------|-------------------------------:|--------------------------:|------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------------:|
| google:gemini-2.5-flash-lite        |                          1     |                     0.566 |                   0.578 |                                 0.557 |                                0.564 |                              0.516 |
| google:gemini-2.5-flash             |                          0.566 |                     1     |                   0.892 |                                 0.634 |                                0.716 |                              0.737 |
| google:gemini-2.5-pro               |                          0.578 |                     0.892 |                   1     |                                 0.674 |                                0.733 |                              0.731 |
| anthropic:claude-haiku-4-5-20251001 |                          0.557 |                     0.634 |                   0.674 |                                 1     |                                0.738 |                              0.751 |
| anthropic:claude-sonnet-4-20250514  |                          0.564 |                     0.716 |                   0.733 |                                 0.738 |                                1     |                              0.82  |
| anthropic:claude-opus-4-20250514    |                          0.516 |                     0.737 |                   0.731 |                                 0.751 |                                0.82  |                              1     |

## Correlation with Gold Metrics

| Judge | F1 (word overlap) |
|-------|----------|
| google:gemini-2.5-flash-lite | 0.139 |
| google:gemini-2.5-flash | 0.301 |
| google:gemini-2.5-pro | 0.348 |
| anthropic:claude-haiku-4-5-20251001 | 0.450 |
| anthropic:claude-sonnet-4-20250514 | 0.397 |
| anthropic:claude-opus-4-20250514 | 0.382 |

## Estimated Cost Breakdown

| Judge | Calls | Est. Cost/Call | Est. Total |
|-------|-------|----------------|------------|
| google:gemini-2.5-flash-lite | 500 | $0.0001 | $0.03 |
| google:gemini-2.5-flash | 500 | $0.0001 | $0.05 |
| google:gemini-2.5-pro | 500 | $0.0010 | $0.50 |
| anthropic:claude-haiku-4-5-20251001 | 500 | $0.0010 | $0.50 |
| anthropic:claude-sonnet-4-20250514 | 500 | $0.0050 | $2.50 |
| anthropic:claude-opus-4-20250514 | 500 | $0.0100 | $5.00 |

## Gold Correctness Summary

- Exact match rate: 76.2%
- Mean word-overlap F1: 0.546

## Failure Stage Breakdown

- **none**: 381 (76.2%)
- **retrieval**: 70 (14.0%)
- **generation**: 48 (9.6%)
- **chunker**: 1 (0.2%)

## Recommendation

*Review the correlation matrix and gold metric correlations above.*
BERTScore (semantic) is more reliable than word-overlap F1 for generated text.
Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.
