# Experiment 0: Scorer Validation Report

## Per-Judge Mean Scores

| Judge | Faithfulness | Relevance | Conciseness | Quality |
|-------|-------------|-----------|-------------|---------|
| google:gemini-2.5-flash-lite | 4.540 | 4.620 | 4.840 | 4.667 |
| google:gemini-2.5-flash | 4.580 | 4.720 | 4.580 | 4.627 |
| google:gemini-2.5-pro | nan | nan | nan | nan |
| google:gemini-3.1-pro-preview | 4.620 | 4.760 | 4.680 | 4.687 |
| anthropic:claude-haiku-4-5-20251001 | 4.520 | 4.640 | 4.560 | 4.573 |
| anthropic:claude-sonnet-4-20250514 | 4.660 | 4.720 | 4.580 | 4.653 |
| anthropic:claude-opus-4-20250514 | 4.820 | 4.880 | 4.700 | 4.800 |

## Inter-Scorer Correlation (Pearson, Quality)

|                                     |   google:gemini-2.5-flash-lite |   google:gemini-2.5-flash |   google:gemini-2.5-pro |   google:gemini-3.1-pro-preview |   anthropic:claude-haiku-4-5-20251001 |   anthropic:claude-sonnet-4-20250514 |   anthropic:claude-opus-4-20250514 |
|:------------------------------------|-------------------------------:|--------------------------:|------------------------:|--------------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------------:|
| google:gemini-2.5-flash-lite        |                          1     |                     0.301 |                     nan |                           0.32  |                                 0.553 |                                0.175 |                              0.413 |
| google:gemini-2.5-flash             |                          0.301 |                     1     |                     nan |                           0.961 |                                 0.558 |                                0.632 |                              0.669 |
| google:gemini-2.5-pro               |                        nan     |                   nan     |                     nan |                         nan     |                               nan     |                              nan     |                            nan     |
| google:gemini-3.1-pro-preview       |                          0.32  |                     0.961 |                     nan |                           1     |                                 0.542 |                                0.734 |                              0.644 |
| anthropic:claude-haiku-4-5-20251001 |                          0.553 |                     0.558 |                     nan |                           0.542 |                                 1     |                                0.525 |                              0.401 |
| anthropic:claude-sonnet-4-20250514  |                          0.175 |                     0.632 |                     nan |                           0.734 |                                 0.525 |                                1     |                              0.611 |
| anthropic:claude-opus-4-20250514    |                          0.413 |                     0.669 |                     nan |                           0.644 |                                 0.401 |                                0.611 |                              1     |

## Correlation with Gold Metrics

| Judge | BERTScore | F1 (word overlap) |
|-------|----------|----------|
| google:gemini-2.5-flash-lite | 0.072 | 0.018 |
| google:gemini-2.5-flash | 0.603 | 0.494 |
| google:gemini-2.5-pro | N/A | N/A |
| google:gemini-3.1-pro-preview | 0.634 | 0.523 |
| anthropic:claude-haiku-4-5-20251001 | 0.368 | 0.301 |
| anthropic:claude-sonnet-4-20250514 | 0.682 | 0.599 |
| anthropic:claude-opus-4-20250514 | 0.481 | 0.388 |

## Estimated Cost Breakdown

| Judge | Calls | Est. Cost/Call | Est. Total |
|-------|-------|----------------|------------|
| google:gemini-2.5-flash-lite | 50 | $0.0001 | $0.00 |
| google:gemini-2.5-flash | 50 | $0.0001 | $0.01 |
| google:gemini-2.5-pro | 50 | $0.0010 | $0.05 |
| google:gemini-3.1-pro-preview | 50 | $0.0100 | $0.50 |
| anthropic:claude-haiku-4-5-20251001 | 50 | $0.0010 | $0.05 |
| anthropic:claude-sonnet-4-20250514 | 50 | $0.0050 | $0.25 |
| anthropic:claude-opus-4-20250514 | 50 | $0.0100 | $0.50 |

## Gold Correctness Summary

- Exact match rate: 74.0%
- Mean word-overlap F1: 0.611
- Mean BERTScore F1: 0.931

## Recommendation

*Review the correlation matrix and gold metric correlations above.*
BERTScore (semantic) is more reliable than word-overlap F1 for generated text.
Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.
