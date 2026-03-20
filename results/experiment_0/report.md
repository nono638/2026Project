# Experiment 0: Scorer Validation Report

## Per-Judge Mean Scores

| Judge | Faithfulness | Relevance | Conciseness | Quality |
|-------|-------------|-----------|-------------|---------|
| google:gemini-2.5-flash-lite | 4.750 | 4.800 | 5.000 | 4.850 |
| google:gemini-2.5-flash | 4.619 | 4.762 | 4.667 | 4.683 |
| google:gemini-2.5-pro | nan | nan | nan | nan |
| google:gemini-3.1-pro-preview | 4.620 | 4.760 | 4.680 | 4.687 |
| anthropic:claude-haiku-4-5-20251001 | 4.520 | 4.640 | 4.560 | 4.573 |
| anthropic:claude-sonnet-4-20250514 | 4.660 | 4.720 | 4.580 | 4.653 |

## Inter-Scorer Correlation (Pearson, Quality)

|                                     |   google:gemini-2.5-flash-lite |   google:gemini-2.5-flash |   google:gemini-2.5-pro |   google:gemini-3.1-pro-preview |   anthropic:claude-haiku-4-5-20251001 |   anthropic:claude-sonnet-4-20250514 |
|:------------------------------------|-------------------------------:|--------------------------:|------------------------:|--------------------------------:|--------------------------------------:|-------------------------------------:|
| google:gemini-2.5-flash-lite        |                          1     |                    -0.149 |                     nan |                          -0.015 |                                 0.579 |                                0.203 |
| google:gemini-2.5-flash             |                         -0.149 |                     1     |                     nan |                           0.96  |                                 0.283 |                                0.645 |
| google:gemini-2.5-pro               |                        nan     |                   nan     |                     nan |                         nan     |                               nan     |                              nan     |
| google:gemini-3.1-pro-preview       |                         -0.015 |                     0.96  |                     nan |                           1     |                                 0.542 |                                0.734 |
| anthropic:claude-haiku-4-5-20251001 |                          0.579 |                     0.283 |                     nan |                           0.542 |                                 1     |                                0.525 |
| anthropic:claude-sonnet-4-20250514  |                          0.203 |                     0.645 |                     nan |                           0.734 |                                 0.525 |                                1     |

## Correlation with Gold Metrics

| Judge | BERTScore | F1 (word overlap) |
|-------|----------|----------|
| google:gemini-2.5-flash-lite | 0.064 | 0.023 |
| google:gemini-2.5-flash | 0.652 | 0.530 |
| google:gemini-2.5-pro | N/A | N/A |
| google:gemini-3.1-pro-preview | 0.634 | 0.523 |
| anthropic:claude-haiku-4-5-20251001 | 0.368 | 0.301 |
| anthropic:claude-sonnet-4-20250514 | 0.682 | 0.599 |

## Estimated Cost Breakdown

| Judge | Calls | Est. Cost/Call | Est. Total |
|-------|-------|----------------|------------|
| google:gemini-2.5-flash-lite | 50 | $0.0001 | $0.00 |
| google:gemini-2.5-flash | 50 | $0.0001 | $0.01 |
| google:gemini-2.5-pro | 50 | $0.0010 | $0.05 |
| google:gemini-3.1-pro-preview | 50 | $0.0100 | $0.50 |
| anthropic:claude-haiku-4-5-20251001 | 50 | $0.0010 | $0.05 |
| anthropic:claude-sonnet-4-20250514 | 50 | $0.0050 | $0.25 |

## Gold Correctness Summary

- Exact match rate: 74.0%
- Mean word-overlap F1: 0.611
- Mean BERTScore F1: 0.931

## Recommendation

*Review the correlation matrix and gold metric correlations above.*
BERTScore (semantic) is more reliable than word-overlap F1 for generated text.
Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.
