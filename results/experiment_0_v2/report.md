# Experiment 0: Scorer Validation Report

**Note:** gemini-3.1-pro-preview scored only 11/150 examples due to hitting Google's
daily quota (250 requests/day for this model). Its numbers below are unreliable.
Re-run with `--judges gemini-3.1-pro-preview` after quota resets to get full data.

## Per-Judge Mean Scores

| Judge | Faithfulness | Relevance | Conciseness | Quality |
|-------|-------------|-----------|-------------|---------|
| google:gemini-2.5-flash-lite | 4.520 | 4.647 | 4.740 | 4.636 |
| google:gemini-2.5-flash | 4.560 | 4.787 | 4.600 | 4.649 |
| google:gemini-2.5-pro | 4.620 | 4.867 | 4.580 | 4.689 |
| google:gemini-3.1-pro-preview | 4.636 | 5.000 | 4.636 | 4.758 |
| anthropic:claude-haiku-4-5-20251001 | 4.580 | 4.707 | 4.427 | 4.571 |
| anthropic:claude-sonnet-4-20250514 | 4.573 | 4.700 | 4.453 | 4.576 |
| anthropic:claude-opus-4-20250514 | 4.667 | 4.807 | 4.560 | 4.678 |

## Inter-Scorer Correlation (Pearson, Quality)

|                                     |   google:gemini-2.5-flash-lite |   google:gemini-2.5-flash |   google:gemini-2.5-pro |   google:gemini-3.1-pro-preview |   anthropic:claude-haiku-4-5-20251001 |   anthropic:claude-sonnet-4-20250514 |   anthropic:claude-opus-4-20250514 |
|:------------------------------------|-------------------------------:|--------------------------:|------------------------:|--------------------------------:|--------------------------------------:|-------------------------------------:|-----------------------------------:|
| google:gemini-2.5-flash-lite        |                          1     |                     0.647 |                   0.586 |                         nan     |                                 0.693 |                                0.704 |                              0.618 |
| google:gemini-2.5-flash             |                          0.647 |                     1     |                   0.841 |                           0.979 |                                 0.729 |                                0.862 |                              0.774 |
| google:gemini-2.5-pro               |                          0.586 |                     0.841 |                   1     |                           0.972 |                                 0.738 |                                0.816 |                              0.752 |
| google:gemini-3.1-pro-preview       |                        nan     |                     0.979 |                   0.972 |                           1     |                                 0.27  |                                0.936 |                              0.619 |
| anthropic:claude-haiku-4-5-20251001 |                          0.693 |                     0.729 |                   0.738 |                           0.27  |                                 1     |                                0.849 |                              0.8   |
| anthropic:claude-sonnet-4-20250514  |                          0.704 |                     0.862 |                   0.816 |                           0.936 |                                 0.849 |                                1     |                              0.884 |
| anthropic:claude-opus-4-20250514    |                          0.618 |                     0.774 |                   0.752 |                           0.619 |                                 0.8   |                                0.884 |                              1     |

## Correlation with Gold Metrics

| Judge | BERTScore | F1 (word overlap) |
|-------|----------|----------|
| google:gemini-2.5-flash-lite | 0.317 | 0.264 |
| google:gemini-2.5-flash | 0.447 | 0.362 |
| google:gemini-2.5-pro | 0.518 | 0.429 |
| google:gemini-3.1-pro-preview | 0.122 | 0.111 |
| anthropic:claude-haiku-4-5-20251001 | 0.640 | 0.568 |
| anthropic:claude-sonnet-4-20250514 | 0.561 | 0.477 |
| anthropic:claude-opus-4-20250514 | 0.513 | 0.430 |

## Estimated Cost Breakdown

| Judge | Calls | Est. Cost/Call | Est. Total |
|-------|-------|----------------|------------|
| google:gemini-2.5-flash-lite | 150 | $0.0001 | $0.01 |
| google:gemini-2.5-flash | 150 | $0.0001 | $0.02 |
| google:gemini-2.5-pro | 150 | $0.0010 | $0.15 |
| google:gemini-3.1-pro-preview | 150 | $0.0100 | $1.50 |
| anthropic:claude-haiku-4-5-20251001 | 150 | $0.0010 | $0.15 |
| anthropic:claude-sonnet-4-20250514 | 150 | $0.0050 | $0.75 |
| anthropic:claude-opus-4-20250514 | 150 | $0.0100 | $1.50 |

## Gold Correctness Summary

- Exact match rate: 74.0%
- Mean word-overlap F1: 0.535
- Mean BERTScore F1: 0.917

## Answer Quality Distribution

- **good**: 73 (48.7%)
- **questionable**: 7 (4.7%)
- **poor**: 70 (46.7%)

## Failure Stage Breakdown

- **none**: 111 (74.0%)
- **retrieval**: 19 (12.7%)
- **generation**: 19 (12.7%)
- **chunker**: 1 (0.7%)

## Recommendation

*Review the correlation matrix and gold metric correlations above.*
BERTScore (semantic) is more reliable than word-overlap F1 for generated text.
Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.
