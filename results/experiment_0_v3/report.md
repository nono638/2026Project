# Experiment 0: Scorer Validation Report

## Per-Judge Mean Scores

| Judge | Faithfulness | Relevance | Conciseness | Quality |
|-------|-------------|-----------|-------------|---------|
| google:gemini-2.5-flash-lite | 4.568 | 4.658 | 4.786 | 4.671 |
| google:gemini-2.5-flash | 4.684 | 4.844 | 4.700 | 4.743 |
| google:gemini-2.5-pro | 4.673 | 4.901 | 4.667 | 4.747 |
| anthropic:claude-haiku-4-5-20251001 | 4.592 | 4.668 | 4.450 | 4.570 |
| anthropic:claude-sonnet-4-20250514 | 4.652 | 4.766 | 4.582 | 4.667 |
| anthropic:claude-sonnet-4-6 | 4.712 | 4.608 | 4.254 | 4.525 |
| anthropic:claude-opus-4-20250514 | 4.722 | 4.804 | 4.664 | 4.730 |
| anthropic:claude-opus-4-7 | 4.712 | 4.591 | 4.106 | 4.470 |
| openai:gpt-5.4-mini | 4.660 | 4.680 | 4.514 | 4.618 |
| openai:gpt-5.4 | 4.706 | 4.600 | 4.472 | 4.593 |

## Inter-Scorer Correlation (Pearson, Quality)

|                                     |   google:gemini-2.5-flash-lite |   google:gemini-2.5-flash |   google:gemini-2.5-pro |   anthropic:claude-haiku-4-5-20251001 |   anthropic:claude-sonnet-4-20250514 |   anthropic:claude-sonnet-4-6 |   anthropic:claude-opus-4-20250514 |   anthropic:claude-opus-4-7 |   openai:gpt-5.4-mini |   openai:gpt-5.4 |
|:------------------------------------|-------------------------------:|--------------------------:|------------------------:|--------------------------------------:|-------------------------------------:|------------------------------:|-----------------------------------:|----------------------------:|----------------------:|-----------------:|
| google:gemini-2.5-flash-lite        |                          1     |                     0.566 |                   0.578 |                                 0.557 |                                0.564 |                         0.406 |                              0.516 |                       0.597 |                 0.399 |            0.422 |
| google:gemini-2.5-flash             |                          0.566 |                     1     |                   0.892 |                                 0.634 |                                0.716 |                         0.574 |                              0.737 |                       0.728 |                 0.48  |            0.602 |
| google:gemini-2.5-pro               |                          0.578 |                     0.892 |                   1     |                                 0.674 |                                0.733 |                         0.584 |                              0.731 |                       0.774 |                 0.513 |            0.635 |
| anthropic:claude-haiku-4-5-20251001 |                          0.557 |                     0.634 |                   0.674 |                                 1     |                                0.738 |                         0.668 |                              0.751 |                       0.719 |                 0.549 |            0.644 |
| anthropic:claude-sonnet-4-20250514  |                          0.564 |                     0.716 |                   0.733 |                                 0.738 |                                1     |                         0.678 |                              0.82  |                       0.799 |                 0.531 |            0.713 |
| anthropic:claude-sonnet-4-6         |                          0.406 |                     0.574 |                   0.584 |                                 0.668 |                                0.678 |                         1     |                              0.645 |                       0.926 |                 0.72  |            0.796 |
| anthropic:claude-opus-4-20250514    |                          0.516 |                     0.737 |                   0.731 |                                 0.751 |                                0.82  |                         0.645 |                              1     |                       0.784 |                 0.507 |            0.645 |
| anthropic:claude-opus-4-7           |                          0.597 |                     0.728 |                   0.774 |                                 0.719 |                                0.799 |                         0.926 |                              0.784 |                       1     |                 0.852 |            0.921 |
| openai:gpt-5.4-mini                 |                          0.399 |                     0.48  |                   0.513 |                                 0.549 |                                0.531 |                         0.72  |                              0.507 |                       0.852 |                 1     |            0.784 |
| openai:gpt-5.4                      |                          0.422 |                     0.602 |                   0.635 |                                 0.644 |                                0.713 |                         0.796 |                              0.645 |                       0.921 |                 0.784 |            1     |

## Correlation with Gold Metrics

| Judge | BERTScore | F1 (word overlap) |
|-------|----------|----------|
| google:gemini-2.5-flash-lite | 0.137 | 0.139 |
| google:gemini-2.5-flash | 0.312 | 0.301 |
| google:gemini-2.5-pro | 0.353 | 0.348 |
| anthropic:claude-haiku-4-5-20251001 | 0.461 | 0.450 |
| anthropic:claude-sonnet-4-20250514 | 0.414 | 0.397 |
| anthropic:claude-sonnet-4-6 | 0.623 | 0.575 |
| anthropic:claude-opus-4-20250514 | 0.392 | 0.382 |
| anthropic:claude-opus-4-7 | 0.698 | 0.635 |
| openai:gpt-5.4-mini | 0.607 | 0.553 |
| openai:gpt-5.4 | 0.638 | 0.605 |

## Estimated Cost Breakdown

| Judge | Calls | Est. Cost/Call | Est. Total |
|-------|-------|----------------|------------|
| google:gemini-2.5-flash-lite | 500 | $0.0001 | $0.03 |
| google:gemini-2.5-flash | 500 | $0.0001 | $0.05 |
| google:gemini-2.5-pro | 500 | $0.0010 | $0.50 |
| anthropic:claude-haiku-4-5-20251001 | 500 | $0.0010 | $0.50 |
| anthropic:claude-sonnet-4-20250514 | 500 | $0.0050 | $2.50 |
| anthropic:claude-sonnet-4-6 | 500 | $0.0050 | $2.50 |
| anthropic:claude-opus-4-20250514 | 500 | $0.0100 | $5.00 |
| anthropic:claude-opus-4-7 | 500 | $0.0100 | $5.00 |
| openai:gpt-5.4-mini | 500 | $0.0015 | $0.75 |
| openai:gpt-5.4 | 500 | $0.0050 | $2.50 |

## Gold Correctness Summary

- Exact match rate: 76.2%
- Mean word-overlap F1: 0.546
- Mean BERTScore F1: 0.917

## Answer Quality Distribution

- **good**: 257 (51.4%)
- **questionable**: 18 (3.6%)
- **poor**: 225 (45.0%)

## Failure Stage Breakdown

- **none**: 381 (76.2%)
- **retrieval**: 70 (14.0%)
- **generation**: 48 (9.6%)
- **chunker**: 1 (0.2%)

## Recommendation

*Review the correlation matrix and gold metric correlations above.*
BERTScore (semantic) is more reliable than word-overlap F1 for generated text.
Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.
