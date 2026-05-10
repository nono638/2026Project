# Experiment 2: Chunking x Model Size Report

## Chunker x Model Quality Heatmap

| chunker   |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:----------|---------------:|-------------:|-------------:|-------------:|
| fixed     |          3.886 |        4.198 |        4.649 |        4.538 |
| recursive |          3.762 |        4.162 |        4.557 |        4.431 |
| semantic  |          3.771 |        4.172 |        4.629 |        4.501 |
| sentence  |          3.731 |        3.932 |        4.474 |        4.314 |

## Per-Chunker Ranking

| chunker   |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| fixed     |  4.318 | 0.893 |     800 |
| semantic  |  4.268 | 0.926 |     800 |
| recursive |  4.228 | 0.937 |     800 |
| sentence  |  4.113 | 0.981 |     800 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:4b   |  4.577 | 0.708 |     800 |
| qwen3.5:9b   |  4.446 | 0.646 |     800 |
| qwen3.5:2b   |  4.116 | 0.991 |     800 |
| qwen3.5:0.8b |  3.787 | 1.114 |     800 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.721 | 3200 |

## Chunking Impact Analysis

Quality delta (chunker mean - overall mean) by model size:

| Chunker | Model | Mean Quality | Delta vs Overall |
|---------|-------|-------------|------------------|
| fixed | qwen3.5:0.8b | 3.886 | -0.346 |
| fixed | qwen3.5:2b | 4.198 | -0.033 |
| fixed | qwen3.5:4b | 4.649 | +0.417 |
| fixed | qwen3.5:9b | 4.538 | +0.307 |
| recursive | qwen3.5:0.8b | 3.762 | -0.470 |
| recursive | qwen3.5:2b | 4.162 | -0.069 |
| recursive | qwen3.5:4b | 4.557 | +0.325 |
| recursive | qwen3.5:9b | 4.431 | +0.199 |
| semantic | qwen3.5:0.8b | 3.771 | -0.461 |
| semantic | qwen3.5:2b | 4.172 | -0.060 |
| semantic | qwen3.5:4b | 4.629 | +0.397 |
| semantic | qwen3.5:9b | 4.501 | +0.269 |
| sentence | qwen3.5:0.8b | 3.731 | -0.501 |
| sentence | qwen3.5:2b | 3.933 | -0.299 |
| sentence | qwen3.5:4b | 4.474 | +0.242 |
| sentence | qwen3.5:9b | 4.314 | +0.082 |

### Best Chunker per Model

- **qwen3.5:0.8b**: fixed (3.886)
- **qwen3.5:2b**: fixed (4.198)
- **qwen3.5:4b**: fixed (4.649)
- **qwen3.5:9b**: fixed (4.538)

## Latency Summary

|                               |   mean |   median |   std |
|:------------------------------|-------:|---------:|------:|
| ('fixed', 'qwen3.5:0.8b')     |   1229 |     1038 |  1594 |
| ('fixed', 'qwen3.5:2b')       |   1624 |     1300 |  2901 |
| ('fixed', 'qwen3.5:4b')       |   1495 |     1215 |  1409 |
| ('fixed', 'qwen3.5:9b')       |   2837 |     2183 |  4341 |
| ('recursive', 'qwen3.5:0.8b') |   1462 |      951 |  4685 |
| ('recursive', 'qwen3.5:2b')   |   1697 |      971 |  4255 |
| ('recursive', 'qwen3.5:4b')   |   1490 |      873 |  3789 |
| ('recursive', 'qwen3.5:9b')   |   2868 |     1786 |  6472 |
| ('semantic', 'qwen3.5:0.8b')  |   1261 |     1097 |   730 |
| ('semantic', 'qwen3.5:2b')    |   1484 |     1306 |   970 |
| ('semantic', 'qwen3.5:4b')    |   1635 |     1191 |  3787 |
| ('semantic', 'qwen3.5:9b')    |   3275 |     2023 |  9803 |
| ('sentence', 'qwen3.5:0.8b')  |   1191 |     1009 |   866 |
| ('sentence', 'qwen3.5:2b')    |   2415 |     1200 |  6686 |
| ('sentence', 'qwen3.5:4b')    |   1301 |      964 |  1054 |
| ('sentence', 'qwen3.5:9b')    |   2919 |     2020 |  5616 |

## Gold Metrics Summary

- Mean gold F1: 0.269
- Exact match rate: 73.7%
- Mean BERTScore F1: 0.872

## Cost Summary

- Total scored answers (any judge non-NaN): 3200
