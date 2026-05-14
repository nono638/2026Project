# Experiment 2: Chunking x Model Size Report

## Chunker x Model Quality Heatmap

| chunker   |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:----------|---------------:|-------------:|-------------:|-------------:|
| fixed     |          3.752 |        4.186 |        4.658 |        4.463 |
| recursive |          3.732 |        4.102 |        4.473 |        4.278 |
| sentence  |          3.671 |        3.959 |        4.468 |        4.299 |

## Per-Chunker Ranking

| chunker   |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| fixed     |  4.265 | 0.899 |     800 |
| recursive |  4.146 | 0.951 |     800 |
| sentence  |  4.099 | 1.004 |     800 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:4b   |  4.533 | 0.724 |     600 |
| qwen3.5:9b   |  4.347 | 0.737 |     600 |
| qwen3.5:2b   |  4.082 | 1.007 |     600 |
| qwen3.5:0.8b |  3.718 | 1.091 |     600 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.719 | 2399 |

## Chunking Impact Analysis

Quality delta (chunker mean - overall mean) by model size:

| Chunker | Model | Mean Quality | Delta vs Overall |
|---------|-------|-------------|------------------|
| fixed | qwen3.5:0.8b | 3.752 | -0.418 |
| fixed | qwen3.5:2b | 4.186 | +0.016 |
| fixed | qwen3.5:4b | 4.657 | +0.487 |
| fixed | qwen3.5:9b | 4.463 | +0.293 |
| recursive | qwen3.5:0.8b | 3.732 | -0.438 |
| recursive | qwen3.5:2b | 4.103 | -0.068 |
| recursive | qwen3.5:4b | 4.473 | +0.303 |
| recursive | qwen3.5:9b | 4.278 | +0.108 |
| sentence | qwen3.5:0.8b | 3.671 | -0.499 |
| sentence | qwen3.5:2b | 3.959 | -0.211 |
| sentence | qwen3.5:4b | 4.468 | +0.297 |
| sentence | qwen3.5:9b | 4.299 | +0.129 |

### Best Chunker per Model

- **qwen3.5:0.8b**: fixed (3.752)
- **qwen3.5:2b**: fixed (4.186)
- **qwen3.5:4b**: fixed (4.657)
- **qwen3.5:9b**: fixed (4.463)

## Latency Summary

|                               |   mean |   median |   std |
|:------------------------------|-------:|---------:|------:|
| ('fixed', 'qwen3.5:0.8b')     |   1279 |      999 |  1732 |
| ('fixed', 'qwen3.5:2b')       |   1304 |     1220 |   696 |
| ('fixed', 'qwen3.5:4b')       |   1521 |     1100 |  3127 |
| ('fixed', 'qwen3.5:9b')       |   3010 |     2172 |  4616 |
| ('recursive', 'qwen3.5:0.8b') |   1305 |      867 |  2705 |
| ('recursive', 'qwen3.5:2b')   |   1348 |      950 |  3657 |
| ('recursive', 'qwen3.5:4b')   |   1063 |      815 |   706 |
| ('recursive', 'qwen3.5:9b')   |   3008 |     1777 |  7692 |
| ('sentence', 'qwen3.5:0.8b')  |   1451 |      959 |  5110 |
| ('sentence', 'qwen3.5:2b')    |   1725 |     1090 |  4292 |
| ('sentence', 'qwen3.5:4b')    |   1811 |      900 |  4923 |
| ('sentence', 'qwen3.5:9b')    |   4672 |     1929 | 14438 |

## Gold Metrics Summary

- Mean gold F1: 0.263
- Exact match rate: 70.7%
- Mean BERTScore F1: 0.868

## Cost Summary

- Total scored answers (any judge non-NaN): 2400
