# Experiment 2: Chunking x Model Size Report

## Chunker x Model Quality Heatmap

| chunker   |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:----------|---------------:|-------------:|-------------:|-------------:|
| fixed     |          3.732 |        4.221 |        4.639 |        4.501 |
| recursive |          3.778 |        4.216 |        4.615 |        4.46  |
| semantic  |          3.872 |        4.076 |        4.666 |        4.522 |
| sentence  |          3.757 |        4.051 |        4.492 |        4.418 |

## Per-Chunker Ranking

| chunker   |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| semantic  |  4.284 | 0.908 |     800 |
| fixed     |  4.273 | 0.907 |     800 |
| recursive |  4.267 | 0.91  |     800 |
| sentence  |  4.179 | 0.93  |     800 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:4b   |  4.603 | 0.666 |     800 |
| qwen3.5:9b   |  4.475 | 0.621 |     800 |
| qwen3.5:2b   |  4.141 | 0.963 |     800 |
| qwen3.5:0.8b |  3.785 | 1.09  |     800 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.703 | 3199 |

## Chunking Impact Analysis

Quality delta (chunker mean - overall mean) by model size:

| Chunker | Model | Mean Quality | Delta vs Overall |
|---------|-------|-------------|------------------|
| fixed | qwen3.5:0.8b | 3.732 | -0.519 |
| fixed | qwen3.5:2b | 4.221 | -0.030 |
| fixed | qwen3.5:4b | 4.639 | +0.388 |
| fixed | qwen3.5:9b | 4.501 | +0.250 |
| recursive | qwen3.5:0.8b | 3.778 | -0.473 |
| recursive | qwen3.5:2b | 4.216 | -0.035 |
| recursive | qwen3.5:4b | 4.615 | +0.364 |
| recursive | qwen3.5:9b | 4.460 | +0.209 |
| semantic | qwen3.5:0.8b | 3.873 | -0.378 |
| semantic | qwen3.5:2b | 4.076 | -0.175 |
| semantic | qwen3.5:4b | 4.666 | +0.415 |
| semantic | qwen3.5:9b | 4.522 | +0.272 |
| sentence | qwen3.5:0.8b | 3.757 | -0.494 |
| sentence | qwen3.5:2b | 4.051 | -0.200 |
| sentence | qwen3.5:4b | 4.492 | +0.241 |
| sentence | qwen3.5:9b | 4.418 | +0.167 |

### Best Chunker per Model

- **qwen3.5:0.8b**: semantic (3.873)
- **qwen3.5:2b**: fixed (4.221)
- **qwen3.5:4b**: semantic (4.666)
- **qwen3.5:9b**: semantic (4.522)

## Latency Summary

|                               |   mean |   median |   std |
|:------------------------------|-------:|---------:|------:|
| ('fixed', 'qwen3.5:0.8b')     |   1533 |     1215 |  3371 |
| ('fixed', 'qwen3.5:2b')       |   1450 |     1351 |   686 |
| ('fixed', 'qwen3.5:4b')       |   1470 |     1266 |   603 |
| ('fixed', 'qwen3.5:9b')       |   2627 |     2268 |  1555 |
| ('recursive', 'qwen3.5:0.8b') |   1373 |     1077 |  2590 |
| ('recursive', 'qwen3.5:2b')   |   1191 |     1006 |   997 |
| ('recursive', 'qwen3.5:4b')   |   1172 |      905 |   986 |
| ('recursive', 'qwen3.5:9b')   |   2520 |     1823 |  5172 |
| ('semantic', 'qwen3.5:0.8b')  |   1497 |     1225 |  1737 |
| ('semantic', 'qwen3.5:2b')    |   3059 |     1520 | 20059 |
| ('semantic', 'qwen3.5:4b')    |   1692 |     1390 |  1548 |
| ('semantic', 'qwen3.5:9b')    |   4876 |     2211 | 33402 |
| ('sentence', 'qwen3.5:0.8b')  |   1627 |     1110 |  5057 |
| ('sentence', 'qwen3.5:2b')    |   1752 |     1277 |  4246 |
| ('sentence', 'qwen3.5:4b')    |   2102 |     1027 |  7311 |
| ('sentence', 'qwen3.5:9b')    |   3599 |     2137 |  8242 |

## Gold Metrics Summary

- Mean gold F1: 0.272
- Exact match rate: 72.6%
- Mean BERTScore F1: 0.873

## Cost Summary

- Total scored answers (any judge non-NaN): 3200
