# Experiment 2: Chunking x Model Size Report

## Chunker x Model Quality Heatmap

| chunker   |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:----------|---------------:|-------------:|-------------:|-------------:|
| fixed     |          3.891 |        4.238 |        4.681 |        4.54  |
| recursive |          3.756 |        4.157 |        4.529 |        4.403 |
| semantic  |          3.856 |        4.159 |        4.686 |        4.522 |
| sentence  |          3.632 |        4.028 |        4.499 |        4.385 |

## Per-Chunker Ranking

| chunker   |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| fixed     |  4.338 | 0.851 |     800 |
| semantic  |  4.306 | 0.901 |     800 |
| recursive |  4.211 | 0.934 |     800 |
| sentence  |  4.136 | 0.987 |     799 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:4b   |  4.599 | 0.67  |     799 |
| qwen3.5:9b   |  4.462 | 0.642 |     800 |
| qwen3.5:2b   |  4.146 | 0.96  |     800 |
| qwen3.5:0.8b |  3.784 | 1.11  |     800 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.711 | 3199 |

## Chunking Impact Analysis

Quality delta (chunker mean - overall mean) by model size:

| Chunker | Model | Mean Quality | Delta vs Overall |
|---------|-------|-------------|------------------|
| fixed | qwen3.5:0.8b | 3.891 | -0.357 |
| fixed | qwen3.5:2b | 4.238 | -0.009 |
| fixed | qwen3.5:4b | 4.681 | +0.433 |
| fixed | qwen3.5:9b | 4.540 | +0.292 |
| recursive | qwen3.5:0.8b | 3.756 | -0.492 |
| recursive | qwen3.5:2b | 4.157 | -0.091 |
| recursive | qwen3.5:4b | 4.529 | +0.282 |
| recursive | qwen3.5:9b | 4.403 | +0.156 |
| semantic | qwen3.5:0.8b | 3.856 | -0.392 |
| semantic | qwen3.5:2b | 4.159 | -0.088 |
| semantic | qwen3.5:4b | 4.686 | +0.438 |
| semantic | qwen3.5:9b | 4.522 | +0.274 |
| sentence | qwen3.5:0.8b | 3.632 | -0.616 |
| sentence | qwen3.5:2b | 4.028 | -0.219 |
| sentence | qwen3.5:4b | 4.499 | +0.252 |
| sentence | qwen3.5:9b | 4.385 | +0.137 |

### Best Chunker per Model

- **qwen3.5:0.8b**: fixed (3.891)
- **qwen3.5:2b**: fixed (4.238)
- **qwen3.5:4b**: semantic (4.686)
- **qwen3.5:9b**: fixed (4.540)

## Latency Summary

|                               |   mean |   median |    std |
|:------------------------------|-------:|---------:|-------:|
| ('fixed', 'qwen3.5:0.8b')     |   1172 |      993 |   1012 |
| ('fixed', 'qwen3.5:2b')       |   1922 |     1183 |   5306 |
| ('fixed', 'qwen3.5:4b')       |   1834 |     1122 |   6247 |
| ('fixed', 'qwen3.5:9b')       |   2635 |     2048 |   3283 |
| ('recursive', 'qwen3.5:0.8b') |   1068 |      864 |   2132 |
| ('recursive', 'qwen3.5:2b')   |   1390 |      900 |   3748 |
| ('recursive', 'qwen3.5:4b')   |   1015 |      750 |   1072 |
| ('recursive', 'qwen3.5:9b')   |   2720 |     1671 |   7357 |
| ('semantic', 'qwen3.5:0.8b')  |   2356 |     1097 |  11894 |
| ('semantic', 'qwen3.5:2b')    |   1736 |     1372 |   2666 |
| ('semantic', 'qwen3.5:4b')    |   1397 |     1213 |    723 |
| ('semantic', 'qwen3.5:9b')    |   2880 |     2086 |   5565 |
| ('sentence', 'qwen3.5:0.8b')  |   1333 |      926 |   2759 |
| ('sentence', 'qwen3.5:2b')    |   1359 |     1088 |   2309 |
| ('sentence', 'qwen3.5:4b')    |  20752 |      903 | 268476 |
| ('sentence', 'qwen3.5:9b')    |   3140 |     1991 |   6205 |

## Gold Metrics Summary

- Mean gold F1: 0.262
- Exact match rate: 72.4%
- Mean BERTScore F1: 0.871

## Cost Summary

- Total scored answers (any judge non-NaN): 3199
