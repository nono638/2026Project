# Experiment 2: Chunking x Model Size Report

## Chunker x Model Quality Heatmap

| chunker   |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:----------|---------------:|-------------:|-------------:|-------------:|
| fixed     |          3.997 |        4.175 |        4.667 |        4.526 |
| recursive |          3.72  |        4.192 |        4.538 |        4.353 |
| semantic  |          3.831 |        4.116 |        4.706 |        4.483 |
| sentence  |          3.752 |        4.12  |        4.517 |        4.37  |

## Per-Chunker Ranking

| chunker   |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| fixed     |  4.341 | 0.844 |     800 |
| semantic  |  4.285 | 0.914 |     799 |
| recursive |  4.201 | 0.932 |     800 |
| sentence  |  4.19  | 0.972 |     800 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:4b   |  4.607 | 0.658 |     800 |
| qwen3.5:9b   |  4.433 | 0.686 |     800 |
| qwen3.5:2b   |  4.151 | 0.947 |     800 |
| qwen3.5:0.8b |  3.825 | 1.108 |     799 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.707 | 3199 |

## Chunking Impact Analysis

Quality delta (chunker mean - overall mean) by model size:

| Chunker | Model | Mean Quality | Delta vs Overall |
|---------|-------|-------------|------------------|
| fixed | qwen3.5:0.8b | 3.997 | -0.257 |
| fixed | qwen3.5:2b | 4.175 | -0.079 |
| fixed | qwen3.5:4b | 4.667 | +0.413 |
| fixed | qwen3.5:9b | 4.526 | +0.272 |
| recursive | qwen3.5:0.8b | 3.720 | -0.534 |
| recursive | qwen3.5:2b | 4.192 | -0.062 |
| recursive | qwen3.5:4b | 4.538 | +0.284 |
| recursive | qwen3.5:9b | 4.353 | +0.099 |
| semantic | qwen3.5:0.8b | 3.831 | -0.423 |
| semantic | qwen3.5:2b | 4.116 | -0.138 |
| semantic | qwen3.5:4b | 4.706 | +0.452 |
| semantic | qwen3.5:9b | 4.483 | +0.229 |
| sentence | qwen3.5:0.8b | 3.752 | -0.502 |
| sentence | qwen3.5:2b | 4.120 | -0.134 |
| sentence | qwen3.5:4b | 4.517 | +0.263 |
| sentence | qwen3.5:9b | 4.370 | +0.116 |

### Best Chunker per Model

- **qwen3.5:0.8b**: fixed (3.997)
- **qwen3.5:2b**: recursive (4.192)
- **qwen3.5:4b**: semantic (4.706)
- **qwen3.5:9b**: fixed (4.526)

## Latency Summary

|                               |   mean |   median |   std |
|:------------------------------|-------:|---------:|------:|
| ('fixed', 'qwen3.5:0.8b')     |   1692 |     1042 |  5049 |
| ('fixed', 'qwen3.5:2b')       |   2005 |     1296 |  5707 |
| ('fixed', 'qwen3.5:4b')       |   1466 |     1238 |   782 |
| ('fixed', 'qwen3.5:9b')       |   2425 |     2135 |  1687 |
| ('recursive', 'qwen3.5:0.8b') |   1318 |      984 |  3649 |
| ('recursive', 'qwen3.5:2b')   |   1173 |      950 |  1021 |
| ('recursive', 'qwen3.5:4b')   |   1250 |      885 |  1918 |
| ('recursive', 'qwen3.5:9b')   |   2765 |     1815 |  6054 |
| ('semantic', 'qwen3.5:0.8b')  |   1246 |     1125 |   511 |
| ('semantic', 'qwen3.5:2b')    |   1572 |     1413 |  1296 |
| ('semantic', 'qwen3.5:4b')    |   1463 |     1248 |  1209 |
| ('semantic', 'qwen3.5:9b')    |   3209 |     2155 |  6177 |
| ('sentence', 'qwen3.5:0.8b')  |   1356 |     1052 |  2273 |
| ('sentence', 'qwen3.5:2b')    |   1191 |     1139 |   429 |
| ('sentence', 'qwen3.5:4b')    |   2036 |     1012 |  6766 |
| ('sentence', 'qwen3.5:9b')    |   7495 |     1979 | 54293 |

## Gold Metrics Summary

- Mean gold F1: 0.273
- Exact match rate: 72.3%
- Mean BERTScore F1: 0.873

## Cost Summary

- Total scored answers (any judge non-NaN): 3199
