# Experiment 1: Strategy x Model Size Report

## Strategy x Model Quality Heatmap

| strategy    |   gemma4:e2b |   gemma4:e4b |   qwen3.5:0.8b |   qwen3.5:2b |   qwen3.5:4b |   qwen3.5:9b |
|:------------|-------------:|-------------:|---------------:|-------------:|-------------:|-------------:|
| adaptive    |        4.017 |        4.162 |          2.174 |        3.966 |        3.435 |        3.811 |
| corrective  |        4.512 |        4.472 |          3.617 |        4.077 |        4.562 |        4.368 |
| multi_query |        4.447 |        4.396 |          3.851 |        4.215 |        4.584 |        4.513 |
| naive       |        4.451 |        4.44  |          3.856 |        4.258 |        4.589 |        4.447 |
| self_rag    |        2.418 |        3.474 |          2.87  |        3.106 |        3.082 |        3.003 |

## Per-Strategy Ranking

| strategy    |   mean |   std |   count |
|:------------|-------:|------:|--------:|
| naive       |  4.34  | 0.84  |    1199 |
| multi_query |  4.334 | 0.818 |    1200 |
| corrective  |  4.268 | 0.881 |    1200 |
| adaptive    |  3.594 | 1.252 |    1200 |
| self_rag    |  2.992 | 1.073 |    1196 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| gemma4:e4b   |  4.189 | 0.866 |    1000 |
| qwen3.5:4b   |  4.054 | 1.146 |     995 |
| qwen3.5:9b   |  4.028 | 1.057 |    1000 |
| gemma4:e2b   |  3.969 | 1.086 |    1000 |
| qwen3.5:2b   |  3.924 | 1.067 |    1000 |
| qwen3.5:0.8b |  3.274 | 1.244 |    1000 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.793 | 5993 |

## Strategy Beats Size Analysis

Cases where a smaller model with a non-naive strategy outperforms a larger model with NaiveRAG:

**8 cases found.**

- corrective + gemma4:e2b (4.512) > naive + gemma4:e4b (4.440) [+0.072]
- corrective + gemma4:e2b (4.512) > naive + qwen3.5:9b (4.447) [+0.065]
- corrective + gemma4:e4b (4.473) > naive + qwen3.5:9b (4.447) [+0.026]
- corrective + qwen3.5:4b (4.562) > naive + gemma4:e4b (4.440) [+0.122]
- corrective + qwen3.5:4b (4.562) > naive + qwen3.5:9b (4.447) [+0.116]
- multi_query + gemma4:e2b (4.447) > naive + gemma4:e4b (4.440) [+0.007]
- multi_query + qwen3.5:4b (4.584) > naive + gemma4:e4b (4.440) [+0.144]
- multi_query + qwen3.5:4b (4.584) > naive + qwen3.5:9b (4.447) [+0.137]

## Latency Summary

|                                 |   mean |   median |    std |
|:--------------------------------|-------:|---------:|-------:|
| ('adaptive', 'gemma4:e2b')      |  20643 |    19917 |   8182 |
| ('adaptive', 'gemma4:e4b')      |   2977 |     1509 |   2733 |
| ('adaptive', 'qwen3.5:0.8b')    |   1768 |     1385 |   3997 |
| ('adaptive', 'qwen3.5:2b')      |   2244 |     1678 |   1707 |
| ('adaptive', 'qwen3.5:4b')      |  10253 |     2120 |  47104 |
| ('adaptive', 'qwen3.5:9b')      |  21984 |    21255 |  10116 |
| ('corrective', 'gemma4:e2b')    |  10875 |    10066 |   3271 |
| ('corrective', 'gemma4:e4b')    |   7504 |     9827 |   4474 |
| ('corrective', 'qwen3.5:0.8b')  |   3653 |     3330 |   1184 |
| ('corrective', 'qwen3.5:2b')    |   4588 |     3380 |   4531 |
| ('corrective', 'qwen3.5:4b')    |   5014 |     3531 |   6390 |
| ('corrective', 'qwen3.5:9b')    |  13527 |    11582 |   5165 |
| ('multi_query', 'gemma4:e2b')   |  20689 |    20677 |    490 |
| ('multi_query', 'gemma4:e4b')   |  21922 |    21871 |    513 |
| ('multi_query', 'qwen3.5:0.8b') |   2306 |     2321 |    577 |
| ('multi_query', 'qwen3.5:2b')   |   3125 |     2677 |   4769 |
| ('multi_query', 'qwen3.5:4b')   |   2744 |     2638 |    447 |
| ('multi_query', 'qwen3.5:9b')   |  16861 |    22274 |   9637 |
| ('naive', 'gemma4:e2b')         |   9077 |     9025 |   1209 |
| ('naive', 'gemma4:e4b')         |   9757 |     9584 |   1875 |
| ('naive', 'qwen3.5:0.8b')       |   1407 |     1118 |   3164 |
| ('naive', 'qwen3.5:2b')         |   1220 |     1131 |    649 |
| ('naive', 'qwen3.5:4b')         |   1573 |      974 |   4397 |
| ('naive', 'qwen3.5:9b')         |   2232 |     1801 |   1557 |
| ('self_rag', 'gemma4:e2b')      |   9931 |     9636 |   1565 |
| ('self_rag', 'gemma4:e4b')      |  18298 |    22455 |   6804 |
| ('self_rag', 'qwen3.5:0.8b')    |   4711 |     4513 |   4461 |
| ('self_rag', 'qwen3.5:2b')      |   7384 |     4465 |  18839 |
| ('self_rag', 'qwen3.5:4b')      |  91284 |     4447 | 544702 |
| ('self_rag', 'qwen3.5:9b')      |   4767 |     4218 |   2726 |

## Gold Metrics Summary

- Mean gold F1: 0.212
- Exact match rate: 59.3%
- Mean BERTScore F1: 0.860

## Cost Summary

- Total scored answers (any judge non-NaN): 5995
