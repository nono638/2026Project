# Experiment 1: Strategy x Model Size Report

## Strategy x Model Quality Heatmap

| strategy   |   qwen3.5:0.8b |   qwen3.5:2b |
|:-----------|---------------:|-------------:|
| naive      |          4.117 |        4     |
| self_rag   |          3.05  |        3.083 |

## Per-Strategy Ranking

| strategy   |   mean |   std |   count |
|:-----------|-------:|------:|--------:|
| naive      |  4.058 | 0.808 |      20 |
| self_rag   |  3.067 | 1.133 |      20 |

## Per-Model Ranking

| model        |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| qwen3.5:0.8b |  3.583 | 1.11  |      20 |
| qwen3.5:2b   |  3.542 | 1.105 |      20 |

## Per-Judge Agreement

Pearson correlation of per-row quality between each judge pair:

| Judge A | Judge B | Pearson r | n |
|---------|---------|-----------|---|
| anthropic_claude_haiku_4_5_20251001 | openai_gpt_5_4_mini | 0.640 | 40 |

## Strategy Beats Size Analysis

Cases where a smaller model with a non-naive strategy outperforms a larger model with NaiveRAG:

**0 cases found.**


## Latency Summary

|                              |   mean |   median |   std |
|:-----------------------------|-------:|---------:|------:|
| ('naive', 'qwen3.5:0.8b')    |    459 |      390 |   252 |
| ('naive', 'qwen3.5:2b')      |   1159 |      635 |  1256 |
| ('self_rag', 'qwen3.5:0.8b') |   1815 |     1892 |   675 |
| ('self_rag', 'qwen3.5:2b')   |   6555 |     1672 | 14941 |

## Gold Metrics Summary

- Mean gold F1: 0.220
- Exact match rate: 50.0%
- Mean BERTScore F1: 0.865

## Cost Summary

- Total scored answers (any judge non-NaN): 40
