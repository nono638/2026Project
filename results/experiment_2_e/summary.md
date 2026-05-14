# Experiment 2-E: Embedder Size Sweep — Summary

Same chunker x model matrix from Experiment 2 (4 chunkers x 4 Qwen 3.5 models x 200 HotpotQA queries) re-run across 5 embedders to isolate the embedder's contribution.

## Overall mean quality by embedder (each embedder's completed configs)

| Embedder | Params (M) | Configs run | Mean consensus quality | Stdev |
|---|---:|---:|---:|---:|
| all-minilm:22m | 22 | 12 (partial; 4 configs missing) | 4.170 | 0.330 |
| nomic-embed-text | 137 | 16 | 4.248 | 0.336 |
| embeddinggemma:300m | 300 | 16 | 4.251 | 0.334 |
| qwen3-embedding:0.6b | 600 | 16 | 4.254 | 0.317 |
| qwen3-embedding:4b | 2500 | 16 | 4.232 | 0.328 |

## Apples-to-apples mean (only the 12 (chunker, model) cells present in every embedder's data)

| Embedder | Params (M) | Mean | Stdev |
|---|---:|---:|---:|
| all-minilm:22m | 22 | 4.170 | 0.330 |
| nomic-embed-text | 137 | 4.228 | 0.338 |
| embeddinggemma:300m | 300 | 4.240 | 0.338 |
| qwen3-embedding:0.6b | 600 | 4.244 | 0.309 |
| qwen3-embedding:4b | 2500 | 4.220 | 0.325 |

## Mean quality by (embedder, chunker), averaged across 4 Qwen 3.5 models

| Embedder | fixed | recursive | semantic | sentence |
|---|---:|---:|---:|---:|
| all-minilm:22m | 4.264 | 4.146 | — | 4.099 |
| nomic-embed-text | 4.338 | 4.211 | 4.306 | 4.136 |
| embeddinggemma:300m | 4.273 | 4.267 | 4.284 | 4.179 |
| qwen3-embedding:0.6b | 4.341 | 4.201 | 4.284 | 4.190 |
| qwen3-embedding:4b | 4.318 | 4.228 | 4.268 | 4.113 |
