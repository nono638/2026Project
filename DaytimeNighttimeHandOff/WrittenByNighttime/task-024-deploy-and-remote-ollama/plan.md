# Plan: task-024 — Deploy Script + Remote Ollama Support

## Files to Create
1. `deploy/setup_pod.py` — End-to-end pod setup script using RunPodManager
2. `tests/test_setup_pod.py` — Tests with mocked HTTP (7 tests)
3. `tests/test_ollama_embedder_host.py` — Tests for host param on OllamaEmbedder (2 tests)

## Files to Modify
1. `src/embedders/ollama.py` — Add `host` parameter to `__init__`
2. `scripts/run_experiment_0.py` — Add `--ollama-host` flag, thread through generate_answers and Ollama connectivity check
3. `scripts/run_experiment.py` — Add `--ollama-host` flag, thread through `_build_llm` and `_build_embedder`

## Approach
- Follow the spec exactly — ollama/ollama Docker image, HTTP API for model pulls, stream:true for pulls
- OllamaEmbedder host param matches existing OllamaLLM pattern
- setup_pod.py uses argparse, dotenv, RunPodManager, and requests for Ollama HTTP API
- All tests mock HTTP calls — no real API/RunPod calls

## Ambiguities
- None significant — spec is detailed with exact function signatures and behavior
