# Plan: task-053 quantization tracking

## Files to modify
1. `scripts/experiment_utils.py`
   - Add `get_ollama_model_details(model_tag, host=None) -> dict`. POST `/api/show` with
     `requests`, 10s timeout, never raise. Always populate `tag` and `captured_at`; other
     fields None on failure.
   - Add `model_details` kwarg (default None) to `write_experiment_metadata`. When not
     None, write under top-level `model_details`.
   - Modify `generate_answer(...)` to accept an optional `model_details: dict | None`
     argument. If provided, set `llm_quantization` in the returned dict to
     `model_details.get("quantization_level") or "unknown"`. If not provided or None,
     set `llm_quantization = "unknown"`.
2. `scripts/run_experiment_0.py` — at startup, build `model_details = {args.model: get_ollama_model_details(args.model, host=args.ollama_host)}`. Pass to generation loop and to `write_experiment_metadata`. The Exp 0 generation builds rows directly (not via `generate_answer`), so add `llm_quantization` to its row dict.
3. `scripts/run_experiment_1.py` — build a `model_details` map keyed by model tag at startup (one lookup per unique model in `models`). Cache as a module-level dict (or local). Pass to `generate_answer(...)` for each row, and to `write_experiment_metadata`.
4. `scripts/run_experiment_2.py` — same.
5. `deploy/setup_pod.py` — pin `qwen3:4b` → `qwen3:4b-q4_K_M` per spec table. Keep `mxbai-embed-large` bare.
6. `scripts/pull_models.py` — verify exists; pin same.
7. `docs/running-experiments.md` and `docs/output-format.md` — update example tags + add `llm_quantization` column note in output-format.md.
8. `scripts/backfill_quant_metadata.py` — new one-shot CLI taking experiment dir(s); reads metadata.json; for each model in config.model / extra.test_models / config.models, calls `get_ollama_model_details`, merges results into a top-level `model_details` map; adds `backfill_note`. Idempotent skip when `model_details` already present.
9. `tests/test_quantization_tracking.py` — new test file with the 9 cases listed in spec.

## Live Ollama verification
Spec asks to run `get_ollama_model_details('qwen3:4b')` against a live Ollama and paste
output. Network access is blocked in this nighttime environment (per CLAUDE.md hard
constraints) and there's no local Ollama running here. **Will note this in result.md
flags** so morning can run the one-line verification on the 5090 before merging.

## What to leave alone
- Existing CSVs/metadata in `results/` directories.
- `OllamaLLM`/`OllamaEmbedder` classes in `src/`.
- Cloud-judge code paths.

## Key design points
- `get_ollama_model_details` returns dict with keys: `tag, digest, quantization_level, parameter_size, family, format, captured_at`. None values on any failure.
- Use `requests.post` with json body; if response.ok, parse `details` dict and top-level `digest`. Treat absence of fields as None, not error.
- Lookup map keyed by tag; experiment scripts call once per unique tag at startup; row-build reads `quantization_level` from cache.
- For tags Exp 1/2 will include after task-054 (Qwen 3.5/3.6, Gemma 4): leave alone in this task — task-053 only handles current matrices, task-054 will land the new tags. Spec confirms: "Independent from task-052 and task-053 — coordinate setup_pod.py change with task-053."
