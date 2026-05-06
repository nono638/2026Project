# Result: task-053 — Quantization tracking
**Status:** done
**Completed:** 2026-05-06T12:35:03

## Commits
(filled in by tracker entry on main)

## Test Results
- New tests: `pytest tests/test_quantization_tracking.py -v` → 16 passed, 0 failed
- Full regression: `pytest tests/ --ignore=DaytimeNighttimeHandOff/ -q` → 669 passed, 0 failed

## Decisions Made
- **Bare tags retained for the existing Qwen3/Gemma3 model lists in `run_experiment_1.py`, `run_experiment_2.py`, and `scripts/pull_models.py`.** Spec section 4 asks for explicit `-q4_K_M` pinning where variants exist, with a verification step against ollama.com. Network is blocked from the nighttime sandbox (CLAUDE.md hard constraint), so the verification couldn't run. Added explanatory comments referencing the runtime helper which captures the resolved quant on every row regardless of tag form. Setup_pod.py was pinned to `qwen3:4b-q4_K_M` — that change was an explicit literal change in the spec, not gated on verification.
- **`requests` import moved to module-level** in `scripts/experiment_utils.py`. The lazy local import worked at runtime but couldn't be cleanly mocked via `patch("experiment_utils.requests.post", ...)` from tests. Top-level import keeps the public surface identical.
- **`model_details_by_tag` initialized to `{}` before the if/else in Exp 1 and Exp 2.** Some paths only build it conditionally; passing `model_details=model_details_by_tag or None` keeps the metadata.json clean (empty dict is omitted by `write_experiment_metadata`).
- **`backfill_quant_metadata.py` collects tags from `config.model`, `config.models`, and `extra.test_models`** to handle Exp 0 (single `model`) and Exp 1/2 (`models` list) shapes.
- **Bare host like `gpu-pod:11434` gets a `http://` prefix automatically** — the Ollama Python client accepts bare hosts; the helper now matches that ergonomics so callers can pass either form.

## Flags for Morning Review
- **Live Ollama verification still pending.** Spec section 7 asks for a manual `get_ollama_model_details('qwen3:4b')` call against a running Ollama (5090 box) and the output pasted here. Couldn't run from the sandbox. Suggested one-liner for morning:
  ```
  python -c "from scripts.experiment_utils import get_ollama_model_details; print(get_ollama_model_details('qwen3:4b'))"
  ```
- **Tag-explicit pinning across `run_experiment_1.py` model list deferred** as noted above. Once the 5090 has the models pulled, run `ollama show <tag>` for each and replace bare tags with the published explicit-quant variants where they exist. The runtime quant-stamping makes this a code-tidiness improvement, not a correctness blocker.
- **`docs/running-experiments.md` example commands kept bare tags** (e.g., `qwen3:4b`) — same rationale; once tag pinning lands, update the example block in lockstep.

## Attempted Approaches (if skipped/blocked)
N/A — completed.
