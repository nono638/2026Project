# Result: task-054 — Update Exp 1/2 model matrices to current open weights
**Status:** done
**Completed:** 2026-05-06T12:53:42

## Commits
(filled in by tracker entry on main)

## Test Results
- Dashboard tests: `pytest tests/test_experiment1_dashboard.py tests/test_experiment2_dashboard.py -q` → 22 passed, 0 failed
- Full regression: `pytest tests/ --ignore=DaytimeNighttimeHandOff/ -q` → 653 passed, 0 failed

## Decisions Made
- **MODEL_ORDER in `generate_experiment1_dashboard.py` ordered by effective active params** (so MoE entries like `qwen3.6:35b-a3b` and `gemma4:26b` sit between similarly-active dense models, not at their nominal totals). This keeps x-axis ordering meaningful for MoE comparisons.
- **Approximate VRAM/disk numbers in `runpod-setup-guide.md` are estimates** for the new models (Qwen 3.5/3.6, Gemma 4) — not measured. Marked them as `~` with the caveat that the implementer didn't pull the artifacts. Daytime should refresh once the 5090 has the actual download sizes.
- **`scripts/run_experiment.py` (the legacy single-script entry point referenced in README.md)** still uses bare `qwen3.5:4b` in its example. The legacy script may not even know about the new tags — left unchanged (per task-053's "What NOT to touch" note about that script being superseded), only the README example command line was updated.
- **Test files like `tests/test_setup_pod.py` retain `qwen3:4b` strings** as inert test fixtures (they call `pull_model("http://fake:11434", "qwen3:4b")` with a mocked HTTP client; the tag string isn't asserted against any real registry). Updating them would be cosmetic.

## Flags for Morning Review
- **Tag verification (spec section 7) couldn't be performed.** Network is blocked from the nighttime sandbox, so `curl -sI https://ollama.com/library/...` checks couldn't run. The 10 tags from spec are baked into the matrix verbatim. Before the first `run_experiment_1` launch, run a quick existence check against ollama.com or a `ollama show <tag>` on the 5090. Tags to verify:
  - qwen3.5:0.8b, qwen3.5:2b, qwen3.5:4b, qwen3.5:9b
  - qwen3.6:27b, qwen3.6:35b-a3b
  - gemma4:e2b, gemma4:e4b, gemma4:26b, gemma4:31b
- **Coordinate with task-053 branch** — both touch `deploy/setup_pod.py:DEFAULT_MODELS` and `scripts/pull_models.py:REQUIRED_MODELS`. Resulting state on this branch (no task-053): `["mxbai-embed-large", "qwen3.5:4b"]`. If task-053 lands first, fold its `-q4_K_M` pin into the model name. If both branches merge cleanly, the daytime reviewer should reconcile by running:
  ```
  DEFAULT_MODELS = ["mxbai-embed-large", "qwen3.5:4b-q4_K_M"]
  ```
- **VRAM estimates in `runpod-setup-guide.md`** (~16 GB for 27b/26b, ~22 GB for 35b-a3b, ~19 GB for 31b) are best-guesses based on Q4_K_M sizes. Should be replaced with measured numbers from the 5090 once models are pulled.

## Attempted Approaches (if skipped/blocked)
N/A — completed.
