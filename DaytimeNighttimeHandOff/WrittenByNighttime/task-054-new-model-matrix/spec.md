# task-054: Update Exp 1 / Exp 2 model matrices to latest open weights (Qwen 3.5/3.6 + Gemma 4)

## Why

The Qwen3 (April 2025) and Gemma 3 (early 2026) model families used in the original
spec are now superseded by Qwen 3.5 small (Mar 2026), Qwen 3.6 (Apr 2026), and
Gemma 4 (Apr 2026). Daytime decision on 2026-05-06: replace both with current open
weights so the writeup's headline matrix uses the same generation of models a reader
would actually run today.

Exp 1 grows from 30 → **50 configurations**. Exp 2 stays at 16 configurations but
swaps to Qwen 3.5 small.

This task is **infrastructure only** — it updates the model lists, default tags, and
documentation. It does **not** run the experiments. That happens after the user
finishes the 5090 migration.

## Requirements

### 1. Exp 1 model matrix — `scripts/run_experiment_1.py`

Replace the existing 6-model `ALL_MODELS` list (currently around line 87) with this
exact ordering (small → large within each family, then cross-family):

```python
ALL_MODELS = [
    # Qwen 3.5 small (Apache 2.0, released 2026-03-02) — full small-tier size curve
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    # Qwen 3.6 large (Apache 2.0, released 2026-04-16/22) — large-tier
    "qwen3.6:27b",          # dense
    "qwen3.6:35b-a3b",      # MoE, ~3B active
    # Gemma 4 (Apache 2.0, released 2026-04-02) — cross-family
    "gemma4:e2b",           # 2.3B effective — pairs with qwen3.5:2b
    "gemma4:e4b",           # 4.5B effective — pairs with qwen3.5:4b
    "gemma4:26b",           # MoE, 3.8B active — pairs with qwen3.6:35b-a3b
    "gemma4:31b",           # dense — pairs with qwen3.6:27b
]
```

5 strategies × 10 models = **50 configurations**. Update the module docstring
(currently says "5 strategies x 6 models = 30 configurations") to reflect the new
counts and the matched-pair rationale.

Also update the file header comment line 7 model enumeration.

### 2. Exp 2 model matrix — `scripts/run_experiment_2.py`

Locate the equivalent `ALL_MODELS` (or whatever it's called — verify by reading the
file) and swap to:

```python
ALL_MODELS = [
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
]
```

4 chunkers × 4 models = **16 configurations** (unchanged scope). Update docstring
and any literal "Qwen3" → "Qwen 3.5" references in the file.

### 3. Default model lists in deployment scripts

- `deploy/setup_pod.py:60` — `DEFAULT_MODELS` currently `["mxbai-embed-large", "qwen3:4b"]`. RunPod isn't going away (still useful as fallback), so keep it functional. Update to use Qwen 3.5: `["mxbai-embed-large", "qwen3.5:4b"]`. **Note:** task-053 will further pin explicit quant tags (`qwen3.5:4b-q4_K_M`); coordinate with that branch — whichever lands second should fold the other's change.
- `scripts/pull_models.py` — read this file to find the model list it pulls. Update
  it to pull all 10 Exp 1 models + the 4 Exp 2 models (deduplicated, since Exp 2's
  models are a subset of Exp 1's small Qwen tier) + `mxbai-embed-large`. The
  resulting list is 11 models. Keep the script's `--models` override flag working.

### 4. Documentation refresh

- `docs/output-format.md:82` and `:88` — update model literal lists.
- `docs/running-experiments.md` — any commands using `qwen3:4b` example tags should
  be updated to `qwen3.5:4b`. Update troubleshooting/expected-runtime sections to
  reflect 50 (Exp 1) and 16 (Exp 2) configs.
- `docs/methodology.html` and `docs/index.html` — these are gallery-generated, not
  hand-edited. **Skip** — they regenerate when the user runs `generate_gallery.py`
  next. Just verify the gallery generator picks up the new model list (it pulls
  from CSV, not from Python constants).
- `README.md` — find any Qwen3 / Gemma 3 references and update.

### 5. setup_pod.py and pod-related guides

`DaytimeOnly/reference/runpod-setup-guide.md` references `qwen3:0.6b`, `qwen3:1.7b`,
etc. Update those tables to the new Qwen 3.5 + Qwen 3.6 + Gemma 4 set.

### 6. Tests

Find tests that hardcode model names (likely in `tests/test_run_experiment_1.py`,
`tests/test_run_experiment_2.py`, and possibly `tests/test_setup_pod.py` or
`tests/test_pull_models.py`). Update assertions to match the new lists. Do NOT
write tests that hit live Ollama — keep all model-tag assertions string-only.

If a test imports `ALL_MODELS` and asserts `len(ALL_MODELS) == 6`, change it to
`== 10`. Likewise for Exp 2 (was 4 Qwen3 — still 4 models, just different names,
so length-based assertions don't change but identity-based ones do).

### 7. Tag verification (one-time, recorded in nighttime_comments)

For each new tag, verify it exists on the public Ollama registry by hitting:
```
curl -sI https://ollama.com/library/<model>/<tag> | head -1
```
or the equivalent. Expected to find:
- qwen3.5:0.8b, qwen3.5:2b, qwen3.5:4b, qwen3.5:9b
- qwen3.6:27b, qwen3.6:35b-a3b
- gemma4:e2b, gemma4:e4b, gemma4:26b, gemma4:31b

**If any tag 404s**, do NOT silently substitute — flag it in `nighttime_comments`
with the failing tag and stop. Daytime will reconcile (e.g., Ollama may publish
`gemma4:35b-a3b` instead of `qwen3.6:35b-a3b` under a slightly different tag).

## What NOT to touch

- The `OllamaLLM` and `OllamaEmbedder` classes in `src/` — they're tag-agnostic,
  no changes needed.
- Existing experiment_0 results, scripts, or schemas — Exp 0 is locked.
- The reranker, scorer, chunker, embedder code paths — only the model dimension
  changes.
- `task-052` and `task-053` branches — those are independent. If both land first,
  this task rebases on top.

## Acceptance

- [ ] `ALL_MODELS` updated in both Exp 1 and Exp 2 scripts with the exact lists above.
- [ ] Docstrings, file headers, and inline comments reflect new counts and model names.
- [ ] `deploy/setup_pod.py` and `scripts/pull_models.py` default model lists updated.
- [ ] Documentation (`docs/output-format.md`, `docs/running-experiments.md`, `README.md`,
      `runpod-setup-guide.md`) refreshed.
- [ ] Tests pass after updates; no live-Ollama dependency added.
- [ ] All 10 model tags verified to exist on the Ollama registry; results pasted into
      `nighttime_comments`.
- [ ] No model generation runs are triggered by this task.

## References

- Qwen 3.5 small announcement: https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/
- Qwen 3.6 27B announcement: https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen-3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/
- Gemma 4 launch: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- Ollama qwen3.5 tags: https://ollama.com/library/qwen3.5
- Ollama qwen3.6 tags: https://ollama.com/library/qwen3.6
- Ollama gemma4 tags: https://ollama.com/library/gemma4
