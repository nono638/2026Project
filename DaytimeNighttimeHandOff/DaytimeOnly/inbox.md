# Inbox

> **Managed by daytime Claude. Night instance does not read this file.**
>
> Zero-friction capture for anything that comes up mid-session but isn't being acted on
> right now. No judgment required to add — if it might matter, capture it with one sentence
> of context. Cleared completely at every session-open triage.

---

<!-- Add captures below with today's date header. Format:
## YYYY-MM-DD
- [one-line capture — what it is and why it came up]
-->

<!-- inbox cleared 2026-03-27 session -->
<!-- Triaged: CostGuard loop abort → done (task-048 merged). CostGuard Opus cost → small fix, do now during daytime. -->

## 2026-05-06
- **Refresh VRAM/disk estimates in runpod-setup-guide.md.** task-054 left ~16GB / ~22GB / ~19GB estimates for the new MoE/large models. After first pull on the 5090, replace with measured values.
- **Pin remaining model tags to `-q4_K_M` once verified.** task-053 left bare tags in `pull_models.py` REQUIRED_MODELS and Exp 1/2 ALL_MODELS pending verification. Once `ollama show <tag>` confirms the published variants on the 5090, swap in explicit pins for determinism.
- **`bert_score` not installed in local daytime venv.** 13 `tests/test_bertscore.py` tests fail with `ModuleNotFoundError: No module named 'bert_score'`. Night venv has it (passed there). Either install locally (`pip install bert_score`) or accept that test_bertscore is GPU-side only — decide and document in ENVIRONMENT.md.

<!-- 2026-05-06: VERIFY OLLAMA TAGS, live get_ollama_model_details smoke test, and backfill_quant_metadata.py — promoted to 5090-migration.md Steps 3.0, 4.5, 4.6 -->
<!-- 2026-05-06: task-055 retroactive Ollama-judge scoring on v3 — promoted to 5090-migration.md Step 8 -->

## 2026-05-07
- **5090 VRAM thrash from concurrent model runs (observed 2026-05-06 night).** Running multiple Ollama models in parallel caused them to bounce in and out of 24 GB VRAM, killing throughput. **Rule going forward: one model at a time, sequentially — finish a full pass with model A before launching model B.** Audit `run_experiment_1.py`, `run_experiment_2.py`, and `run_experiment_0.py --judges ollama` for any concurrency. Likely needs a `--sequential` enforcement or removal of any threadpool/asyncio.gather across models. Critical to fix before tonight's Exp 1 launch.
- **5090 crashed mid-task-055 (qwen3.6:27b judge pass).** Recovery checklist drafted at `DaytimeOnly/reference/5090-crash-recovery.md`. Awaiting access to 5090 tonight to inspect git state + data files. Until recovered, don't reset metadata.json or raw_scores.csv on the planning laptop.

## 2026-05-08 (early morning, 5090 second session)
- **Exp 1 IS RUNNING.** `python scripts/run_experiment_1.py --resume --max-cost 50 --no-gallery` launched 01:32 in background (task `bzds7vqzq`). 45 configs (5 strategies × 9 models — qwen3.6:35b-a3b excluded at launch since pull was in progress). At 02:03 was 63/200 questions into config 1/45. Pace was slow because the 35b-a3b pull was eating bandwidth concurrently; pull finished 02:03 so pace should improve from ~30s/query toward ~7-10s/query (smoke-validated rate).
- **All 10 ALL_MODELS now pulled.** Final addition: qwen3.6:35b-a3b (23 GB). After main Exp 1 finishes, run this to fill in the 5 missing configs: `python scripts/run_experiment_1.py --resume --max-cost 10 --models qwen3.6:35b-a3b`. Then regen gallery.
- **11 prior commits + 1 new commit pushed to origin/main.** Pre-existing 11 commits from last night's session (v3 Ollama judges done) plus tonight's `61bf709 day: disable thinking on Ollama LLM generate calls`. All on origin.
- **NEW BUG FOUND AND FIXED: OllamaLLM.generate was missing `think=False`.** Smoke test hung indefinitely on config 1/4 query 1/10 because qwen3.5:0.8b spent its response budget on hidden chain-of-thought and emitted empty content. Same fix applied to the judge adapter on 2026-05-07 (commit 0024a64) was never applied to the generation path. Fixed in `src/llms/ollama.py` + test updated. This has been latent since qwen3.5 family was added — explains why the loop fix worked on judges but no Exp 1 launch had been attempted on these models.
- **Exp 1 smoke validated** (4 configs × 10 questions, results in `results/experiment_1_smoke/`). EM 50%, BERTScore F1 0.865, both judges agreed (Pearson r=0.640). Loop fix verified — qwen3.5:0.8b ran both strategies before qwen3.5:2b loaded.
- **Network/DNS flakiness during pulls.** Wi-Fi DNS to router (192.168.1.1) drops UDP packets when Ollama pulls saturate bandwidth (~57 Mbps observed). Caused intermittent `getaddrinfo failed` for Python during pulls. Not a bug in this project — just contention. Once pulls finish DNS recovers immediately.
- **`pull_models.py` had a 600s subprocess timeout that's too short for the 23 GB qwen3.6:35b-a3b** (~50 min at 7 MB/s). It marked the model as "failed" but Ollama itself was fine — re-running `ollama pull qwen3.6:35b-a3b` directly succeeded in ~30 min. Either bump the per-model timeout or document as a known limitation.

## 2026-05-08 (night, 5090)
- **v3 Ollama judge run is DONE.** Both `ollama:gemma4:31b` and `ollama:qwen3.5:27b` are 500/500 on `results/experiment_0_v3/raw_scores.csv`. Report.md, gallery (`docs/`), and metadata.json all reflect it. Loop fix held — no more VRAM thrash, model loaded once and stayed resident.
- **Two source-tree fixes shipped on top of the loop fix.** Both were forced by tonight's run:
  - `src/scorers/llm.py` Ollama adapter now sets `num_ctx=8192` — default (32768) put qwen3.5:27b in a 12% CPU / 88% GPU split (22.4/24.4 GB used, 1.7 GB free) and every call 500'd.
  - Same adapter sets `think=False` — qwen3.5:27b's thinking trace blew the 300s read timeout on every row. With it off, ~2s per row.
  - These also benefit gemma4:31b and any future ollama judge. Tests pass.
- **Local main is 10 commits ahead of `origin/main` and not yet pushed.** Network went out around 22:50 — Wi-Fi associated with `192.168.1.17` but gateway and 8.8.8.8 are unreachable. Likely router/ISP outage, not a 5090 issue. Push as soon as network returns: `git push origin main`.
- **Exp 1 prereq: only 3 of the 10 ALL_MODELS are pulled.** Have: `qwen3.5:27b`, `gemma4:26b`, `gemma4:31b` (qwen3.5:27b is a judge, not in ALL_MODELS but on disk). Missing: 4 small Qwens (`qwen3.5:0.8b/2b/4b/9b`), 2 Gemma e-tier (`gemma4:e2b`, `gemma4:e4b`), `qwen3.6:27b`, `mxbai-embed-large` (the embedder). Need network to pull. Until then, **Exp 1 cannot launch** — no embedder = no retrieval.
- **`qwen3.6:35b-a3b` pull was permission-denied tonight** (system flagged "agent-guessed model name" even though it's straight from `ALL_MODELS` in the runbook). Verify the tag on ollama.com when network returns; if real, you may need to retry interactively or whitelist that exact tag.
- **Gallery cost-table bug (small):** the `Estimated Cost Breakdown` table in `report.md` shows `$5.00` per 500 calls for the two Ollama judges. Ollama is free; `generate_gallery.py` defaults the per-call cost without checking the provider. Cosmetic only — the writeup will need a footnote or a tiny fix to `generate_gallery.py`.

