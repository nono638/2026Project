# Known Issues

> **Daytime only** — night instance does not read this file.
>
> Bugs and limitations documented for awareness.

---

## Experiment 0v2 scoring process silent deaths (2026-03-25)
Scoring process died silently twice before incremental checkpointing was added. Possible causes: Windows process timeout, network drop, or silent exception in LLMScorer. Checkpoint fix prevents data loss but root cause unknown. Incremental checkpointing now mitigates this.

## bert_score module not installed (2026-03-26)
13 pre-existing test failures due to missing bert_score package. Not blocking experiments but should be installed for full test suite green.

## 5090 BSOD #2 — SYSTEM_SERVICE_EXCEPTION during Exp 1 resume (2026-05-18)
After the 5090 driver clean reinstall on 2026-05-17 (in response to a `0x133` DPC_WATCHDOG_VIOLATION at 22:32 PM), Experiment 1 resumed at 01:42 AM and crashed again at ~02:14 AM with bugcheck `0x0000003b` SYSTEM_SERVICE_EXCEPTION (`c0000005` access violation). Different bugcheck, same general class — NVIDIA driver under sustained interrupt/allocation pressure.

Root cause hypothesis (code review): the pipeline is fully **serial**, so this isn't a Python-side concurrency problem. The stressor is **Ollama swapping the embedder and chat model in/out of VRAM on every query iteration**. For each row Ollama gets: embed → chat → embed → chat → ... at ~1 s cadence, and without `keep_alive` on the embedder, Ollama's 5 m default lets the embedder get evicted between calls. Constant VRAM allocation churn is the known trigger pattern for the 5090 laptop driver's DPC bug.

Fortifications applied 2026-05-18 (uncommitted; ready for next run):
- `OllamaEmbedder` now passes `keep_alive=30m` on every embed call (matches `OllamaLLM`).
- `run_experiment_1.py`: `CONFIG_COOLDOWN_S` bumped 5 s → 30 s.
- New `--row-pace-s` flag (default 0.5 s sleep between completed rows) + a 10 s rest every 50 rows; both logged in `events.jsonl` as `row_rest`.
- `resume_experiment_1.bat` documents the Ollama env vars to set on the *server* (`OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_QUEUE=1`) if BSODs recur — must be set before `ollama serve`, not in the bat itself.

Resume state at restart: 14 configs complete at 200/200 rows; `adaptive × qwen3.5:4b` at 190/200; 15 configs untouched (the qwen3.5:9b row and both gemma4 columns).

## 5090 BSOD #3 — model runner crash during qwen3.5:9b first query (2026-05-18)

After BSOD #2 + driver reinstall, Experiment 1 resumed at 08:46 AM. Completed adaptive × qwen3.5:4b (10 remaining rows, 43 min — anomalously slow). Config 16 (naive × qwen3.5:9b) started at 09:31:18 and crashed the laptop before query 3 completed.

**Log evidence:** `adaptive × qwen3.5:4b` query 195 hung for 41 minutes before Ollama returned a 500, suggesting GPU flapping already started on the 4b model. For `qwen3.5:9b` query 3, Ollama returned 500 errors after 8 min (retry 1), 17 min (retry 2), and crashed the laptop partway through retry 3. Normal generation time for these models: 30–90 s.

**Root causes (two confirmed, one mitigating):**

1. **No generation timeout.** `OllamaLLM` used `timeout=None` (Ollama Python Client default). When Ollama's model runner crashed mid-generation, the `chat()` call hung forever. Three retries each re-hung. The laptop BSODed during the 4th attempt.

2. **No `num_predict` cap.** Without `options={"num_predict": N}`, Ollama lets models generate indefinitely. Quantized 9B models may generate extremely long sequences on some inputs before the runner crashes, sustaining GPU bandwidth pressure and driver DPC stress.

3. **Confirmed Ollama Blackwell CUDA bug (GitHub #14374):** Filed 2026-02-23, confirmed as an nvcc `-O3` compiler bug producing invalid machine code in the MMQ (quantized matrix multiplication) kernel for Blackwell SM_120. Affects Q4_K_M and other quantized formats. The 4B model with Q4_K_M likely hit this intermittently (explaining the 41-min hang); the 9B model hit it more reliably due to more MMQ kernel invocations per forward pass. Ollama 0.24.0 installed — unknown whether this build has the fix (need to check if #14374 is closed).

4. **GPU detection flapping (GitHub #13338):** On Windows, Ollama 0.13.1+ intermittently misdetects the RTX 5090 as having "0 B VRAM" and falls back to CPU. CPU inference for a 9B model takes hours not seconds, which matches the 8-17 min generation times observed.

**Fortifications applied 2026-05-18:**
- `OllamaLLM`: `DEFAULT_TIMEOUT = 180.0 s` + `DEFAULT_NUM_PREDICT = 512` tokens cap
- `OllamaEmbedder`: `DEFAULT_TIMEOUT = 60.0 s`
- `experiment_utils.py`: runner crash detected by message pattern → 60 s recovery wait + Ollama health poll before retry (instead of 2–8 s backoff that re-hangs into a dead runner)
- `resume_experiment_1.bat`: expanded setup instructions for Ollama env vars

**What to try next if BSODs continue:**
- Check whether Ollama issue #14374 is fixed in current build (look for "CUDA error: device kernel image is invalid" in Ollama logs)
- Try `qwen3.5:9b-q8_0` instead of `qwen3.5:9b` (Q8_0 doesn't use the MMQ kernel path that has the Blackwell bug; uses cuBLAS instead)
- Set `OLLAMA_FLASH_ATTENTION=0` before `ollama serve` to disable flash attention kernel (another Blackwell compatibility point)
- WSL2 has substantially better RTX 5090 support per the issue thread — fallback option if native Windows remains unstable

**Resume state:** 15 configs done at 200/200 (all Qwen3.5 0.8b/2b/4b × all strategies). 15 configs remain: qwen3.5:9b × 5 strategies + gemma4:e2b × 5 + gemma4:e4b × 5.

**Methodological caveat introduced by num_predict cap (write up in paper):**
The 15 already-completed configs were generated with **no** `num_predict` cap; the 15 remaining configs are generated with `num_predict=1024` (initially 512 for `naive × qwen3.5:9b` only, which finished first). This is a methodological asymmetry — answers in the new configs may be clipped at ~4000 chars while the older configs allowed arbitrary length.

Per-strategy clip impact estimate (from completed data >2048 char, equivalent to ~512 tokens; scale roughly halved for 1024 tokens):
- naive, multi_query, corrective: <2% answers clipped — negligible
- adaptive: 1-7% — small
- self_rag: 2-12% — modest, **mainly clips runaway/repetitive outputs that were arguably noise**

Some clipped outputs in the previously-completed data exceeded 100,000 chars (max observed: 1.2M chars for `qwen3.5:4b × self_rag`). These were degenerate runaway generations. The cap arguably *removes noise* from those cases, but the asymmetric application across model sizes weakens the strict "no controlled variable changed" claim. Three honest framings for the writeup: (a) report results as-is with this caveat, (b) re-run completed configs with the same cap for full consistency (3000 rows, ~3 hours), or (c) post-hoc truncate all completed answers to 4096 chars before scoring and re-score.

## Gemini 3.1 Pro Preview is paid-tier-only (2026-04-30)
`google:gemini-3.1-pro-preview` is in `JUDGE_CONFIGS` but produces zero scored rows on every run. Confirmed via direct API smoke test: Google AI Studio free tier has `limit: 0` for this model — it requires paid billing. Other Gemini models (Flash-Lite, Flash, 2.5 Pro) are unaffected. Left in configs intentionally (option C) so it auto-activates if billing is enabled later; runner already skips it gracefully on 429. Reference: `gemini-api-billing-setup.md` for paid-tier setup notes if ever needed.
