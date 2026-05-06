# 5090 Mobile Migration Runbook

> **Goal:** Get the project running locally on the new RTX 5090 mobile (24GB) so
> Experiments 1 & 2 can be generated for free, faster, and with no rate limits.
>
> **Estimated time:** 2–4 hrs of active setup + 2–6 hrs of model pulls in
> background.
>
> **Daytime instance reads this. Night instance does not.**

---

## Pre-flight checklist

- [ ] `nvidia-smi` prints a table showing the 5090 with CUDA Version 12.8+
- [ ] `python --version` prints 3.11 or 3.12 (project requires 3.11+)
- [ ] At least 150 GB free on the drive that will hold Ollama models — total pull is
      ~80–100 GB and Windows likes headroom.
- [ ] Repo cloned to a path without spaces (Ollama and Python venv are happier).

If `nvidia-smi` fails: NVIDIA driver isn't installed correctly. Reinstall drivers
before continuing — nothing else matters until this works.

---

## Step 1 — Install Ollama for Windows

Download from https://ollama.com/download/windows. Run the installer. After
install, open PowerShell and verify:

```powershell
ollama --version
```

Confirm Ollama can see the GPU:

```powershell
ollama run qwen3.5:0.8b "What is 2 + 2?"
```

This pulls the smallest model (~1 GB) and runs one inference. If it returns "4"
quickly, Ollama is using the GPU. If it takes 30+ seconds for that response, it's
on CPU — see Troubleshooting.

---

## Step 2 — Set up the project venv

```powershell
cd <project_root>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This will install PyTorch among other packages. **If pip fails on torch with a
CUDA-related error**, you may need the nightly wheel for sm_120 (Blackwell):

```powershell
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify torch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True NVIDIA GeForce RTX 5090 Laptop GPU` (or similar). If `False`,
the BGE reranker and cross-encoder filter will fall back to CPU. **That's
acceptable for the deadline** — they're optional pipeline stages and CPU is slow
but functional. Don't get stuck here.

---

## Step 3 — Pull models (run in background while you do other things)

Order matters: pull smallest first so you can smoke-test before the big ones
finish.

```powershell
# Tier 1 — small (smoke test set, ~10 GB total)
ollama pull mxbai-embed-large
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull gemma4:e2b
ollama pull gemma4:e4b

# Tier 2 — medium (~10 GB)
ollama pull qwen3.5:9b

# Tier 3 — large (~50 GB total — takes a while)
ollama pull qwen3.6:27b
ollama pull qwen3.6:35b-a3b
ollama pull gemma4:26b
ollama pull gemma4:31b
```

Run Tier 1 and start the smoke test in another terminal while Tier 2/3 pull.

Disk usage check after all pulls:

```powershell
ollama list
```

Should show 11 models, total roughly 80–100 GB on disk.

---

## Step 4 — Smoke test

```powershell
python scripts/smoke_test.py
```

This script (created in task-010) verifies the project's basic pipeline runs.
If it passes against `localhost:11434` with one of the small models, you're
clear to do a longer dry run.

---

## Step 5 — 10-question dry run of Exp 1

Once Tier 1 models are pulled and the smoke test passes:

```powershell
python scripts/run_experiment_1.py --models qwen3.5:0.8b qwen3.5:2b --strategies naive,self_rag --n 10
```

This runs 2 models × 2 strategies × 10 questions = 40 generations against the
local Ollama. Should finish in 5–10 minutes. Verify:

- [ ] `results/experiment_1/raw_answers.csv` is created with 40 rows.
- [ ] `llm_quantization` column appears (will exist after task-053 lands; before
      then it's expected to be missing — note this in your review).
- [ ] No CUDA OOM errors in the log.
- [ ] Latencies are reasonable (< 30 s per generation for small models).

If anything is wrong, fix it before launching the full 50-config run — a bug
that compounds over 10,000 generations is much more painful than catching it
in 40.

---

## Step 6 — Launch full Exp 1

After all models are pulled and the dry run passes:

```powershell
python scripts/run_experiment_1.py --resume
```

Expected runtime: **30–45 hrs** for 50 configs × 200 questions on the 5090.
Plan to leave the laptop running, plugged in, with sleep disabled.

Disable Windows sleep during the run:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```
(Restore later: replace `0` with your preferred minutes.)

---

## Step 7 — Launch Exp 2

Once Exp 1 finishes:

```powershell
python scripts/run_experiment_2.py --resume
```

Expected runtime: 10–15 hrs (4 chunkers × 4 models × 200 questions, smaller
models only).

---

## Troubleshooting

**Ollama is slow / using CPU:**
- Check `ollama ps` while a model is loaded — it should show GPU memory usage.
  If it shows 0 GB GPU, Ollama isn't using the GPU. Most common cause on Windows
  is an outdated NVIDIA driver — Ollama needs CUDA 12.8+.
- Restart Ollama service: `Restart-Service ollama` in admin PowerShell.

**Out of memory (CUDA OOM) on a large model:**
- 24 GB is comfortable for everything in the matrix at Q4_K_M (Ollama default).
  If a 27B+ model OOMs, your reranker may be loaded at the same time. Set
  `OLLAMA_KEEP_ALIVE=0` to unload models between calls (slower but uses less
  VRAM):
  ```powershell
  $env:OLLAMA_KEEP_ALIVE = "0"
  ```

**PyTorch can't see the GPU but `nvidia-smi` works:**
- You probably installed stable torch instead of nightly. Blackwell needs a
  recent torch wheel. See Step 2.
- This breaks the BGE reranker and cross-encoder filter. **Both are optional
  pipeline stages.** The experiments specify `reranker=None` by default —
  not a blocker for the deadline, just a future TODO.

**Disk fills up mid-pull:**
- Check `ollama list` and remove anything you don't need: `ollama rm <model>`.
- Move Ollama's model directory to a larger drive: set `OLLAMA_MODELS` env var
  before starting Ollama service.

**RunPod fallback:**
- Local migration failing isn't a project killer. RunPod still works — `python
  scripts/run_experiment_1.py --ollama-host https://<pod-id>-11434.proxy.runpod.net`.
  Costs ~$0.27/hr × 30 hrs = ~$8 for Exp 1, which is fine.
