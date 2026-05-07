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
- [ ] `python --version` prints 3.11 or 3.12 (project requires 3.11+ — install from
      python.org or `winget install Python.Python.3.12` if it's older)
- [ ] `git --version` works (install from https://git-scm.com or `winget install Git.Git`)
- [ ] At least 150 GB free on the drive that will hold Ollama models — total pull is
      ~80–100 GB and Windows likes headroom.
- [ ] Project source is present: either a Dropbox sync of the project folder, or a
      `git clone https://github.com/nono638/2026Project.git` to a path without spaces.

If `nvidia-smi` fails: NVIDIA driver isn't installed correctly. Reinstall drivers
before continuing — nothing else matters until this works.

---

## Dropbox sync caveats (read this before working in a synced folder)

If the project is reaching the 5090 via Dropbox sync rather than git clone:

- **Delete `.venv/` if Dropbox copies one over.** Virtual envs contain
  machine-specific binaries from the old laptop and will not work on the 5090.
  You must recreate it locally (Step 2 below).
- **`__pycache__/` directories** will sync uselessly. Harmless, but a clean
  `git clone` would skip them entirely.
- **`.env` file** may or may not sync depending on your Dropbox selective-sync
  settings. Verify it lands on the 5090 — see Step 2.5 below.
- **HuggingFace and Ollama model caches** live in your user profile (not the
  project folder), so they don't sync via Dropbox. They'll repopulate from
  scratch on the 5090.

If in doubt, prefer `git clone` over Dropbox sync — fewer footguns.

---

## Step 0.5 — Configure Git identity

If git was just installed (e.g., as a prerequisite for Claude Code) without
identity configuration, the night instance's commits will fail. Set this once:

```powershell
git config --global user.name "Noah"
git config --global user.email "ncollin1985@yahoo.com"
```

(These match the planning laptop's existing identity. Use the same values so
commit history stays consistent.)

For pushing to GitHub from the 5090: the remote is HTTPS
(`https://github.com/nono638/2026Project.git`). On your first `git push`, Git
Credential Manager will open a browser window for OAuth — approve it once and
credentials are cached. Alternatives if GCM doesn't work: `gh auth login` if
GitHub CLI is installed, or create a Personal Access Token at
https://github.com/settings/tokens and paste it as the password on first push.

You don't need to configure auth before setup — only before the first push back
to GitHub.

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

If a `.venv/` was synced from Dropbox, delete it first — the binaries are
machine-specific and won't work here.

```powershell
cd <project_root>
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` already pins both spaCy models (`en_core_web_sm` and
`en_core_web_lg`) as direct URL deps — no separate `python -m spacy download`
step needed despite what older sections of `ENVIRONMENT.md` may say.

After pip finishes, download the NLTK data that `rake-nltk` needs (one-time):

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### PyTorch GPU verification

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

- Prints `True NVIDIA GeForce RTX 5090 ...` → you're done with PyTorch setup.
- Prints `False`, or errors mentioning `sm_120` / "no kernel image is available
  for execution on the device" → install the nightly wheel for Blackwell:
  ```powershell
  pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
  Then re-run the verification.

**If torch GPU support fails even with nightly:** don't get stuck here. The
experiments still run — Ollama doesn't depend on PyTorch. The only fallout is
that the BGE reranker and cross-encoder filter run on CPU (slow but functional),
and the experiment scripts default to `reranker=None` anyway. Make a note in
inbox.md and move on.

## Step 2.5 — API keys (`.env` file)

The experiments use cloud-based LLM judges, so the 5090 needs API keys. Required:

- `ANTHROPIC_API_KEY` — Claude judges
- `OPENAI_API_KEY` — GPT judges
- `GOOGLE_API_KEY` — Gemini judges + Google text embedder
- `RUNPOD_API_KEY` — only if you want RunPod as a fallback

If a `.env` file synced via Dropbox, verify it's present in the project root and
the keys are populated:

```powershell
Get-Content .env | Select-String -Pattern "API_KEY"
```

If missing or incomplete, copy from `.env.example` and paste keys from your
password manager:

```powershell
Copy-Item .env.example .env
notepad .env
```

---

## Step 3 — Pull models (run in background while you do other things)

### 3.0 Verify all tags exist before pulling

The model matrix was bumped to Qwen 3.5 / 3.6 + Gemma 4 in task-054, but those
10 tags were never confirmed against the Ollama registry from the planning
laptop (sandbox network was blocked). A bad tag fails with a confusing 404 in
the middle of an 80 GB pull. Verify first:

```powershell
foreach ($tag in @(
    "qwen3.5:0.8b","qwen3.5:2b","qwen3.5:4b","qwen3.5:9b",
    "qwen3.6:27b","qwen3.6:35b-a3b",
    "gemma4:e2b","gemma4:e4b","gemma4:26b","gemma4:31b",
    "mxbai-embed-large"
)) {
    Write-Host "Checking $tag..." -NoNewline
    $r = Invoke-WebRequest "https://ollama.com/library/$($tag.Split(':')[0])" -UseBasicParsing -Method Head -ErrorAction SilentlyContinue
    if ($r.StatusCode -eq 200) { Write-Host " family OK" } else { Write-Host " MISSING — investigate" }
}
```

(Family-level check; does not validate the exact `:tag` suffix. For a per-tag
check, run `ollama show <tag>` after Step 1's install and see if it pulls
metadata in <2s, vs. starting a download.) If any 404, search ollama.com for
the actual published tag and update `scripts/pull_models.py` REQUIRED_MODELS,
`run_experiment_1.py` ALL_MODELS, `run_experiment_2.py` ALL_MODELS, and
`deploy/setup_pod.py` DEFAULT_MODELS in lockstep before continuing.

### 3.1 Pull tiers

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

## Step 4.5 — Verify Ollama metadata helper (task-053)

task-053 added `get_ollama_model_details()` which calls `/api/show` to capture
the resolved quantization level on every CSV row. The helper has unit tests
against mocks but was never exercised against a live Ollama. Before launching
Exp 1/2 (which depend on it for `llm_quantization` column), confirm it works:

```powershell
python -c "from scripts.experiment_utils import get_ollama_model_details; import json; print(json.dumps(get_ollama_model_details('qwen3.5:4b'), indent=2))"
```

Expected output: a dict with `quantization_level` (e.g., `"Q4_K_M"`),
`digest`, and other model details. If you get an empty dict or an error,
fix before proceeding — every CSV row will otherwise stamp `"unknown"`.

---

## Step 4.6 — Backfill quant metadata for v1/v2/v3 (task-053)

task-053 also added `scripts/backfill_quant_metadata.py` to retroactively
annotate Exp 0 v1/v2/v3 (which were generated before the helper existed). One
shot — Ollama must be running with `qwen3:4b` (the v1/v2/v3 generation model)
loadable:

```powershell
ollama pull qwen3:4b   # if not already pulled — v1/v2/v3 used the older tag
python scripts/backfill_quant_metadata.py
```

After it finishes, the gallery will show the actual quant for the v1/v2/v3
runs instead of `unknown`. Optional but cheap, and the writeup is cleaner
with real values.

---

## Step 5 — 10-question dry run of Exp 1

Once Tier 1 models are pulled and the smoke test passes:

```powershell
python scripts/run_experiment_1.py --models qwen3.5:0.8b qwen3.5:2b --strategies naive,self_rag --n 10
```

This runs 2 models × 2 strategies × 10 questions = 40 generations against the
local Ollama. Should finish in 5–10 minutes. Verify:

- [ ] `results/experiment_1/raw_answers.csv` is created with 40 rows.
- [ ] `llm_quantization` column appears with a real value like `Q4_K_M` (not
      `unknown`). If `unknown` shows up everywhere, the `/api/show` call from
      `get_ollama_model_details()` is failing silently — go back to Step 4.5.
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

## Step 8 — Retroactive Ollama-judge scoring on Exp 0 v3 (task-055)

This validates whether large local LLMs are viable scorer candidates by
re-scoring v3's existing 500 generations with `gemma4:31b` and `qwen3.5:27b`,
then reading the new rows in v3 report.md's "Correlation with Gold Metrics"
table. If either local judge correlates with BERT/F1 on par with the API
judges, future panels can adopt it for free.

This is **not on the critical path** for Exp 1/2. Best run during a quiet
window — model-pull idle time, overnight after Exp 1 launches, or in parallel
with Exp 2 (one judge model fits in <24 GB, can co-exist with Exp 2's small
Qwens at separate VRAM slots if scheduled carefully).

```powershell
# Make sure both judge models are pulled (Step 3 already covers this if
# Tier 3 finished)
ollama pull gemma4:31b
ollama pull qwen3.5:27b

# OLLAMA_HOST default (http://localhost:11434) is correct on the 5090 — no
# action needed unless Ollama is on a different host. The Ollama scorer
# adapter reads OLLAMA_HOST from env; bare hosts get an http:// prefix
# auto-prepended.

python scripts/run_experiment_0.py `
  --output-dir results/experiment_0_v3 `
  --skip-generation `
  --judges qwen3.5:27b gemma4:31b
```

Note on the `--judges` filter: it does substring match against the
**model name**, not the provider, so `--judges ollama` doesn't work.
Pass the full ollama tags (or unique substrings like `qwen3.5 gemma4`)
to select just those two judges.

Expected runtime: ~4 hrs total (500 rows × 2 judges, ~10 s per call on a
31B at Q4_K_M). CostGuard ceiling stays at default $5 — Ollama is free.

After it finishes, regenerate the gallery so the new judges show up:

```powershell
python scripts/generate_gallery.py
```

The `## Correlation with Gold Metrics` table in
`results/experiment_0_v3/report.md` will gain two rows — `ollama:gemma4:31b`
and `ollama:qwen3.5:27b`. That's the answer to the validation question.

---

## Optional (nice-to-have)

- **Claude Code CLI** — if you want me available on the 5090 for live troubleshooting
  during setup. `winget install Anthropic.Claude` or follow the install steps at
  https://claude.ai/code. Note: a fresh Claude Code install on the 5090 starts with
  empty memory; this runbook (read from the synced project folder) is your handoff.
- **VS Code** — if your editor isn't already there. `winget install Microsoft.VisualStudioCode`.
- **PowerShell 7** — Windows 11 ships with PowerShell 5.1; commands in this runbook
  work on either, but PS 7 is nicer. `winget install Microsoft.PowerShell`.

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
