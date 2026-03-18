# RunPod Setup Guide — From Zero to Experiment 0 Results

This guide walks you through everything from creating a RunPod account to having
Experiment 0 results on your screen. No prior cloud GPU experience required.

**Time estimate:** ~30 minutes of your time (plus ~40-60 minutes of automated experiment
runtime). **Cost estimate:** ~$1.80 total ($0.15 GPU + $1.60 LLM scoring APIs).

---

## Table of Contents

### Part 1: RunPod Account Setup
1. [Create a RunPod Account](#1-create-a-runpod-account)
2. [Add Funds to Your Account](#2-add-funds-to-your-account)
3. [Create an API Key](#3-create-an-api-key)
4. [Save All Your API Keys](#4-save-all-your-api-keys)
5. [Test the RunPod API Key](#5-test-the-runpod-api-key)

### Part 2: Launch a GPU Pod with Ollama
6. [Create a Pod with Ollama](#6-create-a-pod-with-ollama)
7. [Pull the AI Models](#7-pull-the-ai-models)
8. [Verify Ollama Is Working](#8-verify-ollama-is-working)

### Part 3: Run Experiment 0
9. [Run the Experiment](#9-run-the-experiment)
10. [Read Your Results](#10-read-your-results)
11. [Shut Everything Down](#11-shut-everything-down)

### Reference
12. [Understand the Costs](#12-understand-the-costs)
13. [Important Safety Rules](#13-important-safety-rules)
14. [Troubleshooting](#14-troubleshooting)
15. [Quick Reference](#15-quick-reference)

---

## 1. Create a RunPod Account

1. Go to **https://www.runpod.io** in your browser.
2. Click **Sign Up** (top right).
3. You can sign up with:
   - **Email + password** (simplest)
   - **Google account**
   - **GitHub account**
4. Verify your email if prompted.
5. You'll land on the RunPod console at `console.runpod.io`.

That's it — account is created. No credit card required just to create the account.

---

## 2. Add Funds to Your Account

RunPod uses a **prepaid credit system**. You load money into your account first, then
it deducts as you use GPU time. Nothing runs without credits, so there's no risk of a
surprise bill.

1. In the RunPod console, click your **profile icon** (top right) → **Billing**.
   - Or go directly to: **https://console.runpod.io/user/billing**
2. Click **Add Funds** (or **Add Credits**).
3. Choose your payment method:
   - **Credit card** — processed through Stripe. Standard Visa/Mastercard/Amex.
   - **Debit card** — works like a credit card.
   - **Prepaid card** — works, BUT you must deposit **at least $100** per transaction
     (Stripe's requirement for prepaid cards).
   - **Cryptocurrency** — available if you prefer.
4. Enter the amount:
   - **Minimum deposit: $10.** This is enough to try things out.
   - **Recommended for RAGBench: $25.** This gives you roughly:
     - ~150 hours on an RTX A4000 ($0.17/hr)
     - ~70 hours on an RTX 4090 ($0.34/hr)
     - Plenty for Experiments 0, 1, and 2 plus demo time
   - You can always add more later.
5. Complete the payment.
6. Your balance should appear in the console within a few seconds.

**Your credits are non-refundable** — you can't withdraw unused balance. Start small.

### How to check your balance

- **Console:** Your balance shows in the top bar or under Billing.
- **API:** Our code can check it too (see Step 7).

---

## 3. Create an API Key

The API key lets our `RunPodManager` code create and manage pods programmatically
(instead of clicking through the web UI every time).

1. In the RunPod console, go to **Settings** → **API Keys**.
   - Or go directly to: **https://console.runpod.io/user/settings**
2. Click **Create API Key**.
3. Give it a name you'll recognize, like `ragbench`.
4. For permissions, choose **All** (full access).
   - We need this because our code creates pods, terminates pods, and queries balance.
   - "Restricted" would work too, but you'd need to manually enable each permission.
     "All" is simpler for a personal project.
5. Click **Generate** (or **Create**).
6. **IMPORTANT: Copy the API key NOW.** RunPod does NOT store it. Once you close this
   dialog, you can never see the key again. If you lose it, you'll have to delete it
   and create a new one.
7. Paste it somewhere safe temporarily (a text file, password manager, etc.).

Your API key looks something like: `rpa_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890` (a long
string starting with `rpa_` or similar).

---

## 4. Save All Your API Keys

Experiment 0 needs **three** API keys. You already have the RunPod key from Step 3.
The other two are for the LLM judges that score the RAG answers.

### 4a. Get your Anthropic API key (for Claude judges)

1. Go to **https://console.anthropic.com/**
2. Sign up or log in.
3. Go to **Settings** → **API Keys** → **Create Key**.
4. Copy the key. It looks like: `sk-ant-api03-...`

If you already have an Anthropic key (e.g., from using Claude API before), use that one.

### 4b. Get your Google AI Studio API key (for Gemini judges)

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with your Google account.
3. Click **Create API key** → select a project (or create one).
4. Copy the key. It looks like: `AIzaSy...`

Google AI Studio is free for moderate usage — the Gemini calls in Experiment 0 cost
fractions of a penny.

### 4c. Save all three keys in a `.env` file

In the RAGBench project root, create (or edit) a file called `.env`:

```
RUNPOD_API_KEY=rpa_YOUR_RUNPOD_KEY_HERE
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_ANTHROPIC_KEY_HERE
GOOGLE_API_KEY=AIzaSy_YOUR_GOOGLE_KEY_HERE
```

Replace each placeholder with your real key.

**CRITICAL:** This file must NOT be committed to git. Verify:
1. Run `git status` in the project directory.
2. If `.env` shows up as untracked, add it to `.gitignore` before doing anything else:
   ```bash
   echo ".env" >> .gitignore
   ```

### Alternative: System environment variables

If you prefer not to use a `.env` file, set them in your terminal:

**Windows (PowerShell):**
```powershell
$env:RUNPOD_API_KEY = "rpa_YOUR_KEY"
$env:ANTHROPIC_API_KEY = "sk-ant-api03-YOUR_KEY"
$env:GOOGLE_API_KEY = "AIzaSy_YOUR_KEY"
```
(This only lasts for the current terminal session.)

**Windows (permanent):**
Start → search "environment variables" → "Edit environment variables for your account"
→ add all three as new User variables. Restart terminals afterward.

**Linux/Mac:**
```bash
export RUNPOD_API_KEY="rpa_YOUR_KEY"
export ANTHROPIC_API_KEY="sk-ant-api03-YOUR_KEY"
export GOOGLE_API_KEY="AIzaSy_YOUR_KEY"
```
Add to `~/.bashrc` or `~/.zshrc` to make permanent.

---

## 5. Test the RunPod API Key

Let's verify your RunPod key works before spending any money. Open a terminal in the
project directory and run:

```bash
# Activate your virtual environment first
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Then test the key:
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from deploy.runpod_manager import RunPodManager

api_key = os.environ.get('RUNPOD_API_KEY')
if not api_key:
    print('ERROR: RUNPOD_API_KEY not set. See Step 4.')
else:
    mgr = RunPodManager(api_key=api_key)
    balance = mgr.get_balance()
    print(f'SUCCESS! Account balance: \${balance:.2f}')
"
```

**Expected output:** `SUCCESS! Account balance: $25.00` (or whatever you deposited).

**If you get an error:**
- `RUNPOD_API_KEY not set` → Go back to Step 4. Make sure `.env` is in the project root.
- `RunPodError` or `401` → Your API key is wrong. Go back to Step 3 and create a new one.
- `ModuleNotFoundError: requests` → Run `pip install -r requirements.txt` first.
- `ModuleNotFoundError: dotenv` → Run `pip install python-dotenv` first.

---

---

# Part 2: Launch a GPU Pod with Ollama

---

## 6. Create a Pod with Ollama

This creates a GPU machine in the cloud with Ollama (the AI model server) pre-installed.
Our `deploy/setup_pod.py` script handles everything automatically.

**This step costs real money** — roughly $0.01-0.03 for setup time.

```bash
python deploy/setup_pod.py
```

**Expected output:**
```
Checking balance... $25.00
Creating pod 'ragbench-gpu' with Ollama...
Pod created: abc123xyz
Waiting for pod to be ready (up to 5 min)...
Pod is ready!
Ollama URL: https://abc123xyz-11434.proxy.runpod.net
Pulling models (this takes 3-5 minutes)...
  Pulling mxbai-embed-large... done
  Pulling qwen3:4b... done
Verifying Ollama is serving...
  mxbai-embed-large: OK
  qwen3:4b: OK

========================================
  Pod is ready for experiments!
  Pod ID:     abc123xyz
  Ollama URL: https://abc123xyz-11434.proxy.runpod.net
  Spend rate: $0.17/hr
  Balance:    $24.95
========================================

Save this URL — you'll need it for Step 9.
```

**Copy the Ollama URL** from the output. It looks like:
`https://abc123xyz-11434.proxy.runpod.net`

You'll paste this into the experiment command in Step 9.

### What just happened?

The script:
1. Created a GPU pod using the `ollama/ollama` Docker image (Ollama pre-installed)
2. Waited for it to boot up (~1-2 minutes)
3. Downloaded two AI models onto the pod (~3-5 minutes):
   - `mxbai-embed-large` — the embedding model (converts text to vectors)
   - `qwen3:4b` — the language model (generates answers)
4. Verified both models respond correctly

### If it fails

- **"No GPUs available"** — all A4000s are sold out. Wait 10 minutes and retry, or the
  script will fall back to RTX 3090 or RTX 4090 automatically.
- **"Timeout waiting for pod"** — the pod is slow to start. Check
  https://console.runpod.io/pods — if it shows "Pending", just wait.
- **"Model pull failed"** — the pod is running but model download failed. Re-run the
  script (it's safe to run multiple times).

---

## 7. Pull the AI Models

**This step is handled automatically by Step 6.** The `setup_pod.py` script pulls both
models for you.

If you ever need to pull additional models (e.g., for Experiments 1 & 2), you can use:

```bash
python deploy/setup_pod.py --pod-id abc123xyz --pull-only --models qwen3:0.6b qwen3:1.7b qwen3:8b
```

(Replace `abc123xyz` with your actual pod ID from Step 6.)

### Model sizes (for reference)

| Model | Size | VRAM needed | Used in |
|-------|------|-------------|---------|
| mxbai-embed-large | ~670 MB | ~1 GB | All experiments (embeddings) |
| qwen3:4b | ~2.5 GB | ~4 GB | Experiment 0 |
| qwen3:0.6b | ~400 MB | ~1 GB | Experiments 1 & 2 |
| qwen3:1.7b | ~1 GB | ~2 GB | Experiments 1 & 2 |
| qwen3:8b | ~5 GB | ~6 GB | Experiments 1 & 2 |
| gemma3:1b | ~800 MB | ~1.5 GB | Experiment 1 |
| gemma3:4b | ~3 GB | ~4 GB | Experiment 1 |

---

## 8. Verify Ollama Is Working

**This step is also handled automatically by Step 6.** But if you want to manually check
that the pod is still responding (e.g., before starting a long experiment):

```bash
python -c "
url = 'YOUR_OLLAMA_URL_HERE'  # paste your URL from Step 6
import requests
resp = requests.get(f'{url}/api/tags')
if resp.ok:
    models = [m['name'] for m in resp.json().get('models', [])]
    print(f'Ollama is running. Models available: {models}')
else:
    print(f'ERROR: Ollama not responding (status {resp.status_code})')
"
```

Replace `YOUR_OLLAMA_URL_HERE` with the URL from Step 6 (e.g.,
`https://abc123xyz-11434.proxy.runpod.net`).

**Expected output:**
```
Ollama is running. Models available: ['mxbai-embed-large:latest', 'qwen3:4b']
```

---

# Part 3: Run Experiment 0

---

## 9. Run the Experiment

Make sure your `.env` file has all three keys (Step 4) and your pod is running (Step 6).

```bash
python scripts/run_experiment_0.py --ollama-host https://abc123xyz-11434.proxy.runpod.net
```

Replace `abc123xyz` with your actual pod ID.

**What happens:**
1. **Downloads 50 HotpotQA examples** (~30 seconds, downloads from HuggingFace)
2. **Generates 50 RAG answers** using Qwen3-4B on your RunPod GPU (~15-20 minutes)
   - Progress shows as `[1/50]`, `[2/50]`, etc.
   - Each query: chunks the document, embeds it, retrieves relevant chunks, generates answer
3. **Scores all 50 answers with 5 judges** (~20-40 minutes)
   - Gemini Flash, Gemini Pro, Claude Haiku, Claude Sonnet, Claude Opus
   - Each judge rates faithfulness, relevance, and conciseness (1-5)
   - Progress shows as each judge starts/finishes
4. **Writes results** to `results/experiment_0/`

**Total time: ~40-60 minutes.** You can walk away and come back.

### Quick test first (optional)

To verify everything works before the full run, do a small test with 5 examples:

```bash
python scripts/run_experiment_0.py --n 5 --model qwen3:4b --ollama-host https://abc123xyz-11434.proxy.runpod.net
```

This takes ~5 minutes and costs ~$0.10. If it works, run the full experiment.

### If generation fails partway through

The script saves raw answers to `results/experiment_0/raw_answers.csv`. If scoring
fails after generation succeeds, you can re-run just the scoring (without re-generating
answers) using:

```bash
python scripts/run_experiment_0.py --skip-generation --ollama-host https://abc123xyz-11434.proxy.runpod.net
```

---

## 10. Read Your Results

After the experiment completes, you'll find three files in `results/experiment_0/`:

| File | What it is |
|------|------------|
| `report.md` | Human-readable summary with tables and recommendations |
| `raw_scores.csv` | Every score from every judge (the raw data) |
| `raw_answers.csv` | The 50 generated answers (so you can re-score without re-generating) |

**Open `results/experiment_0/report.md`** — this is the main output. It contains:

1. **Per-Judge Mean Scores** — average faithfulness/relevance/conciseness/quality per judge
2. **Inter-Scorer Correlation** — do cheap judges agree with expensive ones? (Pearson r)
3. **Correlation with Gold F1** — which judge best predicts actual correctness?
4. **Cost Breakdown** — how much each judge costs per call and total
5. **Recommendation** — if cheap judges correlate highly with expensive ones, use the
   cheap one for Experiments 1 & 2

### What you're looking for

- **High correlation (r > 0.8) between Gemini Flash and Claude Opus** means the cheap
  judge is reliable → use Gemini Flash for Experiments 1 & 2 and save ~$50.
- **Low correlation (r < 0.6)** means cheap judges disagree with expensive ones →
  investigate which one best predicts gold F1 and use that.
- **Gold F1 correlation** tells you which judge most accurately predicts factual
  correctness (the "ground truth").

---

## 11. Shut Everything Down

**Do this as soon as the experiment finishes.** Every minute the pod runs costs money.

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from deploy.runpod_manager import RunPodManager

mgr = RunPodManager(api_key=os.environ['RUNPOD_API_KEY'])

# List running pods
pods = mgr.list_pods()
for pod in pods:
    pod_id = pod.get('id', 'unknown')
    name = pod.get('name', 'unnamed')
    print(f'Terminating pod {pod_id} ({name})...')
    mgr.terminate_pod(pod_id)

balance = mgr.get_balance()
print(f'All pods terminated. Remaining balance: \${balance:.2f}')
"
```

**Or** just go to https://console.runpod.io/pods and click **Terminate** on your pod.

**Double-check:** go to https://console.runpod.io/pods and confirm nothing is running.
Bookmark this page.

---

# Reference

---

## 12. Understand the Costs

### GPU pricing (approximate, Community Cloud)

| GPU | VRAM | ~Cost/hr | Good for |
|-----|------|----------|----------|
| RTX A4000 | 16 GB | $0.17 | Cheapest. Runs all Qwen3 models up to 8B. |
| RTX 3090 | 24 GB | $0.22 | Good fallback. More VRAM headroom. |
| RTX 4090 | 24 GB | $0.34 | Fastest consumer GPU. Widely available. |
| L4 | 24 GB | $0.24 | Data center GPU, very efficient. |
| A40 | 48 GB | $0.39 | Overkill for our models, but good for large batches. |

### Storage pricing

| Type | Cost |
|------|------|
| Volume disk (while pod runs) | $0.10/GB/month |
| Volume disk (pod stopped) | $0.20/GB/month |
| 20 GB volume for one month | ~$2-4 |

### What our experiments will cost (rough estimates)

| Experiment | ~Time | ~GPU cost | ~API scoring cost | ~Total |
|------------|-------|-----------|-------------------|--------|
| Exp 0: Scorer validation (50 queries × 5 judges) | ~1 hr | ~$0.15 | ~$1.60 | **~$1.80** |
| Exp 1: Strategy × Model (30 configs × 100 queries) | ~6-10 hrs | ~$1-2 | TBD* | **~$2-4** |
| Exp 2: Chunking × Model (16 configs × 100 queries) | ~3-5 hrs | ~$0.50-1 | TBD* | **~$1-3** |
| **Total estimate** | | | | **~$5-9** |

*Exp 1 & 2 scoring cost depends on which judge Exp 0 recommends. If Gemini Flash works
(~$0.0001/call), scoring is basically free. If we need Claude Opus (~$0.025/call),
scoring alone costs ~$75 for 3,000 configs. This is exactly why we run Exp 0 first.

$25 in RunPod credits is comfortable for GPU time. API scoring costs depend on the
Exp 0 outcome — budget accordingly.

### Auto-shutdown safety net

RunPod automatically shuts down your pod when your projected balance would cover less
than 10 minutes of runtime. So you literally cannot run out of money without warning.

---

## 13. Important Safety Rules

1. **Always terminate pods when you're done.** A forgotten RTX 4090 pod costs $8/day.
   Our code uses terminate+recreate (not stop+resume) to avoid this.

2. **Check your balance before starting experiments.** Use `mgr.get_balance()` or check
   the console.

3. **Never commit your API key.** The `.env` file must be in `.gitignore`. If you
   accidentally commit it, immediately revoke the key in RunPod console → Settings →
   API Keys and create a new one.

4. **Set a spend limit.** RunPod has a default $80/hr spend limit for new accounts.
   This is fine for us (we'll never hit it with one pod), but you can request a lower
   limit by contacting support if you want extra safety.

5. **Community Cloud is fine.** It's cheaper than Secure Cloud. The only tradeoff is
   slightly less reliability — a community provider could go offline. For experiments
   that save intermediate results, this is an acceptable risk.

6. **Bookmark the pods page:** https://console.runpod.io/pods — so you can always
   quickly check if anything is running.

---

## 14. Troubleshooting

### "No GPUs available"
Some GPU types sell out. Our code has a fallback list (A4000 → 3090 → 4090). If all
three are sold out, wait 10-30 minutes and try again. You can also check availability
on the deploy page: https://console.runpod.io/pods

### "401 Unauthorized"
Your API key is wrong or was revoked. Create a new one (Step 3).

### "Insufficient funds"
Add more credits (Step 2). Remember, minimum deposit is $10 ($100 for prepaid cards).

### Pod stuck on "Pending"
Sometimes a pod takes a few minutes to find a machine. If it's stuck for more than
10 minutes, terminate it and try a different GPU type.

### "Connection refused" when accessing pod URL
The pod is running but the service (Ollama/FastAPI) hasn't started yet. Wait 30-60
seconds after the pod shows "Running" for services to initialize.

### Can't find my pod
Go to https://console.runpod.io/pods — all your pods (running, stopped, pending) are
listed here.

### I accidentally left a pod running
Go to https://console.runpod.io/pods and click **Terminate** on any running pod.
Check your balance under Billing to see how much was spent.

---

## 15. Quick Reference

| What | Where |
|------|-------|
| Sign up | https://www.runpod.io → Sign Up |
| Console (dashboard) | https://console.runpod.io |
| Add funds | https://console.runpod.io/user/billing |
| Create/manage API keys | https://console.runpod.io/user/settings |
| View running pods | https://console.runpod.io/pods |
| GPU pricing | https://www.runpod.io/pricing |
| API docs (REST) | https://docs.runpod.io/api-reference/pods |
| API docs (GraphQL) | https://docs.runpod.io/sdks/graphql/manage-pods |
| Our code | `deploy/runpod_manager.py` |
| Our tests | `tests/test_runpod_manager.py` |
