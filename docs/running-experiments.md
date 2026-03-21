# Running Experiments

## Prerequisites

- Ollama running locally or on a remote server (RunPod)
- API key for scoring in `.env` (GEMINI_API_KEY for default scorer)
- Required models pulled (`python scripts/pull_models.py`)

## Experiment 0: Scorer Validation

**Purpose:** Validate that LLM-as-judge scoring is reliable before trusting it on
thousands of runs.

**What it does:** Takes 50 HotpotQA examples, generates answers with Qwen3-4B + NaiveRAG,
then scores each answer with multiple LLM judges. Compares judge scores against gold
metrics (BERTScore, F1) and inter-judge agreement.

**Run:**
```bash
python scripts/run_experiment_0.py
```

**Options:**
- `--n 50` — Number of HotpotQA examples (default: 50)
- `--skip-generation` — Re-score existing answers without regenerating
- `--judges flash,haiku` — Run specific judges only (substring match)
- `--ollama-host URL` — Remote Ollama server
- `--max-cost 5.0` — API spend ceiling in USD

**Output:** `results/experiment_0/raw_scores.csv`, `results/experiment_0/report.md`

**Expected duration:** ~30 min for generation, ~5 min per judge for scoring.

## Experiment 1: Strategy x Model Size

**Purpose:** Test whether a smart RAG strategy on a small model can outperform a naive
strategy on a large model.

**Matrix:** 5 strategies x 6 models = 30 configurations, each on 200 HotpotQA examples
= 6,000 total runs.

**Held constant:** RecursiveChunker(500, 100), mxbai-embed-large, hybrid retrieval.

**Run:**
```bash
# Full run (several hours on GPU)
python scripts/run_experiment_1.py --ollama-host http://YOUR_RUNPOD_IP:11434

# Resume after interruption
python scripts/run_experiment_1.py --resume --ollama-host http://YOUR_RUNPOD_IP:11434

# Test with subset
python scripts/run_experiment_1.py --models qwen3:4b --strategies naive,self_rag
```

**Options:**
- `--models MODEL[,MODEL]` — Subset of models to run
- `--strategies STR[,STR]` — Subset of strategies to run
- `--resume` — Skip configs already in raw_scores.csv
- `--skip-generation` — Re-score existing answers
- `--scorer provider:model` — Scorer (default: google:gemini-2.5-flash)
- `--ollama-host URL` — Remote Ollama server
- `--max-cost 10.0` — API spend ceiling

**Output:** `results/experiment_1/raw_scores.csv`, `results/experiment_1/report.md`

**Expected duration:** ~6-8 hours on RunPod A5000 for full matrix.

## Experiment 2: Chunking x Model Size

**Purpose:** Measure how much chunking strategy affects RAG quality across model sizes.

**Matrix:** 4 chunkers x 4 Qwen3 models = 16 configurations, each on 200 HotpotQA
examples = 3,200 total runs.

**Held constant:** NaiveRAG strategy, mxbai-embed-large, hybrid retrieval.

**Run:**
```bash
python scripts/run_experiment_2.py --ollama-host http://YOUR_RUNPOD_IP:11434
python scripts/run_experiment_2.py --resume --ollama-host http://YOUR_RUNPOD_IP:11434
```

**Options:** Same as Experiment 1 but `--chunkers` instead of `--strategies`.

**Output:** `results/experiment_2/raw_scores.csv`, `results/experiment_2/report.md`

## Running on RunPod

For experiments that need a GPU:

1. Create a pod: `python deploy/setup_pod.py`
2. Note the pod IP from the output
3. Run experiments with `--ollama-host http://POD_IP:11434`
4. When done, terminate the pod from the RunPod dashboard

The pod runs Ollama and automatically pulls required models during setup.

See `deploy/SETUP_GUIDE.md` for detailed RunPod instructions.

## Generating Visualizations

After experiments complete:

```bash
# Experiment 0 dashboard
python scripts/generate_experiment0_dashboard.py

# Full findings gallery (all experiments)
python scripts/generate_gallery.py
```

Output goes to `visuals/` and `site/`.

## Cost Estimates

| Experiment | Generation Cost | Scoring Cost | Total |
|------------|----------------|--------------|-------|
| Exp 0 (50 queries, 1 config) | ~$0 (local Ollama) | ~$0.05 (6 judges) | ~$0.05 |
| Exp 1 (200 queries, 30 configs) | ~$0 (local Ollama) | ~$0.60 (Flash) | ~$0.60 |
| Exp 2 (200 queries, 16 configs) | ~$0 (local Ollama) | ~$0.32 (Flash) | ~$0.32 |
| RunPod GPU (A5000) | $0.27/hr | — | ~$2-3 per experiment |

All scoring uses Gemini 2.5 Flash by default (free tier covers small experiments).

## Troubleshooting

### "Cannot connect to Ollama"
- Local: Make sure Ollama is running (`ollama serve`)
- Remote: Check the --ollama-host URL includes the port (`:11434`)
- RunPod: Verify the pod is running and port 11434 is exposed

### "COST LIMIT REACHED"
The `--max-cost` ceiling was hit. Results are saved up to that point. Increase the
ceiling or use `--skip-generation` to re-score existing answers only.

### "Model not found"
Run `python scripts/pull_models.py` or `ollama pull MODEL_NAME` to download the model.

### Tests failing with import errors
Activate the virtual environment: `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (Unix). Then `pip install -r requirements.txt`.
