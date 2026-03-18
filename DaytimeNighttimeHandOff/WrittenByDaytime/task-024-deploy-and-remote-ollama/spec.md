# Task 024: Deploy Script + Remote Ollama Support

## What to Build

Two things:
1. `deploy/setup_pod.py` — a script that creates a RunPod pod with Ollama, pulls models,
   and verifies everything works. Fully automated, no SSH required.
2. `--ollama-host` CLI flag on `run_experiment_0.py` and `run_experiment.py`, plus a
   `host` parameter on `OllamaEmbedder`, so both scripts can point at a remote Ollama
   instance (like one running on RunPod).

## Why

The user needs to go from "I have a RunPod account" to "Experiment 0 is running" with
a single command per step. Right now there's a gap: `RunPodManager` can create pods but
there's no automation for installing Ollama, pulling models, or pointing the experiment
scripts at the remote GPU. This task closes that gap.

## Files to Create

### `deploy/setup_pod.py`

End-to-end pod setup script. Uses `RunPodManager` from `deploy/runpod_manager.py`.

**CLI interface:**
```
python deploy/setup_pod.py                              # full setup: create pod, pull default models
python deploy/setup_pod.py --pod-id abc123 --pull-only  # just pull models on existing pod
python deploy/setup_pod.py --pod-id abc123 --pull-only --models qwen3:0.6b qwen3:1.7b
```

**Arguments:**
- `--pod-id` — skip pod creation, use this existing pod. Optional.
- `--pull-only` — skip pod creation, just pull models. Requires `--pod-id`.
- `--models` — space-separated list of models to pull. Default: `mxbai-embed-large qwen3:4b`.
- `--name` — pod name. Default: `ragbench-gpu`.
- `--image` — Docker image. Default: `ollama/ollama` (has Ollama pre-installed and serving on port 11434).
- `--volume-gb` — volume size. Default: `20`.

**What the script does (full setup mode):**

1. Load `.env` file (for `RUNPOD_API_KEY`).
2. Check `RUNPOD_API_KEY` is set. Print clear error if not.
3. Create `RunPodManager` instance.
4. Query and print balance. Exit if balance < $1.00.
5. Create pod:
   - Image: `ollama/ollama`
   - Env: `{"OLLAMA_HOST": "0.0.0.0"}` — required so Ollama accepts external connections.
   - Ports: `["11434/http"]` (just Ollama — FastAPI comes later).
   - Use default GPU fallback list from `RunPodManager`.
6. Wait for pod to be ready (`wait_for_ready`, timeout 300s).
7. Build the proxy URL: `https://{pod_id}-11434.proxy.runpod.net`
8. Wait for Ollama to actually respond (poll `GET {url}/api/tags` until 200, timeout 120s,
   poll every 5s). The pod being "ready" in RunPod doesn't mean Ollama is serving yet.
9. Pull each model by POSTing to `{url}/api/pull` with `{"name": "model_name", "stream": false}`.
   - This is Ollama's HTTP API for pulling models — no SSH needed.
   - Print progress for each model.
   - Timeout: 600s per model (large models take time to download).
   - If a pull fails, print the error but continue to the next model.
10. Verify each model by POSTing to `{url}/api/generate` with a short test prompt
    (`{"model": "model_name", "prompt": "Say hello", "stream": false}`).
    - Just check for a 200 response. Don't validate the content.
11. Print summary block:
    ```
    ========================================
      Pod is ready for experiments!
      Pod ID:     {pod_id}
      Ollama URL: {url}
      Spend rate: ${rate}/hr
      Balance:    ${balance}
    ========================================
    ```

**What the script does (pull-only mode):**

1. Load `.env`, check key, create manager.
2. Build proxy URL from `--pod-id`.
3. Verify Ollama is responding.
4. Pull requested models (step 9 above).
5. Verify models (step 10 above).
6. Print summary.

**Error handling:**
- `RUNPOD_API_KEY` not set → print message pointing to the setup guide, exit 1.
- Balance < $1.00 → print warning and exit 1.
- Pod creation fails → print RunPodError message, exit 1.
- Pod timeout → print message pointing to console URL, exit 1.
- Ollama not responding after pod ready → print troubleshooting tips, exit 1.
- Model pull fails → print error, continue to next model, note failure in summary.
- Model verification fails → print warning, continue.

**Do NOT use SSH.** Everything goes through Ollama's HTTP API on the proxy URL.

### `tests/test_setup_pod.py`

Tests with mocked HTTP responses. No real API calls.

## Files to Modify

### `src/embedders/ollama.py`

Add `host` parameter to `OllamaEmbedder.__init__()`:

```python
def __init__(self, model: str = "mxbai-embed-large", host: str | None = None) -> None:
    self._model = model
    self._client = Client(host=host) if host else Client()
    self._dimension: int | None = None
```

This matches the pattern already used by `OllamaLLM` in `src/llms/ollama.py`.

### `scripts/run_experiment_0.py`

1. Add `--ollama-host` argument to `parse_args()`:
   ```python
   parser.add_argument("--ollama-host", type=str, default=None,
                       help="Ollama server URL (default: localhost:11434). "
                            "Use RunPod proxy URL for remote GPU.")
   ```

2. Pass `host` through to `generate_answers()`. Change the function signature:
   ```python
   def generate_answers(documents, queries, model, ollama_host=None):
   ```

3. Inside `generate_answers()`, pass host to both OllamaLLM and OllamaEmbedder:
   ```python
   llm = OllamaLLM(host=ollama_host)
   ...
   embedder = OllamaEmbedder(host=ollama_host)
   ```

4. Also update the Ollama connectivity check (the `Client().list()` call) to use the
   host parameter:
   ```python
   Client(host=args.ollama_host).list() if args.ollama_host else Client().list()
   ```

5. Thread `args.ollama_host` through the `main()` function to `generate_answers()`.

### `scripts/run_experiment.py`

1. Add `--ollama-host` argument to `parse_args()` (same as above).

2. In `_build_llm()`, pass host:
   ```python
   return OllamaLLM(host=getattr(args, "ollama_host", None))
   ```

3. In `_build_embedder()`, pass host for the Ollama case:
   ```python
   return OllamaEmbedder(host=getattr(args, "ollama_host", None))
   ```

## Files NOT to Touch

- `deploy/runpod_manager.py` — already complete, use as-is
- `src/llms/ollama.py` — already has `host` parameter
- `src/protocols.py` — no protocol changes needed
- Anything in `src/strategies/` — strategies don't know about hosts

## Edge Cases

- `ollama/ollama` image starts Ollama automatically on port 11434, but it may take
  10-30 seconds after pod boot. The script must poll `/api/tags` until it responds.
- Model pulls via HTTP API can be slow (~5 min for a 2.5GB model). Use a long timeout.
- The Ollama HTTP API's `/api/pull` with `stream: false` blocks until complete and returns
  a single JSON response. With `stream: true` it streams progress lines.
  Use `stream: false` for simplicity. Set `requests` timeout to 600s.
- RunPod's proxy has a 100-second Cloudflare timeout. For model pulls that take longer,
  use `stream: true` and read lines to keep the connection alive. **Decision: use
  `stream: true` for pulls** — read and discard progress lines, print a dot every 10
  lines to show activity. This avoids the Cloudflare 524 timeout.
- If `--ollama-host` is not provided, both scripts work exactly as before (localhost).

## Rationale

- **`ollama/ollama` Docker image** rather than installing Ollama via startup script:
  the official image has Ollama pre-installed and auto-starts the server. No SSH, no
  startup scripts, no waiting for installation. Simpler and faster.
  Docs: https://hub.docker.com/r/ollama/ollama
- **HTTP API for model pulling** rather than SSH exec: RunPod doesn't have a command
  execution API. SSH requires key setup. Ollama's HTTP API (`/api/pull`) does the same
  thing and goes through the proxy URL we already have.
  Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model
- **`stream: true` for pulls**: Cloudflare (RunPod's proxy) enforces a 100-second
  connection timeout. A 2.5GB model pull takes longer than that. Streaming keeps the
  connection alive.

## Tests

All tests mock HTTP calls. No real API or RunPod calls.

### `tests/test_setup_pod.py`

1. **test_full_setup_flow** — mock RunPodManager.create_pod, wait_for_ready, get_balance,
   get_spend_per_hour. Mock requests.get for /api/tags. Mock requests.post for /api/pull
   and /api/generate. Verify the script creates a pod, waits, pulls models, verifies.

2. **test_pull_only_mode** — mock with --pod-id and --pull-only. Verify no pod creation,
   just pull and verify.

3. **test_missing_api_key** — unset RUNPOD_API_KEY, verify clean error message and exit 1.

4. **test_low_balance** — mock balance < $1.00, verify exit 1.

5. **test_ollama_not_responding** — mock /api/tags returning errors, verify timeout message.

6. **test_model_pull_failure** — mock /api/pull returning error for one model. Verify
   script continues to next model and notes failure in output.

7. **test_cloudflare_timeout_handled** — mock streaming pull response with progress lines.
   Verify script reads through them without error.

### `tests/test_ollama_embedder_host.py`

8. **test_embedder_default_host** — create OllamaEmbedder() with no host, verify
   Client() called with no args.

9. **test_embedder_custom_host** — create OllamaEmbedder(host="http://remote:11434"),
   verify Client(host=...) called.

### Existing test files (verify no regressions)

Run `python -m pytest tests/ -v` and confirm existing tests still pass.

## Quality Checklist

- [x] Exact files to modify are listed
- [x] All edge cases explicit (Cloudflare timeout, slow pulls, Ollama startup delay)
- [x] All judgment calls made (ollama/ollama image, stream:true for pulls, HTTP not SSH)
- [x] Why answered for every decision
- [x] Research URLs included
- [x] No new dependencies needed (requests already installed, ollama already installed)
- [x] Tests cover key behaviors
- [x] Scoped to one focused session
