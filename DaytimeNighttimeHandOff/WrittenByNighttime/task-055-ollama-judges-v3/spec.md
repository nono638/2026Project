# task-055: Ollama provider adapter + retroactive v3 scoring with two large local judges

## Summary

Add Ollama as a fourth provider to `LLMScorer`, append two large local judges
(`ollama:gemma4:31b`, `ollama:qwen3.6:27b`) to `JUDGE_CONFIGS`, and run the same
retroactive `--skip-generation --judges` step task-050 used for OpenAI to score
Experiment 0 v3's existing 500 generations. The validation question — does a
large local LLM agree with BERTScore/F1 on a level comparable to the API judges?
— answers itself in the existing `## Correlation with Gold Metrics` table in
`results/experiment_0_v3/report.md`: the new judges show up as two more rows.
No new metrics, no new reports, no separate v4 directory.

This is the post-hoc extension pattern proven in task-050 (OpenAI), reused for
local judges so the scorer-validation methodology covers a third axis (cloud
vs. local) on the same evaluation set.

## Why

We chose Haiku 4.5 + GPT-5.4 Mini for the Exp 1/2 default panel based on v3's
correlations against gold metrics. We don't yet know how a large local model
compares — `gemma4:31b` (dense, ~31B active) and `qwen3.6:27b` (dense, 27B
active) are the strongest local judge candidates in our model matrix and fit
comfortably on the RTX 5090 at Q4_K_M. The cost of the experiment is GPU time
on hardware we already pay for; the upside is either (a) a free judge for
future experiments, or (b) a documented negative result that strengthens the
writeup.

Active params, not total params, drive judgment quality on careful-reading
tasks — that's why this task uses `qwen3.6:27b` (dense) instead of
`qwen3.6:35b-a3b` (MoE, ~3B active). The MoE-as-judge question is interesting
but separate; out of scope here.

References:
- task-050 spec for the post-hoc-add pattern.
- v3 report: `results/experiment_0_v3/report.md` shows BERT/F1 correlation
  is the per-judge validation metric.
- task-053 introduced `experiment_utils.get_ollama_model_details()` which
  uses `requests` against Ollama's HTTP API — same pattern reused here.

## Requirements

1. `_ollama_adapter` factory exists in `src/scorers/llm.py`, registered in
   `_ADAPTERS` under the key `"ollama"`. It uses `requests` against
   `/api/chat`; reads `OLLAMA_HOST` env var (default
   `http://localhost:11434`); ignores the `api_key` argument (kept in the
   signature only for API symmetry with the other adapters).
2. `LLMScorer(provider="ollama", model=...)` returns scorer text correctly
   parsed by the existing `_parse_response` JSON extractor (no rubric or
   prompt changes; the rubric is fixed and provider-agnostic).
3. `JUDGE_CONFIGS` in `scripts/run_experiment_0.py` gains two entries:
   `{"provider": "ollama", "model": "gemma4:31b"}` and
   `{"provider": "ollama", "model": "qwen3.6:27b"}`, appended at the end of
   the list with an explanatory comment.
4. `src/cost_guard.py` `COST_PER_CALL` gains `"ollama:gemma4:31b": 0.0` and
   `"ollama:qwen3.6:27b": 0.0` with a comment that Ollama runs locally
   (GPU time only, no API spend). The default fallback ($0.01) must NOT be
   used for Ollama models.
5. `_NON_JUDGE_QUALITY_COLS` in `scripts/generate_experiment0_dashboard.py`
   gains `"consensus_quality"` defensively (not strictly needed today since
   Exp 0 doesn't emit that column, but task-052 added it elsewhere and the
   dashboard would mis-classify it as a judge if Exp 0 ever ran through the
   multi-judge path).
6. The runner's existing init-loop at lines 306–316 of
   `scripts/run_experiment_0.py` (try/except around `LLMScorer(**config, ...)`)
   must continue to gracefully skip Ollama judges when Ollama is unreachable,
   not crash. No code change required here — the existing `except (ScorerError,
   Exception)` already covers this — but the test in requirement 8 must
   verify this behavior.
7. Pre-written tests in `tests/` (this task's directory) cover the new
   adapter and the cost guard entries.
8. The retroactive scoring step (`--skip-generation --judges ollama`) is
   documented in `result.md` flags so the user runs it on the 5090. The
   night instance does NOT attempt to run it — Ollama is not reachable from
   the sandbox (network blocked + no GPU).

## Files to Modify

- `src/scorers/llm.py` — add `_ollama_adapter` (~25 lines), register in
  `_ADAPTERS`, update the docstring's "Provider adapters" header to mention
  Ollama. Do not modify the existing three adapters.
- `src/cost_guard.py` — add the two `ollama:*` entries to `COST_PER_CALL`
  with a section comment.
- `scripts/run_experiment_0.py` — append two entries to `JUDGE_CONFIGS`
  (lines 68–88) with an explanatory comment matching the existing comment
  style (see the OpenAI block at lines 83–87 for the template). Update the
  module-level header comment at line 66 from "9 LLM judges (4 Gemini + 3
  Claude + 2 OpenAI)" to "13 LLM judges (4 Gemini + 5 Claude + 2 OpenAI + 2
  Ollama)" — note that the existing count is wrong (it says 9 total / 3
  Claude but JUDGE_CONFIGS actually has 11 entries / 5 Claude). Fix the
  count while you're there.
- `scripts/generate_experiment0_dashboard.py` — add `"consensus_quality"`
  to `_NON_JUDGE_QUALITY_COLS` (line 71). One-line change with a comment
  referencing task-052.
- `tests/test_llm_scorer.py` — add the four test cases listed in Testing
  Approach below.
- `tests/test_cost_guard.py` — add one test asserting the two
  `ollama:*` cost entries are present and equal to 0.0.

## New Dependencies

None. `requests` is already a dependency (used by task-053's
`get_ollama_model_details`). The Ollama Python package is intentionally
NOT used — direct HTTP calls match the existing pattern in
`scripts/experiment_utils.py` and avoid a new dep.

## Implementation Detail — `_ollama_adapter`

```python
def _ollama_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    """Create an Ollama API caller.

    Reads OLLAMA_HOST from env (default http://localhost:11434). The
    api_key parameter is unused — kept in signature for adapter symmetry.

    Calls /api/chat with stream=False. Returns the assistant message
    content string. The rubric and prompt template are constructed by
    LLMScorer; this adapter only forwards the prompt and returns text.

    Args:
        model: Ollama model tag (e.g., "gemma4:31b", "qwen3.6:27b").
        api_key: Ignored for Ollama. Kept for adapter signature symmetry.

    Returns:
        A callable that takes a prompt string and returns the response text.

    Why /api/chat over /api/generate: chat endpoint accepts the same
    {role, content} message shape as the cloud SDKs, keeping the prompt
    construction in LLMScorer provider-agnostic. /api/generate would
    require a separate prompt format.
    """
    import os
    import requests as _requests  # module-level import in scope; alias to
                                  # avoid shadowing if a test patches the name

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    url = f"{host}/api/chat"

    def call(prompt: str) -> str:
        """Send prompt to Ollama /api/chat and return response text."""
        response = _requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=300,  # large models on slow hardware can take minutes
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    return call
```

Register in `_ADAPTERS`:
```python
_ADAPTERS = {
    "anthropic": _anthropic_adapter,
    "google": _google_adapter,
    "openai": _openai_adapter,
    "ollama": _ollama_adapter,
}
```

## Edge Cases

- **Ollama not reachable.** `requests.post` raises `ConnectionError`. The
  scorer-init loop in `run_experiment_0.py` already wraps `LLMScorer(...)`
  in `try/except (ScorerError, Exception)` and logs a skip. NOTE: the
  adapter creates the client (URL string + closure) eagerly at
  `LLMScorer.__init__` time, but no actual network call happens until
  `score()` runs — so the *init* won't fail, but the first `score()` call
  will. The existing per-row error path in the scorer batch loop handles
  this by leaving NaN in the row's score columns. Verify in the test in
  requirement 8 that a `ConnectionError` from a mocked `requests.post`
  surfaces as `ScorerError` from `LLMScorer.score`, matching the behavior
  of the other adapters.
- **Model not pulled on the host.** Ollama returns HTTP 404 with a body
  like `{"error": "model 'foo' not found"}`. `response.raise_for_status()`
  converts to `HTTPError`; the scorer's existing retry loop catches it,
  retries 3 times, then bubbles up as `ScorerError`. The user sees a clear
  error in the log. Document this in result.md so the morning user runs
  `ollama pull` first.
- **Empty response.** If `response.json()["message"]["content"]` is empty
  string, the existing `_parse_response` already handles non-JSON / empty
  text by raising `ScorerError`, which the scorer's retry loop catches.
  No new logic needed.
- **Slow first call.** Ollama loads the model into VRAM on first call,
  which can take 30+ seconds for a 31B model. The 300-second timeout above
  accommodates this. Subsequent calls are fast.
- **Rate limit / cost guard.** Cost guard is set to $0.0 for Ollama models,
  so it never trips. CostLimitExceeded won't fire on Ollama judges.

## Decisions Made

- **HTTP via `requests`, not `ollama` Python package.** **Why:** matches
  the existing pattern in `scripts/experiment_utils.py` (task-053). Avoids a
  new dependency. The `ollama` package is a thin wrapper around the same
  HTTP API — no benefit for this use case.
- **`OLLAMA_HOST` env var, not adapter parameter.** **Why:** matches how
  Google/Anthropic/OpenAI adapters fall back to env vars (`GOOGLE_API_KEY`,
  etc.) when `api_key=None`. Adding a `host` parameter would change the
  adapter signature and require updates to all four adapter functions and
  `_get_adapter`. Env var is the minimal, idiomatic fix.
- **`stream=False`.** **Why:** the scorer wants a single response string,
  not a token stream. Streaming would complicate the adapter without
  benefit here.
- **`temperature=0.0`.** **Why:** judges should be deterministic. The
  cloud adapters don't set temperature explicitly because their SDK
  defaults are already low for short JSON outputs; Ollama's default is 0.8
  which is too high for a structured rubric.
- **Timeout 300s.** **Why:** first call on a 31B model can take 30s for
  load + 5–10s for inference. Subsequent calls run in 5–10s. 300s gives
  generous margin without hanging indefinitely if the host is unreachable
  but not actively refusing.
- **`qwen3.6:27b` (dense) over `qwen3.6:35b-a3b` (MoE).** **Why:** active
  params drive judgment quality. 27B dense > 3B active MoE for careful
  reading. The MoE-as-judge question is a separate experiment.
- **Cost = $0.0, not absent from `COST_PER_CALL`.** **Why:** the default
  fallback is $0.01, which would falsely consume the cost guard ceiling
  on a 500-row run (500 × 0.01 × 2 judges = $10, hits the $5 ceiling).
  Explicit $0.0 prevents that.
- **Don't run the scoring step at night.** **Why:** the night sandbox has
  no Ollama and no network. Code lands; scoring happens on the 5090
  (matches task-050's pattern).

## Retroactive v3 scoring (run on the 5090, NOT at night)

After the code lands, on the 5090 with both models pulled and Ollama
running:

```bash
# Verify both tags exist (task-054 flag — do this first)
ollama show gemma4:31b
ollama show qwen3.6:27b

# Run the retroactive scoring
python scripts/run_experiment_0.py \
  --version v3 \
  --skip-generation \
  --judges ollama

# Regenerate the dashboard
python scripts/generate_gallery.py
```

Expected new columns in `results/experiment_0_v3/raw_scores.csv`:
- `ollama_gemma4_31b_faithfulness`, `_relevance`, `_conciseness`, `_quality`
- `ollama_qwen3_6_27b_faithfulness`, `_relevance`, `_conciseness`, `_quality`

(Per `_safe_scorer_name`, `:` and `.` and `-` all become `_`.)

The "Correlation with Gold Metrics" section of the regenerated report
gains two rows. That's the answer.

Expected runtime: ~2 hours per judge for 500 rows on a 5090 (rough — 31B
at Q4_K_M typically ~10s/call). Total ~4 hours. CostGuard ceiling stays
at the script default ($5) — Ollama costs $0 against it.

## What NOT to Touch

- Do not modify the existing Anthropic, Google, or OpenAI adapters.
- Do not change the scoring rubric or prompt template.
- Do not change `_parse_response` or any post-processing.
- Do not regenerate v1 or v2 — those datasets are frozen.
- Do not add Ollama judges to Exp 1 or Exp 2 default scorer panel — those
  experiments use a smaller, validated panel; adding an unvalidated judge
  to the headline numbers is what task-055 exists to prevent.
- Do not touch `scripts/run_experiment.py` (legacy single-script entry
  point — superseded).
- Do not refactor `_ADAPTERS` registration or attempt auto-discovery.
- Do not modify `tests/test_experiment_0.py` JUDGE_CONFIGS assertions if
  they exist (task-050 result.md noted no such assertion exists; verify
  again — if one does exist, update the count).

## Testing Approach

Pre-written tests live in
`DaytimeNighttimeHandOff/WrittenByDaytime/task-055-ollama-judges-v3/tests/`.
The night instance copies them into `tests/test_llm_scorer.py` and
`tests/test_cost_guard.py` (or adds them as separate test classes).

Tests cover:

1. `test_ollama_adapter_uses_default_host_when_env_unset` — unset
   `OLLAMA_HOST`, mock `requests.post`, assert URL is
   `http://localhost:11434/api/chat`.
2. `test_ollama_adapter_uses_env_host_when_set` — set `OLLAMA_HOST` to
   `gpu-pod:11434` (bare host, no scheme), assert URL becomes
   `http://gpu-pod:11434/api/chat` (scheme auto-prepended, matching
   task-053's pattern).
3. `test_ollama_adapter_sends_correct_payload` — mock `requests.post`,
   assert the JSON body has `model`, `messages=[{role,content}]`,
   `stream=False`, `options={"temperature": 0.0}`.
4. `test_ollama_adapter_extracts_message_content` — mock response
   `{"message": {"content": "hello"}}`, assert returned string is
   `"hello"`.
5. `test_ollama_adapter_raises_scorer_error_on_connection_error` — mock
   `requests.post` to raise `ConnectionError`, wrap a real `LLMScorer`
   call, assert `ScorerError` propagates from `score()`.
6. `test_llmscorer_with_ollama_provider` — full integration with mocked
   `requests.post`, score one item, assert dict has faithfulness/
   relevance/conciseness keys with floats in [1, 5].
7. `test_cost_guard_has_ollama_zero_costs` — assert
   `COST_PER_CALL["ollama:gemma4:31b"] == 0.0` and
   `COST_PER_CALL["ollama:qwen3.6:27b"] == 0.0`.

Mock everything with `unittest.mock.patch`. No real Ollama or network calls.

Run:
```
pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-055-ollama-judges-v3/tests/ -v
```
All 7 tests must pass after implementation. Full regression must also
pass (or fail only on pre-existing `bert_score` issues already documented).

## Quality Checklist

- [x] Exact files to modify listed with specific sections.
- [x] All edge cases explicit (Ollama down, model not pulled, empty response,
  slow first call, cost guard).
- [x] All judgment calls made and rationalized (HTTP vs SDK, env var vs
  param, stream=False, temp=0, timeout 300, dense vs MoE, $0 cost).
- [x] Why is answered for every non-obvious decision.
- [x] No new dependencies.
- [x] Tests cover key behaviors, not just "does it run".
- [x] Scoped to one focused session (~25 lines adapter + 4 small file
  edits + 7 tests).
