# task-050: OpenAI provider adapter + retroactive v3 scoring + future-run integration

## Goal

Add OpenAI as a third provider to `LLMScorer`, retroactively score Experiment 0v3 with two
OpenAI judges, and add OpenAI judges to the standing `JUDGE_CONFIGS` so they run automatically
in any future Exp 0 run. Also add OpenAI to the cost guard table.

## Why

The professor saw v3 and asked good questions. Adding ChatGPT as a third independent
scorer family (we currently only have Google + Anthropic) strengthens the scorer-validation
methodology — three families is meaningfully more robust than two for inter-judge agreement
analysis. The user has an OpenAI key and wants cost-effective models, not bleeding edge.

The CSV at `results/experiment_0_v3/raw_scores.csv` already contains everything needed
(`context_sent_to_llm`, `rag_answer`, `gold_answer`), and `run_experiment_0.py` already
supports `--skip-generation --judges <substring>` for incremental scoring (task-048).

## Models to use

**Two judges, GPT-5.4 family (NOT GPT-5.5 — that's flagship and pricier):**

- `gpt-5.4-mini` — $0.75 / $4.50 per 1M tokens (input/output). Comparable tier to gemini-2.5-flash and claude-haiku.
- `gpt-5.4` — $2.50 / $15.00 per 1M tokens. Comparable tier to gemini-2.5-pro and claude-sonnet.

Pricing source: OpenAI pricing page screenshot from user (2026-04-29). Knowledge cutoff
Aug 31 2025 (gpt-5.4 family) — fine for our use case since scoring doesn't depend on
recent world knowledge.

**Do NOT use** `gpt-5.5` (flagship, 2x the cost), `o1`, `o3`, `o1-mini`, or any model
with "preview" in the name. The scorer prompt is straightforward JSON output —
flagship and reasoning models add cost without benefit.

If `gpt-5.4` or `gpt-5.4-mini` is unavailable on the account (tier restrictions, etc.),
fall back to `gpt-4o` and `gpt-4o-mini` respectively and flag the substitution in result.md.
Verify availability with a single test call to each model before kicking off the 500-row
scoring run.

## Files to modify

### 1. `src/scorers/llm.py` — add OpenAI adapter

Add `_openai_adapter` alongside `_anthropic_adapter` and `_google_adapter`. Pattern is
identical to the existing two — lazy import, factory returns a `(prompt: str) -> str` callable.

```python
def _openai_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    """Create an OpenAI API caller.

    Lazy-imports the openai SDK so users who only use other providers
    don't need it installed.

    Args:
        model: OpenAI model ID (e.g., "gpt-4o-mini").
        api_key: API key, or None to use OPENAI_API_KEY env var.

    Returns:
        A callable that takes a prompt string and returns the response text.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return call
```

Register it in `_ADAPTERS`:
```python
_ADAPTERS = {
    "anthropic": _anthropic_adapter,
    "google": _google_adapter,
    "openai": _openai_adapter,
}
```

The `openai` package is already in `requirements.txt` (was added in task-020 for the
OpenAI-compatible LLM adapter). Do not add it again. Verify it's installed in the venv;
if not, `pip install openai` and update `requirements.txt`.

### 2. `src/cost_guard.py` — add OpenAI cost entries

Add to `COST_PER_CALL` dict (rough estimate, ~500 input + 100 output tokens per call,
err high):

```python
# OpenAI GPT-5.4 family (~500 input + 100 output tokens per call, err high)
# gpt-5.4-mini: 500*0.75/1M + 100*4.50/1M = $0.00083 → round to $0.0015
# gpt-5.4:      500*2.50/1M + 100*15.0/1M = $0.00275 → round to $0.005
"openai:gpt-5.4-mini": 0.0015,
"openai:gpt-5.4": 0.005,
# Fallbacks in case GPT-5.4 family unavailable
"openai:gpt-4o-mini": 0.0002,
"openai:gpt-4o": 0.005,
```

Cost estimates intentionally err high — CostGuard is a safety net, not an accountant.
Expected v3 backfill cost at 500 rows: ~$0.40 (5.4-mini) + ~$1.40 (5.4) = ~$2 total.

### 3. `scripts/run_experiment_0.py` — add OpenAI judges to `JUDGE_CONFIGS`

Append after the existing Anthropic entries:

```python
JUDGE_CONFIGS = [
    # Gemini judges (free via Google AI Studio)
    {"provider": "google", "model": "gemini-2.5-flash-lite"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "google", "model": "gemini-2.5-pro"},
    {"provider": "google", "model": "gemini-3.1-pro-preview"},
    # Anthropic judges (optional — skipped if ANTHROPIC_API_KEY not set)
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"provider": "anthropic", "model": "claude-opus-4-20250514"},
    # OpenAI judges (optional — skipped if OPENAI_API_KEY not set)
    {"provider": "openai", "model": "gpt-5.4-mini"},
    {"provider": "openai", "model": "gpt-5.4"},
]
```

The existing scorer-init loop already skips configs whose API key is missing — confirm
that path also handles `OPENAI_API_KEY` (it should, since each provider's SDK reads its
own env var; just verify by reading the scorer init block around line 305-311).

### 4. Same in `scripts/run_experiment_1.py` and `scripts/run_experiment_2.py`

Find the scorer config in each (likely a single Gemini Flash scorer for the matrix runs —
not a multi-judge list). **Do not add OpenAI to Exp 1 & 2 by default** — those experiments
score thousands of configs and OpenAI cost would balloon. The user wants OpenAI specifically
for scorer-validation runs (Exp 0), not the matrix experiments. Leave Exp 1 & 2 scorer
configs untouched. Mention this decision explicitly at the top of the spec section in those
files? No — the spec lives here; do not edit those files.

### 5. Tests — `tests/test_scorer.py` (or wherever LLMScorer tests live)

Add tests mirroring the Google/Anthropic ones:
- `test_openai_adapter_creates_callable` — mock `openai.OpenAI`, verify factory returns callable
- `test_openai_adapter_calls_api_with_correct_model` — mock the client, assert `chat.completions.create` called with the right model
- `test_openai_adapter_extracts_text_from_response` — mock response object, assert returned text matches `choices[0].message.content`
- `test_llmscorer_with_openai_provider` — full integration with mocked client, score one item, assert dict structure

Mock everything. No real API calls in tests.

### 6. Add to `tests/test_experiment_0.py`

Update the `JUDGE_CONFIGS` length assertion (currently expects 7) to expect 9. If there's
an assertion listing exact provider/model pairs, add the two OpenAI entries.

## Retroactive v3 scoring run

After the code changes are merged and tests pass, the night instance should run:

```bash
python scripts/run_experiment_0.py \
  --version v3 \
  --skip-generation \
  --judges gpt
```

The `--judges gpt` substring filter will select only the two OpenAI judges and merge their
scores into the existing `results/experiment_0_v3/raw_scores.csv` without retouching the
existing 6 judges' columns (this is the per-judge resume logic from task-048).

Expected new columns:
- `openai_gpt_4o_mini_faithfulness`, `_relevance`, `_conciseness`, `_quality`
- `openai_gpt_4o_faithfulness`, `_relevance`, `_conciseness`, `_quality`

Expected runtime: ~10 minutes for 500 rows × 2 judges (gpt-4o-mini is fast).
Expected cost: ~$1 total. CostGuard ceiling stays at the script default ($5).

**If `OPENAI_API_KEY` is not set in `.env`, abort the scoring run with a clear message —
do not silently skip.** The code-change part of the task can still complete; just don't
run the scoring step.

## After scoring: regenerate v3 report and gallery

```bash
python scripts/regenerate_v3_report.py    # if this exists; otherwise re-run the scoring step which writes report.md
python scripts/generate_gallery.py
```

The v3 dashboard's correlation/agreement charts should now include the two OpenAI judges
automatically (the dashboard reads all `*_quality` columns dynamically — verify this is
true; if not, flag it and don't try to refactor the dashboard in this task).

## What NOT to touch

- Do not modify Anthropic or Google adapters
- Do not refactor `_ADAPTERS` registration to be auto-discovery — explicit registry is fine
- Do not change the scorer prompt or rubric
- Do not add OpenAI judges to Exp 1 or Exp 2 scorer configs (cost reason)
- Do not regenerate v1 or v2 with OpenAI judges — those datasets are frozen
- Do not change cost guard ceiling defaults

## Quality checklist

- [ ] `_openai_adapter` works with mocked client
- [ ] Registered in `_ADAPTERS`
- [ ] CostGuard has both OpenAI model entries
- [ ] `JUDGE_CONFIGS` in `run_experiment_0.py` has both OpenAI entries appended
- [ ] If `OPENAI_API_KEY` missing, scorer init skips OpenAI configs without crashing
- [ ] All tests pass (existing + new)
- [ ] If key is set: v3 CSV has 8 new OpenAI columns, no existing columns changed
- [ ] If key is set: v3 report.md regenerated with OpenAI rows
- [ ] If key is set: gallery regenerated and shows OpenAI judges in correlation chart
- [ ] If key NOT set: code changes still merged; flag in result.md that scoring step was skipped

## Research / references

- OpenAI Python SDK: https://github.com/openai/openai-python
- Chat completions API: https://platform.openai.com/docs/api-reference/chat/create
- Pricing: https://openai.com/api/pricing/
