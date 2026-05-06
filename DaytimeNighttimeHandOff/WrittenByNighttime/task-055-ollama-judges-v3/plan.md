# Plan: task-055 — Ollama provider adapter + 2 large local judges

## Approach

Mirror task-050's OpenAI-add pattern. Five focused edits, no design choices left
open — the spec already locked them.

## Files to modify

1. `src/scorers/llm.py` — add `_ollama_adapter` factory (~25 lines per spec
   block at lines 109–157), register `"ollama"` in `_ADAPTERS`, mention Ollama
   in module docstring's adapter list.
2. `src/cost_guard.py` — add `ollama:gemma4:31b: 0.0` and
   `ollama:qwen3.6:27b: 0.0` to `COST_PER_CALL` with section comment
   explaining why ($0 prevents the $0.01 default fallback from burning the
   $5 ceiling on a 500×2 judge run).
3. `scripts/run_experiment_0.py` — append two ollama entries to
   `JUDGE_CONFIGS`; fix the inaccurate header comment from "9 LLM judges
   (4 Gemini + 3 Claude + 2 OpenAI)" to "13 LLM judges (4 Gemini + 5 Claude
   + 2 OpenAI + 2 Ollama)" (existing list already has 11 entries — comment
   was stale before this task).
4. `scripts/generate_experiment0_dashboard.py` — add `"consensus_quality"`
   to `_NON_JUDGE_QUALITY_COLS` with comment referencing task-052.
5. Tests — copy the two pre-written test files into `tests/`:
   - `tests/test_ollama_scorer.py` (from `test_ollama_adapter.py`)
   - `tests/test_cost_guard_ollama.py` (kept as separate file to keep the
     existing `tests/test_cost_guard.py` clean; spec said separate-file is
     fine).

## Implementation notes

- Adapter uses `requests.post` directly (not the `ollama` python package).
  Spec says match task-053's pattern in `experiment_utils.py`. The tests
  patch `requests.post` at the global `requests` module, so the adapter
  should `import requests` and call `requests.post(...)` — NOT alias it as
  `_requests` at module top (the test docstring shows the patch target is
  `requests.post`, not `src.scorers.llm._requests.post`).
- Spec's reference impl uses `import requests as _requests` "to avoid
  shadowing if a test patches the name". But the actual pre-written tests
  patch the global `requests.post`, so we need the adapter to call into
  whatever `requests.post` resolves to at call time. The simplest correct
  approach: `import requests` inside the closure body (or at module top),
  and call `requests.post(...)`. Then `@patch("requests.post")` correctly
  intercepts it because Python attribute lookup happens at call time.
- `OLLAMA_HOST` env handling: spec defines three cases — unset (use
  localhost:11434), bare host (prepend http://), full URL with trailing
  slash (strip slash). All three are tested.
- The init-time vs call-time distinction matters: `LLMScorer.__init__`
  must NOT make a network call. The factory only constructs the URL string
  and returns the closure. The `ConnectionError` first surfaces during
  `score()` — wrapped by the scorer's retry loop into `ScorerError`. With
  `max_retries=0`, the first attempt's exception bubbles immediately.
- The `ConnectionError` test relies on the retry logic's substring match
  catching "connection" — but with `max_retries=0`, retry path is skipped
  and we go straight to `raise ScorerError(...) from exc`. Either way the
  test passes.

## Ambiguities resolved

- **Where to put adapter `import requests`?** Inside the factory function
  body (not at module top). Lazy-import matches the pattern of the other
  three adapters (anthropic, google, openai are all imported inside their
  factories). Tests still work because `@patch("requests.post")` patches
  the attribute on the module object — and the adapter's call resolves to
  that same module object.
- **Test file naming.** Spec says copy into `tests/test_llm_scorer.py` OR
  separate file. Going separate (`tests/test_ollama_scorer.py`) to avoid
  conflict with existing `test_llm_scorer.py` content and class names.
  Same for cost guard tests.

## Out of scope (per spec "What NOT to Touch")

- Existing adapters, prompt template, `_parse_response`.
- Exp 1/2 default scorer panel.
- v1/v2 datasets.
- The retroactive scoring run itself (morning task on the 5090).

## Tests

Pre-written tests in `WrittenByDaytime/.../tests/`. After copying to
`tests/`, run:
```
pytest tests/test_ollama_scorer.py tests/test_cost_guard_ollama.py -v
```
Then run the full regression (excluding handoff dir):
```
pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v
```
