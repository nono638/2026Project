# Plan: task-050 OpenAI judge

## Files to modify
1. `src/scorers/llm.py` — add `_openai_adapter` + register in `_ADAPTERS`
2. `src/cost_guard.py` — add OpenAI cost entries
3. `scripts/run_experiment_0.py` — append OpenAI judges to JUDGE_CONFIGS
4. `tests/test_llm_scorer.py` — add OpenAI provider tests (mocked)

## Approach
- Mirror existing google/anthropic adapter pattern. Use `from openai import OpenAI` lazy import.
- Cost guard: add 4 entries (gpt-5.4-mini/gpt-5.4 + gpt-4o-mini/gpt-4o fallbacks).
- JUDGE_CONFIGS: append 2 entries (gpt-5.4-mini, gpt-5.4). Use `gpt-5.4*` model IDs per spec.
- Tests: mirror Google pattern with `patch.dict("sys.modules", ...)` for `openai`.

## Ambiguities
- Spec says "JUDGE_CONFIGS length assertion (currently expects 7)" — there is no such assertion; ignore.
- No length asserts in tests/ for judges.

## Skipped from spec
- **Retroactive v3 scoring run** — REQUIRES NETWORK ACCESS, blocked in nighttime mode (CLAUDE.md). Will flag in result.md. The `--judges gpt` command is not run.
- Exp 1 & 2 scorer configs intentionally untouched (per spec).

## Test plan
- Run new test file additions + full pytest excluding handoff dir.
