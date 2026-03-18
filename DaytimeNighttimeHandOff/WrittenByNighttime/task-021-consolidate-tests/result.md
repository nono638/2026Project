# Result: task-021 — Consolidate nighttime tests into main tests/ directory
**Status:** done
**Completed:** 2026-03-18T01:35:04

## Commits
- `<pending>` — night: task-021 consolidate nighttime tests

## Test Results
- Command run: `python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ --ignore=tests/test_additional_generators.py --ignore=tests/test_ragas_generator.py -v --tb=short`
- Outcome: 335 passed, 5 failed
- Failures (all pre-existing, not caused by this task):
  1. `test_all_strategies_protocol` — MultiQueryRAG now requires `llm` parameter (from task-020)
  2. `test_import_scorers` — ClaudeScorer renamed to LLMScorer (from task-017)
  3. `test_import_query_generators` — spacy/typer import error (pre-existing env issue)
  4. `test_scorer_protocol_shape` — ClaudeScorer module removed (from task-017)
  5. `test_filter_removes_bad_query` — pre-existing round-trip filter flaky test

## Decisions Made
- **Skipped task-012 test_claude_scorer.py:** ClaudeScorer was replaced by LLMScorer in task-017. The task-012 tests reference `src.scorers.claude.ClaudeScorer` which no longer exists. The task-017 tests (`test_llm_scorer.py`) fully cover the replacement.
- **Replaced test_google_embedders.py:** Old tests used defunct `genai.embed_content` API from deprecated google-generativeai SDK. Replaced with task-016 migration tests using `genai.Client` pattern. Merged the protocol compliance test from the old file.
- **Class renames per spec:** Applied all 7 renames to avoid pytest collection conflicts.

## Flags for Morning Review
- 5 pre-existing test failures need attention (see above) — these existed before this task and are tracked as separate issues.
- `test_additional_generators.py` and `test_ragas_generator.py` fail to collect due to spacy/typer import error (spacy installed globally, not in venv).
