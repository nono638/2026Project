# Result: task-017 — Refactor ClaudeScorer → LLMScorer
**Status:** done
**Completed:** 2026-03-17T23:57:44

## Commits
- `<sha>` — night: task-017 refactor ClaudeScorer → LLMScorer

## Test Results
- Command run: `python -m pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-017-llm-scorer/tests/test_llm_scorer.py -v`
- Outcome: 13 passed, 1 failed
- Failures:
  - `TestGoogleProvider::test_score_returns_dict` — test bug, not implementation bug. The test decorator `@patch("builtins.__import__")` injects a mock arg but the method `test_score_returns_dict(self)` doesn't accept it (TypeError: takes 1 positional argument but 2 were given). The method body uses `patch.dict("sys.modules", ...)` which works correctly in `test_name_property` — the decorator is extraneous.

## Decisions Made
- Deleted `src/scorers/claude.py` entirely per spec (no backwards-compat alias).
- Updated `src/protocols.py` Scorer.name docstring example from 'claude:...' to 'anthropic:...'.
- Had to install `anthropic==0.84.0` — was in requirements.txt but not installed in venv.

## Flags for Morning Review
- `TestGoogleProvider::test_score_returns_dict` has a test bug — the `@patch("builtins.__import__")` decorator should be removed. The test body already uses `patch.dict("sys.modules", ...)` context manager which handles the mocking correctly.
- `anthropic` was not installed in the venv despite being in requirements.txt.

## Attempted Approaches (if skipped/blocked)
n/a
