# Result: task-055 — Ollama provider adapter + 2 large local judges
**Status:** done
**Completed:** 2026-05-06T16:03:41

## Commits
- `<sha>` — night: task-055 Ollama provider adapter + gemma4:31b/qwen3.6:27b judges

## Test Results
- Command run: `.venv/Scripts/python.exe -m pytest tests/ --ignore=DaytimeNighttimeHandOff/ -q`
- Outcome: 672 passed, 12 skipped, 4 warnings
- New tests added: 15 (12 in `tests/test_ollama_scorer.py`, 3 in
  `tests/test_cost_guard_ollama.py`) — all green.
- Failures: none.

## Changes Made

1. **`src/scorers/llm.py`** — added `_ollama_adapter` (HTTP via `requests`,
   `OLLAMA_HOST` env var, `/api/chat`, `stream=False`, `temperature=0.0`,
   300s timeout). Registered `"ollama"` in `_ADAPTERS`. Updated module
   docstring's provider list to include OpenAI and Ollama.
2. **`src/cost_guard.py`** — added `ollama:gemma4:31b: 0.0` and
   `ollama:qwen3.6:27b: 0.0` to `COST_PER_CALL` with explanatory comment.
3. **`scripts/run_experiment_0.py`** — appended two ollama configs to
   `JUDGE_CONFIGS`. Fixed the inaccurate header comment (was "9 LLM judges
   (4 Gemini + 3 Claude + 2 OpenAI)" — the list already had 11 entries / 5
   Claude before this task) to "13 LLM judges (4 Gemini + 5 Claude + 2
   OpenAI + 2 Ollama)".
4. **`scripts/generate_experiment0_dashboard.py`** — added
   `"consensus_quality"` to `_NON_JUDGE_QUALITY_COLS` (defensive, per spec
   requirement 5). Also added two new entries to `JUDGE_DISPLAY_NAMES`
   ("Gemma 4 31B (local)" and "Qwen 3.6 27B (local)") — see Decisions.
5. **Tests** — copied pre-written tests into `tests/test_ollama_scorer.py`
   and `tests/test_cost_guard_ollama.py`.

## Decisions Made

- **Adapter `import requests` placement.** Imported lazily inside the
  closure body, not aliased at module top. The pre-written tests patch
  `requests.post` (the global), so the adapter must resolve `requests.post`
  at call time. The spec's reference impl had `import requests as
  _requests` to "avoid shadowing if a test patches the name", but the
  actual tests patch the global, so the bare `import requests` inside the
  closure is what makes them pass. No behavioral difference at runtime.
- **JUDGE_DISPLAY_NAMES updated even though the spec didn't list it as a
  file to modify.** A pre-existing test
  (`TestJudgeConfigDisplayNamesInSync.test_every_judge_has_a_display_name`)
  enforces that every entry in `JUDGE_CONFIGS` has a display name. Adding
  the two ollama configs without updating the dict would fail regression.
  Names chosen to match the existing style and to flag local-vs-cloud at
  a glance in dashboard charts.
- **Test files kept as separate files** rather than appended to existing
  `tests/test_llm_scorer.py` and `tests/test_cost_guard.py`. Spec said
  either was fine; separate files keep the diff small and avoid touching
  the existing file's classes. Names: `tests/test_ollama_scorer.py` and
  `tests/test_cost_guard_ollama.py`.

## Flags for Morning Review

- **The retroactive `--skip-generation --judges ollama` scoring run is
  morning-side**, not done at night. Ollama is unreachable from the
  sandbox (network blocked + no GPU). Steps for the 5090:
  ```bash
  # First verify both tags exist (task-054 inbox flag — gate)
  ollama show gemma4:31b
  ollama show qwen3.6:27b
  # Pull if missing:
  # ollama pull gemma4:31b
  # ollama pull qwen3.6:27b

  python scripts/run_experiment_0.py --version v3 --skip-generation --judges ollama
  python scripts/generate_gallery.py
  ```
  Expected runtime: ~4 hours total (500 rows × 2 judges × ~10s/call). The
  existing per-judge BERT/F1 correlation table in
  `results/experiment_0_v3/report.md` gains two new rows — that's the
  validation answer.

- **Comment-count discrepancy noted:** the original `JUDGE_CONFIGS`
  header comment claimed "9 LLM judges (4 Gemini + 3 Claude + 2 OpenAI)"
  but the list already had 11 entries (5 Claude). The comment was stale
  before this task — now corrected to "13 LLM judges (4 Gemini + 5 Claude
  + 2 OpenAI + 2 Ollama)".

- **Untracked files preexisting on main** (left untouched, not staged):
  `WebsiteIdea1.png`, `results/experiment_0_v3/raw_scores_checkpoint.csv`.
  The orphaned stash from a prior session also still exists
  (`stash@{0}: WIP on main: 9dd89b2 tracker: task-049 done`). None of
  these are related to task-055.

## Attempted Approaches (if skipped/blocked)

N/A — task completed successfully on the first pass. The only mid-task
correction was adding `JUDGE_DISPLAY_NAMES` entries after the regression
test caught the omission.
