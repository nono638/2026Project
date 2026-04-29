# Result: task-050 — OpenAI provider adapter + retroactive v3 scoring

**Status:** done
**Completed:** 2026-04-29T13:25:00

## Commits
- (filled in after commit) — night: task-050 OpenAI judge adapter

## Test Results
- Command run: `python -m pytest tests/test_llm_scorer.py tests/test_cost_guard.py -v`
- Outcome: 23 passed
- Full suite (`python -m pytest tests/ --ignore=DaytimeNighttimeHandOff/`): 637 passed, 15 failed
- Failures: all 15 are pre-existing bert_score / exp12 scorer_columns failures unrelated to this task (bert_score module install issues, flagged in task-047)

## Decisions Made
- Used `gpt-5.4-mini` and `gpt-5.4` as model IDs per spec (no fallback substitution made — code-only changes; if these models are unavailable at runtime, the existing scorer-init try/except will skip them and the user can re-run with `--judges gpt-4o` after manually editing JUDGE_CONFIGS).
- Updated `tests/test_cost_guard.py::test_unknown_model_uses_default` to use `mystery-provider/unknown-model-xyz` instead of `openai/gpt-4o-mini`, since `openai:gpt-4o-mini` is now a known cost (would have broken the test).
- Spec mentioned updating a `JUDGE_CONFIGS` length assertion expecting 7 in `tests/test_experiment_0.py` — no such assertion exists in the codebase, so nothing to update.

## Flags for Morning Review
- **Retroactive v3 scoring run NOT executed** — nighttime mode blocks network access (CLAUDE.md). The `--judges gpt` command must be run manually by the user when ready:
  ```
  python scripts/run_experiment_0.py --version v3 --skip-generation --judges gpt
  ```
  Then regenerate the gallery: `python scripts/generate_gallery.py`.
- Verify `gpt-5.4-mini` and `gpt-5.4` are available on the user's OpenAI account before running. If not, fall back to `gpt-4o-mini` / `gpt-4o` (cost guard entries already include fallback pricing).
- The v3 dashboard correlation/agreement charts must read `*_quality` columns dynamically for the new OpenAI judges to appear automatically — not verified in this task.

## Attempted Approaches
N/A — straightforward implementation.
