# task-052: Multi-judge support for Experiments 1 and 2

## Summary

Currently `scripts/run_experiment_1.py` and `scripts/run_experiment_2.py` accept a single
`--scorer provider:model` argument. The final scorer panel decision (documented in
`docs/methodology.html` and `scripts/generate_gallery.py`) is **Claude Haiku 4.5 +
GPT-5.4 Mini** as a two-judge cross-provider panel. Add multi-judge support to both
scripts, aligning their column schema with Experiment 0's existing `<safe_name>_quality`
convention so downstream dashboards and the scorer-validation work transfer cleanly.

This task does **not** run real Exp 1 or Exp 2 data — that's task-053+ (deferred for the
RTX 5090 leverage discussion). This task only changes the CLI surface, scoring loop,
column naming, and tests.

## Requirements

1. `run_experiment_1.py` and `run_experiment_2.py` accept `--scorers PROVIDER:MODEL [PROVIDER:MODEL ...]`
   (nargs='+'). The default value is `["anthropic:claude-haiku-4-5-20251001", "openai:gpt-5.4-mini"]`.
2. The old singular `--scorer` flag is **removed** (no backward-compat alias). This is a
   research repo, not a published library; clean break is fine.
3. For each row in the run, every judge in the panel scores the answer. Per-judge columns
   use the Experiment 0 schema:
   - `<safe_name>_faithfulness`, `<safe_name>_relevance`, `<safe_name>_conciseness`,
     `<safe_name>_quality`, `<safe_name>_scorer_latency_ms`
   - where `safe_name = _safe_scorer_name("provider:model")` (already exported from
     `scripts/run_experiment_0.py`). Examples:
     - `anthropic:claude-haiku-4-5-20251001` → `anthropic_claude_haiku_4_5_20251001`
     - `openai:gpt-5.4-mini` → `openai_gpt_5_4_mini`
4. The bare `faithfulness/relevance/conciseness/quality/scorer_latency_ms` columns are
   **removed** from the CSV schema. report.md and any downstream consumers must read from
   the per-judge columns or a derived consensus column (see Decision 3 below).
5. A **consensus** column `consensus_quality` is written: simple unweighted mean of each
   judge's `<safe_name>_quality` per row. NaN-safe — if one judge returns NaN, the
   consensus uses the remaining judge(s); if all are NaN, consensus is NaN.
6. report.md uses `consensus_quality` for headline tables and per-config summaries. A new
   "Per-judge agreement" section reports the Pearson correlation between each pair of
   judges' quality scores (e.g., `r(haiku, gpt-mini) = 0.72`).
7. Cost guard tracks aggregate spend across all judges. The single `--max-cost` argument
   continues to be the global ceiling (not per-judge). When the limit is hit mid-row, all
   subsequent judges for that row return NaN; the row is still written.
8. The metadata.json sidecar (already wired via `write_experiment_metadata`) lists every
   judge in the panel with its `n_scored` count, mirroring Experiment 0's structure.
9. `--skip-generation` mode (re-score existing answers) supports per-judge resume: if a
   row already has non-NaN values for a given judge's `_quality` column, that judge is
   skipped for that row. New judges added to the panel are scored from scratch on every
   row. Same pattern as `score_all_answers` in `run_experiment_0.py`.

## Files to Modify

- `scripts/run_experiment_1.py`
  - **CLI** (line 154): replace `--scorer` argparse entry with `--scorers` nargs='+',
    `default=["anthropic:claude-haiku-4-5-20251001", "openai:gpt-5.4-mini"]`
  - **Scorer construction** (line 385, the `build_scorer(args.scorer, ...)` line): build a
    list of scorers — `scorers = [build_scorer(s, max_cost=args.max_cost) for s in args.scorers]`.
    Note that `build_scorer` attaches a `CostGuard` per scorer; the cost guard will need a
    shared instance — see Decision 6.
  - **Scoring loops** (~line 400 for `--skip-generation`, ~line 501 for full run): replace
    each `scores = score_answer(scorer, ...)` call with a loop that scores against each
    judge and merges results into per-judge-prefixed keys. Compute and add
    `consensus_quality` after.
  - **Row schema** (~line 517-540): drop bare `faithfulness/relevance/conciseness/quality`
    and `scorer_latency_ms` keys; replace with the per-judge prefixed columns plus
    `consensus_quality`.
  - **report.md generation** (search for `quality` references in report-building code): use
    `consensus_quality` for headline. Add per-judge agreement section.
  - **metadata write** (already calls `write_experiment_metadata`): pass the panel of
    judges as the `judges=` arg, populated like `run_experiment_0.write_metadata` does.
  - **Logging / cost summary** (search for `args.scorer`): update prints/logs to list all
    judges.
- `scripts/run_experiment_2.py`
  - Same set of changes as Exp 1 (lines are slightly different — search by name, not
    line number).
- `scripts/experiment_utils.py`
  - Add a new helper `score_answer_multi(scorers: list, query: str, context: str, answer: str,
    existing_row: dict | None = None) -> dict` that:
    - For each scorer, computes the safe_name prefix.
    - If `existing_row` is provided AND has non-NaN `<safe>_quality`, skips that scorer
      and copies the existing values (for resume support).
    - Otherwise calls `scorer.score(...)` and prefixes the resulting keys.
    - On cost-limit-exceeded, returns NaN for that judge but continues with remaining judges.
    - Computes and includes `consensus_quality` (NaN-safe mean).
    - Returns the merged dict.
  - Keep the existing single-scorer `score_answer(...)` function intact — it's still used
    by `run_experiment_0.py` for individual per-judge calls.
- `tests/test_experiment_utils.py` (or create if missing)
  - Add tests for `score_answer_multi` covering: two-judge happy path, one-judge-fails-NaN,
    consensus computation including NaN-safe averaging, and resume-skip behavior.
- `scripts/generate_experiment0_dashboard.py` — **read but NOT modify**. The Exp 0
  dashboard already handles the per-judge prefix schema.
- Any Exp 1 / Exp 2 dashboard scripts under `scripts/` (search for `experiment_1` and
  `experiment_2`) that consume bare `quality` — update to use `consensus_quality`.

## New Dependencies

None — all required packages are already installed.

## Edge Cases

- **All judges in panel use the same provider** (e.g., `--scorers anthropic:claude-haiku-4-5
  anthropic:claude-sonnet-4-6`): no special handling, the safe_name prefix already
  disambiguates by model.
- **Same scorer specified twice on the CLI** (e.g., `--scorers a:b a:b`): treat as a config
  error, exit with a clear message before any API calls. (Otherwise we'd double-cost and
  produce a single column representing two runs averaged silently.)
- **Cost limit hit on first judge of a row**: the row is still written with all per-judge
  columns NaN and consensus NaN. Run continues until the next config boundary, where
  `run_experiment_1` already has the existing `cost_limit_hit` exit path — preserve it.
- **One judge returns NaN, the other succeeds**: consensus uses the surviving judge's
  quality value.
- **All judges return NaN for a row**: consensus is NaN. report.md should drop NaN rows
  before computing aggregate stats (the existing `quality.mean()` calls already handle
  this in pandas, but verify).
- **`--skip-generation` with a panel containing a judge that wasn't in the original run**:
  the new judge's columns are absent in the loaded CSV, so every row is scored from
  scratch for that judge. Existing judges' columns are preserved and skipped per-row.
- **`--skip-generation` with fewer judges than the CSV originally had**: the missing
  judges' columns are left untouched (not deleted). Consensus is recomputed from the
  judges in the *current* panel only.

## Decisions Made

1. **CLI flag rename** (`--scorer` → `--scorers`, no alias): clean break. **Why:** this is
   a research script, not a library — preserving the singular flag would invite confusion
   ("which one wins if I pass both?"). Documented in this spec; no other repo touches the
   flag.
2. **Default panel = `claude-haiku-4-5-20251001` + `gpt-5.4-mini`**. **Why:** decided in
   conversation 2026-04-30 based on Exp 0 v3 results. Cost vs accuracy trade documented in
   the methodology page (`docs/methodology.html` "Scorer Selection" section). ~$32 total
   for default-N runs of both experiments vs ~$106 for Sonnet 4.6 panel.
3. **Headline column = `consensus_quality` (unweighted mean)**, not "designate one judge
   as primary." **Why:** with two judges of comparable accuracy and different bias profiles,
   averaging is more honest than picking one. The per-judge columns remain available for
   anyone who wants to audit disagreement.
4. **Drop bare `faithfulness/relevance/etc.` columns entirely** (not keep them as
   "primary judge" duplicates). **Why:** existing dashboards that consume bare `quality`
   need updating regardless; making it explicit avoids the trap of code reading bare
   `quality` and silently treating it as primary.
5. **Per-judge agreement reported as Pearson r between quality scores**, not as e.g.
   inter-rater kappa. **Why:** Pearson aligns with what Exp 0 v3 already reports for
   judge-judge correlation (`docs/experiment_0_v3.html` shows the cross-judge correlation
   heatmap). Same metric across experiments is easier to interpret.
6. **Single shared `CostGuard` instance across all judges in the panel**, not one
   guard per scorer. **Why:** `--max-cost` is a global budget, not a per-judge budget.
   Modify `build_scorer` to optionally accept an existing CostGuard instance, or
   build the guard once outside and inject it into each scorer via a small helper.
   Reference: `src/cost_guard.py` already supports a single guard tracking calls across
   any number of `record_call("provider:model", ...)` invocations.
7. **Resume on `--skip-generation` is per-judge per-row**, not per-config. **Why:** the
   existing config-level `completed_configs` resume happens at strategy×model granularity,
   which is the right unit for generation. But for re-scoring, judge×row is finer-grained
   and lets us add a new judge cheaply without re-paying for the existing ones. Same
   pattern as `score_all_answers` in `run_experiment_0.py`.

## What NOT to Touch

- `scripts/run_experiment_0.py` — already multi-judge; its schema is the reference, not
  the target.
- `scripts/generate_experiment0_dashboard.py` — works with the per-judge schema already.
- `src/scorers/llm.py` and `src/cost_guard.py` — no API changes needed; just call sites.
- The generation loop's strategy×model checkpoint logic — that's correct and stays.
- Any answer-generation code paths — this task is scoring-only.

## Testing Approach

Pre-written tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-052-multi-judge-exp12/tests/`:

- `test_score_answer_multi.py`:
  - `test_two_judges_happy_path` — both judges return scores, consensus is mean of the two.
  - `test_one_judge_nan` — one judge fails (returns NaN), consensus uses the survivor.
  - `test_all_judges_nan` — all fail, consensus is NaN.
  - `test_resume_skip` — `existing_row` has non-NaN values for one judge → that judge is
    skipped, scorer.score is not called for it.
  - `test_safe_name_prefix` — column keys use `<safe_name>_<metric>` not bare metric names.
- `test_cli_scorers_flag.py`:
  - `test_default_panel` — argparse default is the two-judge panel.
  - `test_duplicate_scorers_rejected` — `--scorers a:b a:b` exits with non-zero.
  - `test_old_singular_flag_removed` — `--scorer` raises SystemExit (argparse error).

Tests should pass after implementation. They will fail before implementation because
`score_answer_multi` doesn't exist and `--scorers` isn't wired up.

To run after implementation:
```
.venv/Scripts/python -m pytest tests/test_score_answer_multi.py tests/test_cli_scorers_flag.py -v
```
