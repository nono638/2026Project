# Plan: task-052 multi-judge support for Exp 1/2

## Files to modify
1. `scripts/experiment_utils.py`
   - Add `score_answer_multi(scorers, query, context, answer, existing_row=None) -> dict`.
   - Modify `build_scorer(scorer_str, max_cost=10.0, cost_guard=None)` to accept a shared CostGuard;
     when None, build one from `max_cost` (preserves Exp 0 callers).
2. `scripts/run_experiment_1.py`
   - Replace `--scorer` with `--scorers` nargs+, default panel.
   - Reject duplicate scorers in argparse-time validation.
   - Build a single shared CostGuard, list of scorers via `build_scorer(s, cost_guard=guard)`.
   - Replace `score_answer` call sites (skip-generation + full run) with `score_answer_multi`.
     For full run, no existing_row; for skip-generation, pass the row dict.
   - Drop bare `faithfulness/relevance/conciseness/quality/scorer_latency_ms` columns;
     replace with per-judge prefixed cols + `consensus_quality`.
   - Cost-limit handling: if `score_answer_multi` returns all-NaN row, treat as cost-limit hit
     (we set a flag inside the helper or check whether all judges NaN'd via CostLimitExceeded).
     Simplest: catch CostLimitExceeded inside `score_answer_multi` per-scorer and set NaN; the
     outer loop checks if guard tripped via `guard.total_estimated_cost`/exception flag.
   - Update report.md generation to use `consensus_quality`. Add per-judge agreement section
     (Pearson r between each pair).
   - Update metadata write to enumerate all judges (with display_name from JUDGE_DISPLAY_NAMES).
3. `scripts/run_experiment_2.py` — mirror Exp 1 changes (chunker checkpoint key stays).
4. `scripts/generate_experiment1_dashboard.py` — switch `quality` → `consensus_quality`.
5. `scripts/generate_experiment2_dashboard.py` — switch `quality` → `consensus_quality`.
6. `tests/test_experiment_utils.py` — already exists? Check; otherwise, adapter test files
   are pre-written in WrittenByDaytime tests/. Copy/symlink tests under `tests/` to keep
   pytest discovery clean.

## Cost limit semantics
- Single shared CostGuard; once it raises CostLimitExceeded for any scorer call, that judge's
  scores become NaN, and `score_answer_multi` continues to call remaining judges (which will
  also raise immediately; their values become NaN too). The outer experiment loop detects
  cost-limit by checking whether ANY scorer raised CostLimitExceeded — easiest approach: have
  `score_answer_multi` return a `cost_limit_hit` flag in a sentinel key, or have the outer
  loop track via try/except wrapping. Simplest: re-raise CostLimitExceeded from
  `score_answer_multi` after attempting all judges in panel for that row, OR expose via a
  module-level inspection. **Decision:** track via guard inspection — outer loop calls
  `score_answer_multi`; if guard's `total_estimated_cost > max_cost`, set cost_limit_hit=True
  and break after writing the row (whose judges naturally NaN'd inside the helper).

## --scorers CLI duplicate check
Validate after argparse: if `len(set(args.scorers)) != len(args.scorers)`, `parser.error()`.
This produces non-zero exit with "error" in stderr — the test checks for "duplicate" or
"repeated" or "unique" so we must include one of those words.

## Dashboards
Replace bare `quality` (col read) with `consensus_quality`. The bare key is also referenced
in helper text — replace inline.

## Tests
Pre-written tests live in `DaytimeNighttimeHandOff/WrittenByDaytime/task-052-multi-judge-exp12/tests/`.
Copy them into `tests/` for pytest discovery (Step 9 copies anyway). Run with the project venv.
