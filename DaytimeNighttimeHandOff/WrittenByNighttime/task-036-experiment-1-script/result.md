# Result: task-036 — Experiment 1 Script + experiment_utils.py
**Status:** done
**Completed:** 2026-03-21T01:47:39

## Commits
- (pending commit on branch)

## Test Results
- Command run: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-036-experiment-1-script/tests/ -v`
- Outcome: 33 passed, 0 failed (23 experiment_utils + 10 experiment_1)
- Command run: `pytest tests/ --ignore=DaytimeNighttimeHandOff/ -v`
- Outcome: 527 passed, 0 failed
- Failures: none

## Decisions Made
- `load_hotpotqa_examples` returns (docs, queries) tuple instead of list of dicts as spec suggested, because that's what the actual dataset API returns and downstream code needs.
- `generate_answer` accepts strategy/chunker/embedder objects rather than constructing them internally, to allow the main loop to reuse instances across queries.
- Scores are stored flat (faithfulness, relevance, conciseness, quality) rather than with scorer name prefix as in Exp 0, since Exp 1 uses a single scorer.

## Flags for Morning Review
None.

## Attempted Approaches (if skipped/blocked)
N/A
