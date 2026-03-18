# Result: task-022 — CLI flags for run_experiment.py
**Status:** done
**Completed:** 2026-03-18T01:43:10

## Commits
- `<pending>` — night: task-022 expose CLI flags in run_experiment.py

## Test Results
- Command run: `python scripts/run_experiment.py --help` + validation tests
- Outcome: All 6 new arguments parse correctly, validation catches invalid inputs, --help shows all options with examples
- Also ran full test suite: 205 passed, 5 failed (pre-existing), 7 errors (pre-existing google embedder tests on main)

## Decisions Made
- **sample_hotpotqa/sample_squad require load first:** The spec pseudo-code showed `sample_hotpotqa(n=n, seed=42)` but the actual functions take `(documents, queries, n, seed)`. Implemented correct two-step load-then-sample pattern.
- **Experiment constructor missing scorer:** The original script did not pass a `scorer` to `Experiment()`, which is now a required parameter. This is a pre-existing issue not in scope for this task. Left as-is.
- **Original script used `corpus=` kwarg:** The original passed `corpus=corpus_dicts` to `Experiment()` which isn't in the current `__init__`. Changed to use `load_corpus()` method which is the correct API.
- **_load_builtin_dataset returns 3 values:** Returns `(corpus_dicts, query_dicts, queries)` where the third value is the raw Query objects for saving to JSON for reproducibility.
- **Quick mode chunker:** Changed from `SemanticChunker` to `RecursiveChunker` per spec (project default baseline).

## Flags for Morning Review
- The `Experiment()` constructor requires `scorer` parameter, but the script doesn't configure one. This was also true before this task. A scorer will need to be added before the script is runnable end-to-end.
- The quick mode default chunker changed from semantic to recursive — intentional per spec.
