# Plan: task-021 — Consolidate Nighttime Tests

## Files to modify
- `tests/test_google_embedders.py` — replace with task-016 migration tests (old SDK tests are broken)
- Copy 7 new test files into `tests/`

## Files to create
- `tests/test_hotpotqa_loader.py` (from task-013, with class renames)
- `tests/test_experiment_timing.py` (from task-014, no conflicts)
- `tests/test_squad_loader.py` (from task-015, with class renames)
- `tests/test_llm_scorer.py` (from task-017)
- `tests/test_experiment_zero.py` (from task-018)
- `tests/test_hybrid_retrieval.py` (from task-019, with class rename)
- `tests/test_llm_protocol.py` (from task-020)
- `pytest.ini` — add testpaths config

## Skip
- task-012 `test_claude_scorer.py` — ClaudeScorer no longer exists, replaced by LLMScorer in task-017. Tests are redundant.

## Class renames (to avoid pytest collection conflicts)
- `test_hotpotqa_loader.py`: TestDocumentFormat → TestHotpotqaDocumentFormat, TestEdgeCases → TestHotpotqaEdgeCases, TestCompatibility → TestHotpotqaCompatibility
- `test_squad_loader.py`: TestDocumentFormat → TestSquadDocumentFormat, TestEdgeCases → TestSquadEdgeCases, TestCompatibility → TestSquadCompatibility
- `test_hybrid_retrieval.py`: TestEdgeCases → TestHybridRetrievalEdgeCases

## Task-016 embedder migration
- Replace `test_google_embedders.py` with task-016's `test_embedder_migration.py` content
- The old file uses defunct `genai.embed_content` API; new file uses `genai.Client` pattern
- Merge protocol compliance test from old file into new file

## Approach
1. Copy files, apply renames
2. Replace google embedder test file
3. Create pytest.ini with testpaths = tests
4. Run tests to verify
