# Result: task-003 — Implement Remaining Strategies, Chunkers, and Embedders
**Status:** done
**Completed:** 2026-03-16T23:00:00

## Commits
- See branch `night/task-003-remaining-components`

## Test Results
- Command run: `.venv/Scripts/python -m pytest tests/test_core.py tests/test_components.py -v`
- Outcome: 29 passed, 0 failed
- Failures: none

## Decisions Made
- Installed sentence-transformers (and its dependencies: torch, transformers, etc.) into the venv for HuggingFaceEmbedder. Updated requirements.txt with pinned versions.
- Strategy tests requiring Ollama are not included as automated tests since Ollama is a network service. Protocol compliance is tested instead.
- FixedSizeChunker uses word-based splitting (not token-based) as specified in the spec.
- CorrectiveRAG merges chunks from both the original and reformulated queries to maximize context quality.
- AdaptiveRAG defaults to "moderate" when classification fails to parse, as spec requires.

## Flags for Morning Review
- sentence-transformers adds ~2GB of dependencies (torch). If disk space is a concern, consider making it optional.
- Strategy integration tests (marked @pytest.mark.slow in spec) not yet written — need Ollama running to test.

## Attempted Approaches (if skipped/blocked)
n/a
