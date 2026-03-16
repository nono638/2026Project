# Plan: task-003 — Implement Remaining Components

## Files to Create
1. `src/strategies/multi_query.py` — MultiQueryRAG
2. `src/strategies/corrective.py` — CorrectiveRAG
3. `src/strategies/adaptive.py` — AdaptiveRAG
4. `src/chunkers/fixed.py` — FixedSizeChunker
5. `src/chunkers/recursive.py` — RecursiveChunker
6. `src/chunkers/sentence.py` — SentenceChunker
7. `src/embedders/huggingface.py` — HuggingFaceEmbedder
8. `tests/test_components.py` — Tests for new components

## Files to Modify
1. `src/chunkers/__init__.py` — add new chunker exports
2. `src/embedders/__init__.py` — add HuggingFaceEmbedder export
3. `src/strategies/__init__.py` — add new strategy exports

## Approach
1. Install sentence-transformers for HuggingFaceEmbedder
2. Create all new component files as specified
3. Update __init__.py re-exports
4. Write tests (chunker tests don't need Ollama, strategy tests marked @pytest.mark.slow)
5. Run non-slow tests

## Ambiguities
- sentence-transformers needs to be installed — network blocked but pip install from cache may work
- If sentence-transformers install fails, will create the file anyway and mark HuggingFaceEmbedder tests as skippable
