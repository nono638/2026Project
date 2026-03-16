# Plan: task-002 — Migrate Existing Components to Pluggable Interfaces

## Files to Create
1. `src/chunkers/__init__.py` — exports SemanticChunker
2. `src/chunkers/semantic.py` — wraps LangChain SemanticChunker, implements Chunker protocol
3. `src/embedders/__init__.py` — exports OllamaEmbedder
4. `src/embedders/ollama.py` — wraps Ollama embedding, implements Embedder protocol
5. `src/scorers/__init__.py` — exports ClaudeScorer
6. `src/scorers/claude.py` — wraps Claude-as-judge, implements Scorer protocol
7. `src/strategies/__init__.py` — exports NaiveRAG, SelfRAG
8. `src/strategies/naive.py` — wraps naive RAG, implements Strategy protocol
9. `src/strategies/self_rag.py` — wraps Self-RAG flow, implements Strategy protocol

## Files to Delete
- `src/config.py`
- `src/data/__init__.py`, `src/data/generate.py`, `src/data/features.py`
- `src/pipeline/__init__.py`, `src/pipeline/chunking.py`, `src/pipeline/retrieval.py`, `src/pipeline/scoring.py`
- `src/pipeline/strategies/__init__.py`, `src/pipeline/strategies/naive.py`, `src/pipeline/strategies/self_rag.py`
- `src/pipeline/strategies/multi_query.py`, `src/pipeline/strategies/corrective.py`, `src/pipeline/strategies/adaptive.py`
- Empty directories: `src/pipeline/strategies/`, `src/pipeline/`, `src/data/`

## Approach
1. Create all new package files from spec (matching existing logic exactly)
2. Delete old files and directories
3. Run tests/test_core.py to verify no regressions
4. Verify imports work

## Ambiguities
- None significant — spec is very explicit about what goes where.
