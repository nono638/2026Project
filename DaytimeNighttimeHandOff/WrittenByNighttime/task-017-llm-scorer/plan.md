# Plan: task-017 — Refactor ClaudeScorer → LLMScorer

## Files to Create
- `src/scorers/llm.py` — new LLMScorer with adapter pattern

## Files to Modify
- `src/scorers/__init__.py` — replace ClaudeScorer with LLMScorer + ScorerError exports
- `src/protocols.py` — update Scorer.name docstring example
- `src/scorers/claude.py` — delete (replaced by llm.py)

## Approach
1. Create `src/scorers/llm.py` carrying over all logic from claude.py
2. Add adapter factory pattern: _anthropic_adapter and _google_adapter
3. Replace ClaudeScorer class with LLMScorer(provider, model, api_key)
4. Update __init__.py exports
5. Update protocols.py docstring
6. Run tests

## Ambiguities
- None — spec includes full architecture and adapter examples.
