# Plan: task-012 — ClaudeScorer (LLM-as-Judge)

## Files to Modify
- `src/scorers/claude.py` — rewrite to match spec (JSON output, edge cases, batch, error handling)
- `src/scorers/__init__.py` — add docstring, keep existing export

## Approach
1. Rewrite `src/scorers/claude.py` to match spec exactly:
   - Move `import anthropic` inside `__init__` (lazy import for testability)
   - Add `api_key` parameter to constructor
   - Add `ScorerError` exception class
   - Replace line-based prompt/parsing with JSON-based approach
   - Add `_build_prompt()` with verbatim rubric from spec
   - Add `_parse_response()` with code-fence and fallback handling
   - Add `score_batch()` (sequential loop)
   - Add edge case handling: empty answer (1/1/5), empty context (faithfulness=1)
   - Store reasoning in `self._last_reasoning`
2. Update `__init__.py` with docstring per spec
3. Run pre-written tests

## Ambiguities
- The existing implementation uses a text-based response format; spec requires JSON. Will switch to JSON.
- Existing `__init__` has no `api_key` param; spec requires it. Will add.
- The Scorer protocol only defines `name` and `score()`, not `score_batch()`. Adding `score_batch()` as an extra method is fine — structural subtyping still works.
