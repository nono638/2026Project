# task-017: Refactor ClaudeScorer → LLMScorer with provider adapters

## Summary
Replace the Claude-specific `ClaudeScorer` with a provider-agnostic `LLMScorer` class
that supports multiple LLM backends (Anthropic and Google initially) via an adapter
pattern. Users configure the provider and model at construction time. The scoring rubric,
prompt, parsing, and edge case handling are shared across all providers. This makes
RAGBench's scoring layer usable with cheaper APIs (Gemini) and extensible to others
(OpenAI, Ollama) in the future.

## Requirements
1. `LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")` works identically
   to the current `ClaudeScorer`.
2. `LLMScorer(provider="google", model="gemini-2.5-flash")` works with the Google GenAI
   API using `google-genai` SDK.
3. The `score()` method signature is unchanged: `score(query, context, answer) -> dict[str, float]`
   returning `{"faithfulness": float, "relevance": float, "conciseness": float}`.
4. The `score_batch()` method works identically across providers.
5. The `name` property returns `"<provider>:<model>"` (e.g., `"anthropic:claude-sonnet-4-20250514"`,
   `"google:gemini-2.5-flash"`).
6. The scoring rubric, prompt template, and response parsing are shared — not duplicated
   per provider.
7. Adding a new provider requires only writing a new adapter function (~15 lines), not
   modifying core scoring logic.
8. `ClaudeScorer` is removed entirely — no backwards-compatibility alias.
9. All existing tests pass with updated imports/mocks.

## Files to Modify
- `src/scorers/claude.py` — **delete** (or rename to `src/scorers/llm.py`)
- `src/scorers/llm.py` — **create**. The new LLMScorer class with adapter pattern.
- `src/scorers/__init__.py` — **modify**. Replace `ClaudeScorer` export with `LLMScorer`.
  Also export `ScorerError`.
- `src/protocols.py` — **modify** (docstring only). Update the `Scorer.name` docstring
  example from `'claude:claude-sonnet-4-20250514'` to `'anthropic:claude-sonnet-4-20250514'`.

## Files to Read for Context
- `src/scorers/claude.py` — the code being refactored (keep all logic, restructure)
- `src/protocols.py` — Scorer protocol interface

## New Dependencies
None — `anthropic` and `google-genai` are both already installed.

## Architecture

```
LLMScorer
├── __init__(provider, model, api_key)
│   └── self._call_llm = _get_adapter(provider, model, api_key)
├── score(query, context, answer)        # shared
│   ├── _build_prompt(query, context, answer)  # shared
│   ├── self._call_llm(prompt)                 # provider-specific
│   └── _parse_response(text)                  # shared
├── score_batch(items)                   # shared
└── name → "<provider>:<model>"          # shared pattern

Adapters (module-level functions):
├── _anthropic_adapter(model, api_key) → callable(prompt) → str
├── _google_adapter(model, api_key) → callable(prompt) → str
└── (future: _openai_adapter, _ollama_adapter)
```

Each adapter is a **factory function** that returns a callable. The factory creates the
client once; the returned callable sends a prompt and returns the response text.

```python
def _anthropic_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    return call

def _google_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    from google import genai
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    def call(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    return call
```

The `_get_adapter` dispatcher:
```python
_ADAPTERS = {
    "anthropic": _anthropic_adapter,
    "google": _google_adapter,
}

def _get_adapter(provider: str, model: str, api_key: str | None) -> Callable[[str], str]:
    factory = _ADAPTERS.get(provider)
    if factory is None:
        raise ScorerError(
            f"Unknown provider '{provider}'. Supported: {list(_ADAPTERS.keys())}"
        )
    return factory(model, api_key)
```

## API Key Resolution
- **Anthropic**: `api_key` parameter → `ANTHROPIC_API_KEY` env var (SDK default behavior)
- **Google**: `api_key` parameter → `GOOGLE_API_KEY` or `GEMINI_API_KEY` env var (SDK default behavior)
- If no key is found, the SDK itself raises an error. Let it propagate — don't add extra validation.

## Edge Cases
- **Empty answer**: return `{"faithfulness": 1.0, "relevance": 1.0, "conciseness": 5.0}` —
  same as current. Handled in shared `score()` before calling adapter.
- **Empty context**: override faithfulness to 1.0 after scoring — same as current.
- **Unknown provider**: raise `ScorerError("Unknown provider 'xxx'. Supported: [...]")`.
- **API failure**: raise `ScorerError(f"{provider} API call failed: {exc}")`.
- **Malformed JSON response**: return defaults `{m: 3.0 for m in metrics}` — same as current.
- **JSON wrapped in markdown fences**: strip fences before parsing — same as current.

## Decisions Made
- **Adapter factory pattern**: each adapter is a factory that returns a callable. **Why:**
  the client is created once (expensive), the callable is invoked per-score (cheap). Clean
  separation without class hierarchy overhead.
- **No LiteLLM dependency**: write small adapters directly. **Why:** LiteLLM is heavyweight
  (~100+ transitive deps) for what's currently 2 providers. When/if we need 5+, reconsider.
- **No backwards-compatible ClaudeScorer alias**: the project hasn't released anything yet
  — no external users to break. **Why:** dead aliases create confusion.
- **Lazy imports inside adapters**: `import anthropic` and `from google import genai` happen
  inside the adapter factory, not at module level. **Why:** users who only use one provider
  shouldn't need the other's SDK installed. Same pattern ClaudeScorer already uses.
- **Default provider/model**: no defaults — both `provider` and `model` are required params.
  **Why:** forcing explicit choice prevents accidentally burning API credits on the wrong
  provider.
- **Gemini default model**: `gemini-2.5-flash` for the spec examples. **Why:** stable,
  cheapest, best price-performance. Not going anywhere (unlike 2.0-flash which retires
  June 2026).

## What NOT to Touch
- `src/protocols.py` — Scorer protocol interface is unchanged (except one docstring update)
- `src/experiment.py` — the experiment runner calls `scorer.score()` which is unchanged
- `src/embedders/` — embedders are a separate concern
- Any test files outside this task's test directory

## Testing Approach
- Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-017-llm-scorer/tests/test_llm_scorer.py`
- Mock both `anthropic.Anthropic` and `genai.Client` — no real API calls
- Test classes:
  - `TestAnthropicProvider` — score, score_batch, name property, API error handling
  - `TestGoogleProvider` — score, score_batch, name property, API error handling
  - `TestSharedBehavior` — empty answer, empty context, malformed JSON, fenced JSON,
    unknown provider error
  - `TestAdapterPattern` — adapters are lazy-imported, adapter returns callable
- Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-017-llm-scorer/tests/`
