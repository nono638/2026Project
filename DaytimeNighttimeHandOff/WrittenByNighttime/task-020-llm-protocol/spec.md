# task-020: LLM Protocol — abstract generation backend (Ollama + OpenAI-compatible)

## Summary

All 5 strategies hardcode `ollama.Client()` for text generation. This refactor introduces
an `LLM` protocol and two adapters (Ollama, OpenAI-compatible) so strategies accept any
generation backend. The OpenAI-compatible adapter covers LM Studio, vLLM, llama.cpp server,
and actual OpenAI — all of which speak the same API format.

The refactor is mechanical: every strategy currently does
`self._client.chat(model, messages).message.content`. After the refactor, they do
`self._llm.generate(model, prompt)`. The `Strategy` protocol signature is unchanged
(`run(query, retriever, model) -> str`).

## Requirements

1. Add an `LLM` protocol to `src/protocols.py` with `name` property and
   `generate(model: str, prompt: str) -> str` method.
2. Create `src/llms/` package with two adapters:
   - `OllamaLLM` — wraps `ollama.Client().chat()`
   - `OpenAICompatibleLLM` — wraps `openai.OpenAI(base_url=...).chat.completions.create()`
3. All 5 strategies accept `llm: LLM` as a required `__init__` parameter. Replace
   `self._client = Client()` with `self._llm = llm`. Replace every
   `self._client.chat(model=model, messages=[...]).message.content` with
   `self._llm.generate(model, prompt)`.
4. Update `scripts/run_experiment.py` `build_components()` to create an `OllamaLLM()`
   instance and pass it to each strategy constructor.
5. The `Strategy` protocol in `src/protocols.py` is **unchanged** — `run(query, retriever,
   model) -> str` stays the same.
6. `LLM` protocol is `@runtime_checkable` (consistent with all other protocols in the file).
7. All existing tests continue to pass. Mock strategies in tests don't use the real
   `ollama.Client()` so they are unaffected.

## Files to Create

- `src/llms/__init__.py` — exports `OllamaLLM`, `OpenAICompatibleLLM`
- `src/llms/ollama.py` — `OllamaLLM` adapter
- `src/llms/openai_compat.py` — `OpenAICompatibleLLM` adapter

## Files to Modify

- `src/protocols.py` — Add `LLM` protocol (after `Scorer`, before `QueryGenerator`).
  Pattern:
  ```python
  @runtime_checkable
  class LLM(Protocol):
      """Interface for text generation backends (Ollama, LM Studio, OpenAI, etc.)."""

      @property
      def name(self) -> str:
          """Backend identifier (e.g., 'ollama', 'openai-compat:localhost:1234')."""
          ...

      def generate(self, model: str, prompt: str) -> str:
          """Generate a text response for the given prompt.

          Args:
              model: Model identifier (e.g., 'qwen3:4b' for Ollama,
                     'local-model' for LM Studio).
              prompt: The complete prompt text.

          Returns:
              The model's generated text response.
          """
          ...
  ```
- `src/strategies/naive.py` — Replace `from ollama import Client` with
  `from src.protocols import LLM`. Change `__init__` to accept `llm: LLM`. Replace
  `self._client = Client()` with `self._llm = llm`. Replace chat call (line 57-60) with
  `return self._llm.generate(model, prompt)`.
- `src/strategies/multi_query.py` — Same pattern. Two `self._client.chat()` calls to
  replace (one for query rephrasing, one for final answer).
- `src/strategies/corrective.py` — Same pattern. Multiple calls: per-chunk relevance
  rating loop, query reformulation, final answer generation.
- `src/strategies/self_rag.py` — Same pattern. Calls: retrieval decision, direct
  generation, per-chunk relevance evaluation, answer generation, self-critique.
- `src/strategies/adaptive.py` — Same pattern. Calls: complexity classification, plus
  1-4 generation calls depending on path.
- `src/strategies/__init__.py` — Add re-export of `OllamaLLM` and `OpenAICompatibleLLM`
  from `src.llms` for convenience.
- `scripts/run_experiment.py` — In `build_components()` (line 126+): create
  `llm = OllamaLLM()` once, pass to each strategy constructor:
  `strategies.append(strategy_map[name](llm=llm))` instead of
  `strategies.append(strategy_map[name]())`.

## New Dependencies

None — `ollama` and `openai` are both already installed in requirements.txt.

## Edge Cases

- **Ollama not running**: `OllamaLLM.generate()` will raise a connection error. Same
  behavior as current code — strategies don't catch Ollama exceptions. Let it propagate.
- **LM Studio not running**: `OpenAICompatibleLLM.generate()` will raise a connection
  error. Same treatment.
- **Model name mismatch**: If the user passes an Ollama model name to an OpenAI-compatible
  backend (or vice versa), the backend will return an error. Not our problem — the user
  configures model names to match their backend.
- **Empty/None response**: If the model returns an empty string, pass it through. Strategies
  and the scorer will handle it (or flag it as low quality).
- **OpenAI API key for LM Studio**: LM Studio doesn't validate API keys. Default to
  `api_key="lm-studio"` (convention). For actual OpenAI, user passes their real key.

## Decisions Made

- **`generate(model, prompt) -> str`**, not `chat(model, messages)`: Every strategy passes
  a single user message. None use multi-turn conversation. `generate()` is simpler and
  hides the chat/messages format difference between Ollama and OpenAI. If multi-turn is
  ever needed, add a `chat()` method later. **Why:** Minimal interface that covers 100% of
  current usage.
- **`llm` is a required parameter**, not optional with a default: Forces explicitness about
  which backend is in use. The project isn't public yet; all call sites are known and will
  be updated. **Why:** Avoid hidden Ollama dependency in strategy constructors.
- **Two adapters only (Ollama + OpenAI-compatible)**: OpenAI-compatible covers LM Studio,
  vLLM, llama.cpp server, and actual OpenAI — all same API format. Anthropic and Google
  cloud APIs are not needed for experiments and can be added later as separate adapters.
  **Why:** Maximum coverage with minimum code.
- **`name` property on LLM**: Consistent with every other protocol (Chunker, Embedder,
  Strategy, Scorer all have `name`). Useful for logging.
- **No temperature/options parameters**: No strategy currently passes temperature, top_k,
  or other generation options. Don't add what isn't needed. Can be added later via
  `**kwargs` or an options dict if needed.

## Implementation Details

### OllamaLLM (src/llms/ollama.py)

```python
from ollama import Client
from src.protocols import LLM

class OllamaLLM:
    """Ollama generation backend.

    Wraps ollama.Client().chat() into the LLM protocol.
    Default host: http://localhost:11434 (Ollama default).
    """

    def __init__(self, host: str | None = None) -> None:
        """Initialize the Ollama client.

        Args:
            host: Ollama server URL. None uses the default localhost:11434.
        """
        self._client = Client(host=host) if host else Client()

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, model: str, prompt: str) -> str:
        """Generate via Ollama chat API."""
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
```

### OpenAICompatibleLLM (src/llms/openai_compat.py)

```python
from openai import OpenAI

class OpenAICompatibleLLM:
    """OpenAI-compatible generation backend.

    Works with LM Studio, vLLM, llama.cpp server, and actual OpenAI.
    All speak the same chat completions API format.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            base_url: API endpoint URL. Defaults to LM Studio's default.
                      For vLLM: "http://localhost:8000/v1"
                      For OpenAI: "https://api.openai.com/v1"
            api_key: API key. LM Studio/vLLM don't validate this.
                     For actual OpenAI, pass your real key.
        """
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._base_url = base_url

    @property
    def name(self) -> str:
        return f"openai-compat:{self._base_url}"

    def generate(self, model: str, prompt: str) -> str:
        """Generate via OpenAI chat completions API."""
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

### Strategy refactor pattern (applied identically to all 5)

Before:
```python
from ollama import Client

class NaiveRAG:
    def __init__(self) -> None:
        self._client = Client()

    def run(self, query, retriever, model):
        # ... build prompt ...
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
```

After:
```python
from src.protocols import LLM

class NaiveRAG:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, query, retriever, model):
        # ... build prompt (unchanged) ...
        return self._llm.generate(model, prompt)
```

For strategies with multiple generation calls (CorrectiveRAG, SelfRAG, AdaptiveRAG),
apply the same replacement to EVERY `self._client.chat()` call site:
- `self._client.chat(model=model, messages=[...]).message.content`
  becomes `self._llm.generate(model, prompt_text)`
- Where `prompt_text` is whatever string was in the message content.
- Some strategies do `.message.content.strip().lower()` — keep the `.strip().lower()` part,
  just change the generation call: `self._llm.generate(model, prompt).strip().lower()`

### run_experiment.py changes

In `build_components()`:
```python
from src.llms import OllamaLLM

# Create LLM backend once
llm = OllamaLLM()

# Pass to each strategy
strategies.append(strategy_map[name](llm=llm))
```

## What NOT to Touch

- **`src/protocols.py` Strategy protocol**: `run(query, retriever, model) -> str` is
  unchanged. The `model` parameter stays as a string passed per-call.
- **`src/experiment.py`**: The experiment runner doesn't know about LLMs. It calls
  `strategy.run()` and that's it. No changes needed.
- **`src/retriever.py`**: Completely independent of generation.
- **`src/scorers/`**: Scorers have their own API clients. LLM protocol is for generation
  only.
- **Mock strategies in tests**: Tests that use MockStrategy (test_experiment_timing.py,
  test_e2e_smoke.py, etc.) create their own mock classes. They don't construct real
  strategies. No changes needed.

## Testing Approach

- Pre-written tests in `tests/test_llm_protocol.py` (see tests/ directory in this task)
- Tests cover:
  - LLM protocol compliance for both adapters (runtime_checkable)
  - OllamaLLM instantiation and interface
  - OpenAICompatibleLLM instantiation and interface
  - Strategy constructors accept LLM parameter
  - Strategy.run() delegates to LLM.generate() (mock LLM)
  - All 5 strategies can be constructed with a mock LLM
- API calls are mocked — tests don't require Ollama or LM Studio running
- Run with: `pytest tests/test_llm_protocol.py -v`
