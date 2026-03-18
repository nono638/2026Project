# Plan: task-020 — LLM Protocol

## Approach

Mechanical refactor: introduce `LLM` protocol, create two adapters, then update all 5
strategies to accept `llm: LLM` instead of hardcoding `ollama.Client()`.

## Files to Create

1. `src/llms/__init__.py` — exports OllamaLLM, OpenAICompatibleLLM
2. `src/llms/ollama.py` — OllamaLLM wrapping ollama.Client().chat()
3. `src/llms/openai_compat.py` — OpenAICompatibleLLM wrapping openai.OpenAI()

## Files to Modify

1. `src/protocols.py` — Add `LLM` protocol (after Scorer, before QueryGenerator)
2. `src/strategies/naive.py` — Replace `from ollama import Client` → accept `llm: LLM`
3. `src/strategies/multi_query.py` — Same pattern (2 chat calls)
4. `src/strategies/corrective.py` — Same pattern (multiple chat calls)
5. `src/strategies/self_rag.py` — Same pattern (5+ chat calls)
6. `src/strategies/adaptive.py` — Same pattern (4+ chat calls)
7. `src/strategies/__init__.py` — Add re-export of OllamaLLM, OpenAICompatibleLLM
8. `scripts/run_experiment.py` — Create OllamaLLM(), pass to strategy constructors

## Strategy Refactor Pattern

Each strategy:
- Remove `from ollama import Client`
- Add `from src.protocols import LLM` (TYPE_CHECKING import)
- Change `__init__(self)` → `__init__(self, llm: LLM)`
- Replace `self._client = Client()` → `self._llm = llm`
- Replace `self._client.chat(model=model, messages=[{"role": "user", "content": X}]).message.content`
  → `self._llm.generate(model, X)`
- Keep any `.strip().lower()` post-processing after generate()

## Ambiguities

None — spec is very prescriptive with exact code patterns.
