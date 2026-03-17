# Spec: task-012 — Claude Scorer (LLM-as-Judge)

## What

Implement `ClaudeScorer` — a `Scorer` protocol implementation that uses the Anthropic
API to judge RAG-generated answers on three metrics: **faithfulness**, **relevance**,
and **conciseness**. Each metric gets a 1–5 integer score.

This is the project's primary scoring mechanism. The professor specifically flagged
that LLM-as-judge can be noisy, so the implementation should return raw reasoning
alongside scores to support manual validation later.

## Files to Create

### `src/scorers/__init__.py`
```python
"""Scorer implementations."""
from src.scorers.claude import ClaudeScorer

__all__ = ["ClaudeScorer"]
```

### `src/scorers/claude.py`
The main implementation. Structure:

```python
class ClaudeScorer:
    """Scores RAG answers using Claude as an LLM judge.

    Uses the Anthropic API to evaluate answers on faithfulness, relevance,
    and conciseness. Returns integer scores (1-5) plus reasoning text.

    Why Claude as judge: LLM-as-judge is standard practice in RAG evaluation
    (see RAGAS, ARES). Using a stronger model (Claude) to judge weaker models
    (Qwen3/Gemma3) avoids self-evaluation bias.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        """
        Args:
            model: Anthropic model ID. Default to Sonnet for cost efficiency
                   — scoring thousands of answers at Opus prices would be
                   prohibitive. Sonnet is sufficient for structured rubric
                   evaluation.
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY
                     env var (standard Anthropic SDK behavior).
        """

    @property
    def name(self) -> str:
        """Returns 'claude:<model_id>'."""

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        """Score a single answer. Returns dict with keys:
        - faithfulness (1-5): Is the answer supported by the context?
        - relevance (1-5): Does the answer address the query?
        - conciseness (1-5): Is the answer appropriately brief?

        Calls the Anthropic API with a structured prompt and parses
        the response.
        """

    def score_batch(
        self,
        items: list[dict[str, str]],
        max_concurrent: int = 5,
    ) -> list[dict[str, float]]:
        """Score multiple items. Each dict must have keys: query, context, answer.

        Uses sequential calls (not async) with a simple loop. Async would be
        faster but adds complexity for minimal gain at our experiment scale
        (~150-300 items per experiment run).

        Args:
            items: List of dicts with query/context/answer keys.
            max_concurrent: Unused for now (sequential). Reserved for future
                           async implementation.

        Returns:
            List of score dicts in the same order as input.
        """

    def _build_prompt(self, query: str, context: str, answer: str) -> str:
        """Build the scoring prompt. Internal method.

        The prompt must:
        1. Present the context, query, and answer clearly
        2. Define each metric with a rubric (what 1 means, what 5 means)
        3. Ask for JSON output: {"faithfulness": N, "relevance": N, "conciseness": N,
           "reasoning": {"faithfulness": "...", "relevance": "...", "conciseness": "..."}}
        4. Include the instruction to output ONLY valid JSON, no markdown fencing
        """

    def _parse_response(self, response_text: str) -> dict[str, float]:
        """Parse the JSON response from Claude.

        Must handle:
        - Clean JSON
        - JSON wrapped in markdown code fences (```json ... ```)
        - Missing keys (default to 3.0 — middle of scale)
        - Non-numeric values (default to 3.0)

        Returns dict with faithfulness, relevance, conciseness as floats.
        Stores raw reasoning in self._last_reasoning (for debugging/validation).
        """
```

## Scoring Rubric

Use this rubric in the prompt (the night agent should embed this verbatim):

```
Faithfulness (1-5):
  1 = Answer contradicts or fabricates information not in the context
  2 = Answer mostly unsupported, with significant claims beyond context
  3 = Answer partially supported, some claims lack context backing
  4 = Answer well-supported, minor extrapolations only
  5 = Answer entirely grounded in the provided context

Relevance (1-5):
  1 = Answer is completely off-topic
  2 = Answer tangentially related but doesn't address the question
  3 = Answer partially addresses the question
  4 = Answer addresses the question with minor gaps
  5 = Answer directly and completely addresses the question

Conciseness (1-5):
  1 = Extremely verbose, buries the answer in filler
  2 = Noticeably padded with unnecessary content
  3 = Adequate length but could be tighter
  4 = Well-focused with minimal excess
  5 = Precisely as long as needed, no wasted words
```

## Edge Cases

- **Empty answer**: Return `{"faithfulness": 1.0, "relevance": 1.0, "conciseness": 5.0}`.
  An empty answer is unfaithful and irrelevant but technically concise.
- **Empty context**: Score faithfulness as 1.0 (can't be grounded in nothing). Score
  relevance and conciseness normally.
- **API error**: Raise `ScorerError` (define in the module) with the error message.
  Do NOT silently return default scores — the caller needs to know scoring failed.
- **Rate limiting**: The Anthropic SDK handles retries automatically. Do not add
  additional retry logic.

## What NOT to Do

- Do NOT add async support. Sequential is fine for our scale.
- Do NOT cache scores. Each answer is unique.
- Do NOT add a `score_detailed` method that returns reasoning separately — just store
  `self._last_reasoning` for debugging and keep the `score()` interface clean.
- Do NOT import or use the `anthropic` SDK at module level. Import inside `__init__`
  so the module can be imported without the SDK installed (for testing).

## Dependencies

- `anthropic` — the official Anthropic Python SDK. Should already be in requirements.txt
  from the RAGAS task (which installed openai, which shares some deps). If not, install it.

## Rationale

- **Sonnet default over Opus**: Scoring is structured rubric evaluation, not creative
  reasoning. Sonnet is 5x cheaper and fast enough. User can override to Opus if needed.
- **Three metrics, not more**: Faithfulness, relevance, and conciseness are the standard
  triad in RAG evaluation (used by RAGAS, TruLens, LlamaIndex). Adding more would dilute
  the signal at our experiment scale.
- **JSON output, not structured tool use**: Tool use would be more reliable for parsing,
  but adds complexity for a simple 3-field response. JSON with fallback parsing is sufficient.
