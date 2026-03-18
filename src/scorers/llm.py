"""LLMScorer — provider-agnostic LLM-as-judge scoring.

Scores RAG answers using any supported LLM backend on three metrics:
faithfulness, relevance, and conciseness (1-5 each).

Provider adapters are factory functions that return callables. Each factory
creates the API client once (expensive), and the returned callable sends
prompts and returns response text (cheap). Adding a new provider requires
only writing a new ~15-line adapter function.

Why LLM-as-judge: standard practice in RAG evaluation (see RAGAS, ARES,
TruLens). Using a stronger model to judge weaker models avoids
self-evaluation bias.
Methodology: Saad-Falcon et al. (2024, NAACL) showed LLM-as-judge
correlates well with human judgment for RAG evaluation.

Refactored from ClaudeScorer in task-017 to support multiple providers
(Anthropic, Google) without duplicating scoring logic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ScorerError(Exception):
    """Raised when scoring fails (API errors, invalid responses, etc.)."""


# Rubric embedded verbatim from the spec — this is the single source of truth
# for how scores should be assigned.
_SCORING_RUBRIC = """\
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
  5 = Precisely as long as needed, no wasted words"""

_METRICS = ("faithfulness", "relevance", "conciseness")


# ---------------------------------------------------------------------------
# Provider adapters — each factory returns a callable(prompt) -> str
# ---------------------------------------------------------------------------

def _anthropic_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    """Create an Anthropic API caller.

    Lazy-imports the anthropic SDK so users who only use Google
    don't need it installed.

    Args:
        model: Anthropic model ID (e.g., "claude-sonnet-4-20250514").
        api_key: API key, or None to use ANTHROPIC_API_KEY env var.

    Returns:
        A callable that takes a prompt string and returns the response text.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def call(prompt: str) -> str:
        """Send prompt to Anthropic and return response text."""
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def _google_adapter(model: str, api_key: str | None) -> Callable[[str], str]:
    """Create a Google GenAI API caller.

    Lazy-imports the google-genai SDK so users who only use Anthropic
    don't need it installed.

    Args:
        model: Google model ID (e.g., "gemini-2.5-flash").
        api_key: API key, or None to use GOOGLE_API_KEY env var.

    Returns:
        A callable that takes a prompt string and returns the response text.
    """
    from google import genai

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def call(prompt: str) -> str:
        """Send prompt to Google GenAI and return response text."""
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    return call


# Registry of provider name → adapter factory function
_ADAPTERS: dict[str, Callable[[str, str | None], Callable[[str], str]]] = {
    "anthropic": _anthropic_adapter,
    "google": _google_adapter,
}


def _get_adapter(provider: str, model: str, api_key: str | None) -> Callable[[str], str]:
    """Look up and instantiate the adapter for the given provider.

    Args:
        provider: Provider name (e.g., "anthropic", "google").
        model: Model ID to pass to the adapter factory.
        api_key: API key to pass to the adapter factory.

    Returns:
        A callable that sends prompts and returns response text.

    Raises:
        ScorerError: If the provider is not in the registry.
    """
    factory = _ADAPTERS.get(provider)
    if factory is None:
        raise ScorerError(
            f"Unknown provider '{provider}'. Supported: {list(_ADAPTERS.keys())}"
        )
    return factory(model, api_key)


# ---------------------------------------------------------------------------
# LLMScorer
# ---------------------------------------------------------------------------

class LLMScorer:
    """Provider-agnostic LLM-as-judge scorer.

    Scores RAG answers on faithfulness, relevance, and conciseness (1-5)
    using any supported LLM backend. The scoring rubric, prompt template,
    and response parsing are shared across all providers — only the API
    call differs.

    Implements the Scorer protocol from src.protocols.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        """Initialize with provider and model.

        Both provider and model are required — no defaults. This prevents
        accidentally burning API credits on the wrong provider.

        Args:
            provider: LLM provider name ("anthropic" or "google").
            model: Model ID (e.g., "claude-sonnet-4-20250514", "gemini-2.5-flash").
            api_key: API key. If None, the provider SDK resolves from env vars.

        Raises:
            ScorerError: If the provider is unknown or client initialization fails.
        """
        self._provider = provider
        self._model = model
        self._last_reasoning: dict[str, str] | None = None

        try:
            self._call_llm = _get_adapter(provider, model, api_key)
        except ScorerError:
            raise  # Re-raise unknown provider errors directly
        except Exception as exc:
            raise ScorerError(
                f"Failed to initialize {provider} client: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        """Return '<provider>:<model>' identifier."""
        return f"{self._provider}:{self._model}"

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        """Score a single answer.

        Returns dict with keys:
        - faithfulness (1-5): Is the answer supported by the context?
        - relevance (1-5): Does the answer address the query?
        - conciseness (1-5): Is the answer appropriately brief?

        Args:
            query: The original question.
            context: The source document text (ground truth).
            answer: The model's generated answer.

        Returns:
            Dict with 'faithfulness', 'relevance', 'conciseness' scores (1-5).

        Raises:
            ScorerError: If the API call fails.
        """
        # Edge case: empty answer is unfaithful and irrelevant but technically concise
        if not answer or not answer.strip():
            self._last_reasoning = None
            return {"faithfulness": 1.0, "relevance": 1.0, "conciseness": 5.0}

        prompt = self._build_prompt(query, context, answer)

        try:
            text = self._call_llm(prompt)
        except Exception as exc:
            raise ScorerError(
                f"{self._provider} API call failed: {exc}"
            ) from exc

        scores = self._parse_response(text)

        # Edge case: empty context means faithfulness can't be assessed —
        # override to 1.0 regardless of what the model returned.
        if not context or not context.strip():
            scores["faithfulness"] = 1.0

        return scores

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
        return [
            self.score(
                query=item["query"],
                context=item["context"],
                answer=item["answer"],
            )
            for item in items
        ]

    def _build_prompt(self, query: str, context: str, answer: str) -> str:
        """Build the scoring prompt.

        The prompt presents the context, query, and answer clearly, defines
        each metric with a rubric, and asks for JSON output. Shared across
        all providers — the rubric is the same regardless of which LLM judges.

        Args:
            query: The original question.
            context: The source document text.
            answer: The model's generated answer.

        Returns:
            The complete prompt string.
        """
        return f"""\
You are evaluating the quality of an answer generated by a language model.

**Query:**
{query}

**Context (ground truth source):**
{context}

**Model's answer:**
{answer}

Rate the answer on the following three dimensions using this rubric:

{_SCORING_RUBRIC}

Respond with ONLY valid JSON (no markdown fencing, no explanation outside the JSON). Use this exact structure:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "conciseness": <int 1-5>, "reasoning": {{"faithfulness": "<brief explanation>", "relevance": "<brief explanation>", "conciseness": "<brief explanation>"}}}}"""

    def _parse_response(self, response_text: str) -> dict[str, float]:
        """Parse the JSON response from the LLM.

        Handles clean JSON, JSON wrapped in markdown code fences,
        missing keys (default to 3.0), and non-numeric values (default to 3.0).

        Stores raw reasoning in self._last_reasoning for debugging/validation.

        Args:
            response_text: Raw text response from the LLM.

        Returns:
            Dict with faithfulness, relevance, conciseness as floats.
        """
        text = response_text.strip()

        # Strip markdown code fences if present — LLMs sometimes wrap JSON
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails entirely, return defaults (middle of scale)
            self._last_reasoning = None
            return {m: 3.0 for m in _METRICS}

        # Extract and store reasoning for debugging/validation
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            self._last_reasoning = {k: str(v) for k, v in reasoning.items()}
        else:
            self._last_reasoning = None

        # Extract scores with fallbacks
        scores: dict[str, float] = {}
        for metric in _METRICS:
            raw = data.get(metric, 3.0)
            try:
                scores[metric] = float(raw)
            except (ValueError, TypeError):
                # Non-numeric value — default to middle of scale
                scores[metric] = 3.0

        return scores
