"""ClaudeScorer — LLM-as-judge scoring using Claude.

Scores RAG answers using the Anthropic API on three metrics:
faithfulness, relevance, and conciseness (1-5 each).

Why Claude as judge: LLM-as-judge is standard practice in RAG evaluation
(see RAGAS, ARES, TruLens). Using a stronger model (Claude) to judge weaker
models (Qwen3/Gemma3) avoids self-evaluation bias.
Methodology: Saad-Falcon et al. (2024, NAACL) showed LLM-as-judge
correlates well with human judgment for RAG evaluation.

Known limitation: if Claude is also used for query classification,
both features and labels carry Claude's biases. Flagged for future work.
"""

from __future__ import annotations

import json
import re
from typing import Any


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
    ) -> None:
        """Initialize with the Claude model to use for judging.

        Args:
            model: Anthropic model ID. Default to Sonnet for cost efficiency
                   — scoring thousands of answers at Opus prices would be
                   prohibitive. Sonnet is sufficient for structured rubric
                   evaluation.
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY
                     env var (standard Anthropic SDK behavior).
        """
        # Import inside __init__ so the module can be imported without the SDK
        # installed (e.g., during testing with mocked anthropic module).
        import anthropic  # noqa: F811 — intentional lazy import for testability

        self._model = model
        self._last_reasoning: dict[str, str] | None = None
        try:
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        except Exception as exc:
            raise ScorerError(f"Failed to initialize Anthropic client: {exc}") from exc

    @property
    def name(self) -> str:
        """Returns 'claude:<model_id>'."""
        return f"claude:{self._model}"

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
            response = self._client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        except Exception as exc:
            raise ScorerError(f"Anthropic API call failed: {exc}") from exc

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
        each metric with a rubric, and asks for JSON output.

        Args:
            query: The original question.
            context: The source document text.
            answer: The model's generated answer.

        Returns:
            The complete prompt string for Claude.
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
        """Parse the JSON response from Claude.

        Handles clean JSON, JSON wrapped in markdown code fences,
        missing keys (default to 3.0), and non-numeric values (default to 3.0).

        Stores raw reasoning in self._last_reasoning for debugging/validation.

        Args:
            response_text: Raw text response from Claude.

        Returns:
            Dict with faithfulness, relevance, conciseness as floats.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails entirely, return defaults
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
