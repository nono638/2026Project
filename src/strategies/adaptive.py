"""Adaptive RAG strategy.

Classifies query complexity first, then routes accordingly:
- Simple (lookup): skip retrieval, answer from model weights
- Moderate (synthesis): single retrieval pass
- Complex (multi_hop): multiple retrieval passes with iterative refinement

Based on: Jeong et al. (2024). Adaptive-RAG. NAACL.

Migrated from src/pipeline/strategies/adaptive.py concept with new
Protocol-conforming interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ollama import Client

if TYPE_CHECKING:
    from src.retriever import Retriever


CLASSIFY_PROMPT = """Classify this question's complexity:
- "simple" if it asks for a single fact or definition
- "moderate" if it requires combining 2-3 pieces of information
- "complex" if it requires multi-step reasoning or comparing multiple concepts

Question: {query}
Answer with just: simple, moderate, or complex"""


class AdaptiveRAG:
    """Adaptive RAG strategy.

    Routes queries to different retrieval depths based on complexity.
    Based on Jeong et al. (2024, NAACL).
    """

    def __init__(self) -> None:
        """Initialize the Ollama client for generation."""
        self._client = Client()

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "adaptive"

    def _classify(self, query: str, model: str) -> str:
        """Classify query complexity as simple, moderate, or complex.

        Args:
            query: The user's question.
            model: Ollama model name.

        Returns:
            One of "simple", "moderate", "complex".
        """
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}],
        ).message.content.strip().lower()

        # Parse classification, default to moderate if unparseable
        for level in ("simple", "moderate", "complex"):
            if level in response:
                return level
        return "moderate"  # Safe middle ground if classification fails

    def _simple_path(self, query: str, model: str) -> str:
        """Generate answer directly without retrieval.

        Args:
            query: The user's question.
            model: Ollama model name.

        Returns:
            The model's generated answer.
        """
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": f"Answer this question:\n{query}\n\nAnswer:"}],
        )
        return response.message.content

    def _moderate_path(self, query: str, retriever: Retriever, model: str) -> str:
        """Standard retrieve + generate (same as NaiveRAG).

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Ollama model name.

        Returns:
            The model's generated answer.
        """
        retrieved = retriever.retrieve(query)
        context = "\n\n".join(r["text"] for r in retrieved)

        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    def _complex_path(self, query: str, retriever: Retriever, model: str) -> str:
        """Two-pass retrieval with iterative refinement.

        First retrieval → intermediate answer → follow-up query →
        second retrieval → final answer combining both contexts.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Ollama model name.

        Returns:
            The model's generated answer.
        """
        # First pass
        retrieved1 = retriever.retrieve(query)
        context1 = "\n\n".join(r["text"] for r in retrieved1)

        intermediate = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": (
                f"Based on this context, give a preliminary answer to the question. "
                f"Note what information might still be missing.\n\n"
                f"Context:\n{context1}\n\n"
                f"Question: {query}\n\n"
                f"Preliminary answer:"
            )}],
        ).message.content

        # Second pass: use intermediate answer to formulate follow-up
        followup = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": (
                f"Based on this preliminary answer, what follow-up question would help "
                f"complete the answer?\n\n"
                f"Original question: {query}\n"
                f"Preliminary answer: {intermediate}\n\n"
                f"Follow-up question:"
            )}],
        ).message.content.strip()

        retrieved2 = retriever.retrieve(followup)
        context2 = "\n\n".join(r["text"] for r in retrieved2)

        # Final answer combining both contexts
        combined_context = f"{context1}\n\n---\n\n{context2}"
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": (
                f"Answer the following question using only the provided context.\n\n"
                f"Context:\n{combined_context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )}],
        )
        return response.message.content

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run Adaptive RAG: classify complexity, then route accordingly.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Ollama model name.

        Returns:
            The model's generated answer.
        """
        complexity = self._classify(query, model)

        if complexity == "simple":
            return self._simple_path(query, model)
        elif complexity == "complex":
            return self._complex_path(query, retriever, model)
        else:
            return self._moderate_path(query, retriever, model)
