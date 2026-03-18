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

if TYPE_CHECKING:
    from src.protocols import LLM
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

    def __init__(self, llm: LLM) -> None:
        """Initialize with an LLM backend for generation.

        Args:
            llm: An LLM instance for text generation.
        """
        self._llm = llm

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "adaptive"

    def _classify(self, query: str, model: str) -> str:
        """Classify query complexity as simple, moderate, or complex.

        Args:
            query: The user's question.
            model: Model name for generation.

        Returns:
            One of "simple", "moderate", "complex".
        """
        response = self._llm.generate(
            model, CLASSIFY_PROMPT.format(query=query)
        ).strip().lower()

        # Parse classification, default to moderate if unparseable
        for level in ("simple", "moderate", "complex"):
            if level in response:
                return level
        return "moderate"  # Safe middle ground if classification fails

    def _simple_path(self, query: str, model: str) -> str:
        """Generate answer directly without retrieval.

        Args:
            query: The user's question.
            model: Model name for generation.

        Returns:
            The model's generated answer.
        """
        return self._llm.generate(
            model, f"Answer this question:\n{query}\n\nAnswer:"
        )

    def _moderate_path(self, query: str, retriever: Retriever, model: str) -> str:
        """Standard retrieve + generate (same as NaiveRAG).

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name for generation.

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

        return self._llm.generate(model, prompt)

    def _complex_path(self, query: str, retriever: Retriever, model: str) -> str:
        """Two-pass retrieval with iterative refinement.

        First retrieval → intermediate answer → follow-up query →
        second retrieval → final answer combining both contexts.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name for generation.

        Returns:
            The model's generated answer.
        """
        # First pass
        retrieved1 = retriever.retrieve(query)
        context1 = "\n\n".join(r["text"] for r in retrieved1)

        intermediate = self._llm.generate(
            model,
            f"Based on this context, give a preliminary answer to the question. "
            f"Note what information might still be missing.\n\n"
            f"Context:\n{context1}\n\n"
            f"Question: {query}\n\n"
            f"Preliminary answer:",
        )

        # Second pass: use intermediate answer to formulate follow-up
        followup = self._llm.generate(
            model,
            f"Based on this preliminary answer, what follow-up question would help "
            f"complete the answer?\n\n"
            f"Original question: {query}\n"
            f"Preliminary answer: {intermediate}\n\n"
            f"Follow-up question:",
        ).strip()

        retrieved2 = retriever.retrieve(followup)
        context2 = "\n\n".join(r["text"] for r in retrieved2)

        # Final answer combining both contexts
        combined_context = f"{context1}\n\n---\n\n{context2}"
        return self._llm.generate(
            model,
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:",
        )

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run Adaptive RAG: classify complexity, then route accordingly.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name for generation.

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
