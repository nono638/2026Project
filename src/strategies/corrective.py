"""Corrective RAG strategy.

After retrieval, scores each chunk's relevance and discards low-scoring ones.
If too many are discarded, triggers a fresh retrieval with a reformulated query.
Addresses small model hallucination from irrelevant context.

Based on: Shi et al. (2024). Corrective RAG. arXiv:2401.15884.

Migrated from src/pipeline/strategies/corrective.py concept with new
Protocol-conforming interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import LLM
    from src.retriever import Retriever


RELEVANCE_PROMPT = """Rate how relevant this passage is to answering the question.
Question: {query}
Passage: {chunk}
Rate as "relevant", "partially relevant", or "irrelevant". Answer with just the rating."""

REFORMULATE_PROMPT = """The following question did not find good matches in the document.
Reformulate it to be more specific or use different terminology.
Original question: {query}
Reformulated question:"""


class CorrectiveRAG:
    """Corrective RAG strategy.

    Retrieves, filters by relevance, and reformulates the query if too
    many chunks are discarded. Based on Shi et al. (2024, arXiv:2401.15884).
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
        return "corrective"

    def _filter_relevant(self, query: str, retrieved: list[dict], model: str) -> list[str]:
        """Filter retrieved chunks by prompting the model for relevance ratings.

        Args:
            query: The user's question.
            retrieved: List of retrieval result dicts with 'text' key.
            model: Model name for generation.

        Returns:
            List of chunk texts rated as "relevant" or "partially relevant".
        """
        relevant: list[str] = []
        for r in retrieved:
            rating = self._llm.generate(
                model, RELEVANCE_PROMPT.format(query=query, chunk=r["text"])
            ).strip().lower()

            # Keep anything that isn't explicitly "irrelevant"
            if "irrelevant" not in rating:
                relevant.append(r["text"])
        return relevant

    def run(
        self,
        query: str,
        retriever: Retriever,
        model: str,
        diagnostics: dict | None = None,
    ) -> str:
        """Run Corrective RAG: retrieve, filter, optionally reformulate, generate.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name for generation.
            diagnostics: Optional dict populated with pipeline internals.

        Returns:
            The model's generated answer.
        """
        # Step 1: Retrieve
        retrieved = retriever.retrieve(query)
        all_retrieved = list(retrieved)  # Track all chunks across rounds
        queries_used = [query]

        # Step 2-3: Filter by relevance
        relevant_chunks = self._filter_relevant(query, retrieved, model)

        # Step 4: If fewer than 2 chunks survive, reformulate and retry
        if len(relevant_chunks) < 2:
            reformulated = self._llm.generate(
                model, REFORMULATE_PROMPT.format(query=query)
            ).strip()
            queries_used.append(reformulated)

            # Second retrieval with reformulated query
            retrieved2 = retriever.retrieve(reformulated)
            all_retrieved.extend(retrieved2)
            relevant_chunks2 = self._filter_relevant(reformulated, retrieved2, model)

            # Merge both sets of relevant chunks
            relevant_chunks = relevant_chunks + relevant_chunks2

        # If still empty after reformulation, use top 2 original chunks as fallback
        if not relevant_chunks:
            relevant_chunks = [r["text"] for r in retrieved[:2]]

        context = "\n\n".join(relevant_chunks)

        if diagnostics is not None:
            diagnostics["retrieved_chunks"] = all_retrieved
            diagnostics["filtered_chunks"] = relevant_chunks
            diagnostics["context_sent_to_llm"] = context
            diagnostics["retrieval_queries"] = queries_used
            diagnostics["skipped_retrieval"] = False

        # Step 5: Generate answer
        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return self._llm.generate(model, prompt)
