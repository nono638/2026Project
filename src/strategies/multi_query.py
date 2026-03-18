"""Multi-Query RAG strategy.

Instead of retrieving once with the original question, generates several
different phrasings, retrieves for each, then merges results before generating.
Compensates for weak query understanding in small models by casting a wider
retrieval net.

Migrated from src/pipeline/strategies/multi_query.py concept with new
Protocol-conforming interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import LLM
    from src.retriever import Retriever


REPHRASE_PROMPT = """Generate 3 alternative phrasings of this question. Each should ask the same thing
in a different way. Return only the 3 questions, one per line, no numbering.

Question: {query}"""


class MultiQueryRAG:
    """Multi-Query RAG strategy.

    Generates alternative phrasings of the query, retrieves for each,
    merges results (deduplicating by chunk index, keeping max score),
    then generates an answer from the merged context.
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
        return "multi_query"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run Multi-Query RAG: rephrase, multi-retrieve, merge, generate.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name for generation.

        Returns:
            The model's generated answer.
        """
        # Step 1: Generate alternative phrasings
        rephrase_response = self._llm.generate(
            model, REPHRASE_PROMPT.format(query=query)
        )

        alt_queries = [
            line.strip()
            for line in rephrase_response.strip().split("\n")
            if line.strip()
        ]

        # Edge case: if model fails to generate distinct phrasings, use original only
        if not alt_queries:
            alt_queries = []

        all_queries = [query] + alt_queries[:3]  # Original + up to 3 alternatives

        # Step 2: Retrieve for each query
        # Deduplicate by chunk index, keeping the max score
        merged: dict[int, dict] = {}
        for q in all_queries:
            results = retriever.retrieve(q)
            for r in results:
                idx = r["index"]
                if idx not in merged or r["score"] > merged[idx]["score"]:
                    merged[idx] = r

        # Step 3: Re-rank by score (descending)
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        context = "\n\n".join(r["text"] for r in ranked[:5])

        # Step 4: Generate answer
        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return self._llm.generate(model, prompt)
