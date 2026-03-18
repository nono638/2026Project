"""Naive RAG strategy — the baseline.

Query -> retrieve top-k chunks -> generate answer.
No special logic. Raw model capability is the only variable.

Migrated from src/pipeline/strategies/naive.py. Key change: receives a
Retriever instance instead of raw (chunks, index) pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import LLM
    from src.retriever import Retriever


class NaiveRAG:
    """Naive RAG — the baseline.

    Query -> retrieve top-k chunks -> generate answer.
    No special logic. Raw model capability is the only variable.
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
        return "naive"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run naive RAG: retrieve chunks, generate answer.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Model name (e.g., 'qwen3:0.6b').

        Returns:
            The model's generated answer string.
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
