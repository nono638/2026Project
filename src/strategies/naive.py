"""Naive RAG strategy — the baseline.

Query -> retrieve top-k chunks -> generate answer.
No special logic. Raw model capability is the only variable.

Migrated from src/pipeline/strategies/naive.py. Key change: receives a
Retriever instance instead of raw (chunks, index) pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ollama import Client

if TYPE_CHECKING:
    from src.retriever import Retriever


class NaiveRAG:
    """Naive RAG — the baseline.

    Query -> retrieve top-k chunks -> generate answer.
    No special logic. Raw model capability is the only variable.
    """

    def __init__(self) -> None:
        """Initialize the Ollama client for generation."""
        self._client = Client()

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "naive"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run naive RAG: retrieve chunks, generate answer.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Ollama model name (e.g., 'qwen3:0.6b').

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

        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
