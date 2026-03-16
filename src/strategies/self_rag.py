"""Self-RAG strategy — prompted approximation.

The most complex strategy. The model is prompted to:
1. Decide if retrieval is needed
2. Evaluate retrieved chunk relevance
3. Generate an answer
4. Critique whether its answer is well-supported

This is a prompted approximation of Asai et al. (2024), not a fine-tuned
Self-RAG model — a practical constraint of running local models via Ollama.

Hypothesis: highest ceiling but also highest capability floor. May hurt
very small models (0.6B, 1.7B) because self-critique requires reasoning
about reasoning.

Migrated from src/pipeline/strategies/self_rag.py. Key change: receives a
Retriever instance instead of raw (chunks, index) pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ollama import Client

if TYPE_CHECKING:
    from src.retriever import Retriever


RETRIEVAL_DECISION_PROMPT = """Given this question, do you need to look up external information to answer it well?
Question: {query}
Answer only "yes" or "no"."""

RELEVANCE_PROMPT = """Rate how relevant this passage is to answering the question.
Question: {query}
Passage: {chunk}
Rate as "relevant", "partially relevant", or "irrelevant". Answer with just the rating."""

GENERATE_PROMPT = """Answer the question using only the provided context.

Context:
{context}

Question: {query}

Answer:"""

CRITIQUE_PROMPT = """You just generated this answer to a question based on provided context.

Question: {query}
Context: {context}
Your answer: {answer}

Is your answer well-supported by the context? If not, revise it. If it is, return it unchanged.

Final answer:"""


class SelfRAG:
    """Self-RAG: decide, retrieve, evaluate, generate, critique.

    Implements the Strategy protocol from src.protocols.
    """

    def __init__(self) -> None:
        """Initialize the Ollama client for generation."""
        self._client = Client()

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "self_rag"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Run Self-RAG: decide, retrieve, evaluate, generate, critique.

        Args:
            query: The user's question.
            retriever: A Retriever instance for chunk retrieval.
            model: Ollama model name.

        Returns:
            The model's final answer after self-critique.
        """
        # Step 1: Does the model think it needs retrieval?
        decision = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": RETRIEVAL_DECISION_PROMPT.format(query=query)}],
        ).message.content.strip().lower()

        if "no" in decision:
            # Generate without retrieval
            response = self._client.chat(
                model=model,
                messages=[{"role": "user", "content": f"Answer this question:\n{query}\n\nAnswer:"}],
            )
            return response.message.content

        # Step 2: Retrieve
        retrieved = retriever.retrieve(query)

        # Step 3: Evaluate relevance of each chunk
        relevant_chunks: list[str] = []
        for r in retrieved:
            rating = self._client.chat(
                model=model,
                messages=[{"role": "user", "content": RELEVANCE_PROMPT.format(
                    query=query, chunk=r["text"]
                )}],
            ).message.content.strip().lower()

            if "irrelevant" not in rating:
                relevant_chunks.append(r["text"])

        # If all chunks filtered out, use top 2 anyway
        if not relevant_chunks:
            relevant_chunks = [r["text"] for r in retrieved[:2]]

        context = "\n\n".join(relevant_chunks)

        # Step 4: Generate
        answer = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": GENERATE_PROMPT.format(
                context=context, query=query
            )}],
        ).message.content

        # Step 5: Self-critique
        final = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": CRITIQUE_PROMPT.format(
                query=query, context=context, answer=answer
            )}],
        ).message.content

        return final
