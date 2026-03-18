"""Protocol definitions for pluggable RAG research tool components.

Uses typing.Protocol with @runtime_checkable for all interfaces.
Protocols over ABCs because: structural subtyping (duck typing) lets users
write classes that happen to have the right methods without importing or
inheriting anything from the framework. A research tool should have minimal
boilerplate — users just implement the methods and pass instances directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from src.document import Document
    from src.query import Query
    from src.retriever import Retriever


@runtime_checkable
class Chunker(Protocol):
    """Interface for document chunking strategies."""

    @property
    def name(self) -> str:
        """Unique identifier for this chunker config (e.g., 'semantic:mxbai-embed-large')."""
        ...

    def chunk(self, text: str) -> list[str]:
        """Split document text into chunks."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Interface for text embedding backends."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'ollama:mxbai-embed-large')."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension. May be detected lazily on first embed() call."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns array of shape (len(texts), dimension)."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """Interface for RAG strategy implementations."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'naive', 'self_rag')."""
        ...

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Execute the RAG strategy and return the generated answer.

        Args:
            query: The user's question.
            retriever: A Retriever instance (wraps chunks + index + embedder).
            model: Ollama model name for generation (e.g., 'qwen3:0.6b').
        """
        ...


@runtime_checkable
class Scorer(Protocol):
    """Interface for answer scoring backends."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'anthropic:claude-sonnet-4-20250514')."""
        ...

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        """Score a generated answer. Returns dict of metric_name -> score (1-5)."""
        ...


@runtime_checkable
class QueryGenerator(Protocol):
    """Interface for evaluation query generation backends.

    Implementations produce Query objects from Document objects. Multiple
    generators exist by design — RAGAS, templates, human-curated, BEIR —
    because single-source evaluation is a methodological weakness.
    """

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'ragas:gpt-4o-mini', 'template:factoid')."""
        ...

    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
        """Generate evaluation queries from documents.

        Args:
            documents: List of Document objects from src.document.
            queries_per_doc: Target number of queries to generate per document.

        Returns:
            List of Query objects from src.query.
        """
        ...


@runtime_checkable
class QueryFilter(Protocol):
    """Interface for query validation/filtering backends.

    Filters validate generated queries and remove low-quality ones.
    Implementations include round-trip consistency, cross-encoder scoring,
    and LLM-based judgment.
    """

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'round_trip:k=5', 'llm_judge')."""
        ...

    def filter(
        self,
        queries: list[Query],
        documents: list[Document],
    ) -> list[Query]:
        """Filter queries, returning only those that pass validation.

        Args:
            queries: List of Query objects to validate.
            documents: The source documents (needed for context-dependent filters
                       like round-trip retrieval).

        Returns:
            Filtered list of Query objects.
        """
        ...
