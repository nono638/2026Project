"""Round-trip consistency filter for generated queries.

Validates that a generated query retrieves its source document/passage when
run through the retriever. Queries that don't retrieve their source are
likely vague, off-topic, or poorly worded.

Based on the Promptagator approach (arxiv:2209.11755). This is one
implementation of the QueryFilter protocol.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.protocols import Chunker, Embedder
from src.retriever import Retriever

if TYPE_CHECKING:
    from src.document import Document
    from src.query import Query


class RoundTripFilter:
    """Filter queries by round-trip retrieval consistency.

    A query passes if its source document appears in the top-k retrieved
    results. This catches vague, off-topic, or poorly worded queries that
    don't actually relate to their claimed source document.

    Implements the QueryFilter protocol from src.protocols.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        """Configure the round-trip filter.

        Args:
            chunker: Used to chunk documents for building the retriever index.
            embedder: Used to embed chunks and queries for similarity search.
            top_k: Number of retrieval results to check for source document.
            min_score: Minimum similarity score threshold (0.0 = any match counts).
        """
        self._chunker = chunker
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score

    @property
    def name(self) -> str:
        """Return filter identifier including embedder name and top_k."""
        return f"round_trip:{self._embedder.name}:k={self._top_k}"

    def filter(
        self,
        queries: list[Query],
        documents: list[Document],
    ) -> list[Query]:
        """Filter queries by checking if retrieval finds their source document.

        Builds a single FAISS index over all document chunks, then for each
        query checks whether any top-k result belongs to the query's claimed
        source document.

        Args:
            queries: List of Query objects to validate.
            documents: Source documents to build the retrieval index from.

        Returns:
            Filtered list of queries whose source document appears in retrieval results.
        """
        if not queries or not documents:
            return []

        # Build chunk-to-document mapping and combined chunk list
        all_chunks: list[str] = []
        chunk_doc_map: list[str] = []  # chunk index → doc title
        doc_titles: set[str] = {doc.title for doc in documents}

        for doc in documents:
            if not doc.text:
                continue
            chunks = self._chunker.chunk(doc.text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_doc_map.append(doc.title)

        if not all_chunks:
            return []

        # Build a single retriever over all chunks
        retriever = Retriever(all_chunks, self._embedder, top_k=self._top_k)

        # Validate each query
        passed: list[Query] = []
        for query in queries:
            # Skip queries referencing unknown documents
            if query.source_doc_title not in doc_titles:
                print(
                    f"RoundTripFilter: skipping query with unknown source doc "
                    f"'{query.source_doc_title}'",
                    file=sys.stderr,
                )
                continue

            results = retriever.retrieve(query.text, top_k=self._top_k)

            # Check if any retrieved chunk belongs to the source document
            source_found = False
            for result in results:
                chunk_idx = result["index"]
                score = result["score"]
                if (
                    chunk_doc_map[chunk_idx] == query.source_doc_title
                    and score >= self._min_score
                ):
                    source_found = True
                    break

            if source_found:
                passed.append(query)

        return passed
