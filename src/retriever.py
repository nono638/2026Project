"""Retriever class wrapping FAISS index + embedder + chunks.

Concrete class, not a Protocol. Strategies receive this; users don't implement it.
Built once per (document, chunker, embedder) triple and cached by the Experiment
runner to avoid redundant embedding work.
"""

from __future__ import annotations

import numpy as np
import faiss

from src.protocols import Embedder


class Retriever:
    """Wraps a FAISS index + embedder + chunks for retrieval.

    Built once per (document, chunker, embedder) triple and cached
    by the Experiment runner to avoid redundant embedding work.
    """

    def __init__(self, chunks: list[str], embedder: Embedder, top_k: int = 5) -> None:
        """Initialize the retriever, building the FAISS index from chunks.

        Args:
            chunks: List of text chunks to index.
            embedder: An Embedder instance for converting text to vectors.
            top_k: Default number of results to return from retrieve().
        """
        self._chunks = chunks
        self._embedder = embedder
        self._top_k = top_k

        # Handle empty chunk lists gracefully — create an empty index
        if not chunks:
            self._index = faiss.IndexFlatIP(embedder.dimension)
            return

        # Build FAISS index with normalized vectors for cosine similarity
        embeddings = embedder.embed(chunks)
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embedder.dimension)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve top-k chunks for a query.

        Args:
            query: The search query text.
            top_k: Number of results to return. Defaults to the value set at init.

        Returns:
            List of dicts with 'text', 'score', 'index' keys,
            sorted by descending similarity.
        """
        k = top_k or self._top_k

        # Empty index returns no results
        if self._index.ntotal == 0:
            return []

        query_emb = self._embedder.embed([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self._chunks[idx],
                "score": float(score),
                "index": int(idx),
            })
        return results

    @property
    def chunks(self) -> list[str]:
        """Access the underlying chunks (for building context strings)."""
        return self._chunks
