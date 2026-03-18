"""Retriever class wrapping FAISS index + BM25 index + embedder + chunks.

Concrete class, not a Protocol. Strategies receive this; users don't implement it.
Built once per (document, chunker, embedder) triple and cached by the Experiment
runner to avoid redundant embedding work.

Supports three retrieval modes:
- "hybrid" (default): Dense (FAISS) + sparse (BM25) fused with Reciprocal Rank
  Fusion (RRF, k=60). Production standard — NDCG +26-31% over dense-only
  (Blended RAG, IBM 2024).
- "dense": FAISS cosine similarity only (original behavior).
- "sparse": BM25 keyword matching only.

RRF avoids the weight-tuning problem by using rank positions instead of raw scores.
BM25 scores are unbounded while cosine similarity is 0-1, making direct score
combination fragile. RRF sidesteps this entirely.
See: https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf
"""

from __future__ import annotations

import string

import numpy as np
import faiss
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.protocols import Embedder

# Punctuation translation table for BM25 tokenization — built once at import
# time for efficiency. str.translate with str.maketrans is the fastest pure
# Python approach for character-level removal.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_VALID_MODES = ("hybrid", "dense", "sparse")


class Retriever:
    """Wraps a FAISS index + BM25 index + embedder + chunks for retrieval.

    Built once per (document, chunker, embedder) triple and cached
    by the Experiment runner to avoid redundant embedding work.
    """

    def __init__(
        self,
        chunks: list[str],
        embedder: Embedder,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> None:
        """Initialize the retriever, building FAISS and BM25 indexes from chunks.

        Always builds both indexes regardless of mode — negligible cost, and
        simplifies mode switching without reconstruction.

        Args:
            chunks: List of text chunks to index.
            embedder: An Embedder instance for converting text to vectors.
            top_k: Default number of results to return from retrieve().
            mode: Retrieval mode — "hybrid", "dense", or "sparse".

        Raises:
            ValueError: If mode is not one of the valid options.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
            )

        self._chunks = chunks
        self._embedder = embedder
        self._top_k = top_k
        self._mode = mode

        # Handle empty chunk lists gracefully — create empty indexes
        if not chunks:
            self._index = faiss.IndexFlatIP(embedder.dimension)
            self._bm25 = None
            return

        # Build FAISS index with normalized vectors for cosine similarity
        embeddings = embedder.embed(chunks)
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embedder.dimension)
        self._index.add(embeddings)

        # Build BM25 index from tokenized chunks — BM25Okapi is the universal
        # default across Elasticsearch, Lucene, LangChain, and LlamaIndex
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25: lowercase, strip punctuation, split whitespace.

        Simple and deterministic — no external tokenizer dependency. Production
        systems often use more sophisticated tokenization, but for research
        reproducibility and minimal dependencies, simple is better.

        Args:
            text: The text to tokenize.

        Returns:
            List of lowercase tokens with punctuation removed.
        """
        cleaned = text.translate(_PUNCT_TABLE).lower()
        tokens = cleaned.split()
        return [t for t in tokens if t]  # Filter empty strings

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve top-k chunks for a query using the configured mode.

        Args:
            query: The search query text.
            top_k: Number of results to return. Defaults to the value set at init.

        Returns:
            List of dicts with 'text', 'score', 'index' keys,
            sorted by descending score.
        """
        k = top_k or self._top_k

        # Empty index returns no results
        if not self._chunks:
            return []

        if self._mode == "dense":
            return self._retrieve_dense(query, k)
        elif self._mode == "sparse":
            return self._retrieve_sparse(query, k)
        else:
            # hybrid — combine dense and sparse with RRF
            return self._retrieve_hybrid(query, k)

    def _retrieve_dense(self, query: str, top_k: int) -> list[dict]:
        """Retrieve using FAISS dense vector search only.

        Args:
            query: The search query text.
            top_k: Number of results to return.

        Returns:
            List of result dicts sorted by cosine similarity.
        """
        if self._index.ntotal == 0:
            return []

        query_emb = self._embedder.embed([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, min(top_k, self._index.ntotal))

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

    def _retrieve_sparse(self, query: str, top_k: int) -> list[dict]:
        """Retrieve using BM25 keyword matching only.

        Args:
            query: The search query text.
            top_k: Number of results to return.

        Returns:
            List of result dicts sorted by BM25 score.
        """
        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by BM25 score
        ranked_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(bm25_scores[idx])
            results.append({
                "text": self._chunks[idx],
                "score": score,
                "index": int(idx),
            })
        return results

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[dict]:
        """Retrieve using hybrid dense + sparse with RRF fusion.

        Over-retrieves from both sources (up to 100 candidates each) before
        fusing. FAISS IndexFlatIP brute-forces all vectors anyway (no speed
        penalty for larger k). BM25 get_scores() scores all documents. The
        fused top-k has better recall.

        Args:
            query: The search query text.
            top_k: Number of final results to return.

        Returns:
            List of result dicts sorted by RRF score.
        """
        # Over-retrieve candidates from both sources for better recall
        n_candidates = min(len(self._chunks), 100)

        dense_results = self._retrieve_dense(query, n_candidates)
        sparse_results = self._retrieve_sparse(query, n_candidates)

        return self._fuse_rrf(dense_results, sparse_results, top_k)

    def _fuse_rrf(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of dense and sparse results.

        RRF score for document d = sum over retrievers R of: 1 / (k + rank_R(d))
        where rank_R(d) is the 1-based rank of d in retriever R's results.
        Documents not returned by a retriever get no contribution from that retriever.

        k=60 is the default in Elasticsearch 8.8+, Azure AI Search, OpenSearch,
        and Chroma. No weight parameter to tune = no weight to hold constant.

        Args:
            dense_results: Results from dense retrieval, ranked.
            sparse_results: Results from sparse retrieval, ranked.
            top_k: Number of fused results to return.
            k: RRF constant (default 60, industry standard).

        Returns:
            List of result dicts sorted by RRF score.
        """
        # Accumulate RRF scores by chunk index
        rrf_scores: dict[int, float] = {}
        chunk_texts: dict[int, str] = {}

        for rank, result in enumerate(dense_results, start=1):
            idx = result["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
            chunk_texts[idx] = result["text"]

        for rank, result in enumerate(sparse_results, start=1):
            idx = result["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
            chunk_texts[idx] = result["text"]

        # Sort by RRF score descending, take top_k
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        return [
            {
                "text": chunk_texts[idx],
                "score": rrf_scores[idx],
                "index": idx,
            }
            for idx in top_indices
        ]

    @property
    def chunks(self) -> list[str]:
        """Access the underlying chunks (for building context strings)."""
        return self._chunks
