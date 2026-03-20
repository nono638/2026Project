"""MiniLM cross-encoder reranker — lightweight (22M params), CPU-fast.

Uses ms-marco-MiniLM-L-6-v2, the most widely used lightweight open-source
reranker. Trained on MS MARCO passage ranking, generalizes well to other
domains.

Model card: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
Benchmark: https://research.aimultiple.com/rerankers/
"""

from __future__ import annotations

import math


class MiniLMReranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 (22M params).

    Loads the model lazily on first rerank() call. The model downloads
    automatically from HuggingFace Hub (~25MB) on first use.
    """

    def __init__(self) -> None:
        """Initialize the MiniLM reranker (model loads lazily)."""
        self._model = None

    def _load_model(self) -> None:
        """Load the CrossEncoder model on first use.

        Lazy loading avoids the ~25MB download until actually needed,
        and keeps import time fast for tests that only check protocol compliance.
        """
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @property
    def name(self) -> str:
        """Unique identifier for this reranker."""
        return "minilm:ms-marco-MiniLM-L-6-v2"

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank retrieved chunks by cross-encoder relevance to query.

        Builds (query, chunk_text) pairs, scores with the cross-encoder,
        applies sigmoid to normalize logits to 0-1, and sorts by rerank_score.

        Args:
            query: The search query text.
            chunks: List of dicts with 'text', 'score', 'index' keys.
            top_k: If provided, return only top_k results. None returns all.

        Returns:
            List of dicts sorted by rerank_score descending, each containing
            'text', 'score' (original), 'rerank_score', and 'index'.
        """
        if not chunks:
            return []

        if top_k is not None and top_k <= 0:
            return []

        if self._model is None:
            self._load_model()

        # Build query-passage pairs for cross-encoder scoring
        pairs = [(query, chunk["text"]) for chunk in chunks]
        raw_scores = self._model.predict(pairs)

        # Apply sigmoid to normalize raw logits to 0-1 range
        # Same pattern as CrossEncoderFilter (src/query_filters/cross_encoder.py)
        # Makes scores comparable across models and interpretable as features
        results = []
        for chunk, raw_score in zip(chunks, raw_scores):
            results.append({
                "text": chunk["text"],
                "score": chunk["score"],
                "rerank_score": _sigmoid(float(raw_score)),
                "index": chunk["index"],
            })

        # Sort by rerank_score descending
        results.sort(key=lambda r: r["rerank_score"], reverse=True)

        # Truncate to top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results


def _sigmoid(x: float) -> float:
    """Apply sigmoid function to convert logit to 0-1 probability.

    Args:
        x: Raw logit value from cross-encoder.

    Returns:
        Probability in [0, 1].
    """
    return 1.0 / (1.0 + math.exp(-x))
