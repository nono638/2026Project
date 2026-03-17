"""Cross-encoder relevance filter for generated queries.

Scores each (query, source passage) pair using a cross-encoder reranker model.
More precise than bi-encoder round-trip filtering because cross-encoders jointly
encode the query and passage, modeling token-level interactions.

Based on the InPars-v2 filtering methodology (Bonifacio et al., 2023,
arxiv:2301.01820). Key finding: cross-encoder filtering improved downstream
quality more than generating more queries.

Uses a small local model (~100M params) — no API calls required.
Ref: MonoT5 reranker architecture (Nogueira et al., 2020, arxiv:2003.06713).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from src.document import Document
    from src.query import Query


class CrossEncoderFilter:
    """Filters queries by cross-encoder relevance scoring against source documents.

    Uses a small cross-encoder model to score (query, passage) pairs and filters
    queries below a relevance threshold. More precise than bi-encoder round-trip
    because cross-encoders model token-level query-passage interactions.

    Args:
        model_name: Cross-encoder model identifier. Default is MS MARCO MiniLM,
            a small (~25MB) model trained on query-passage relevance.
        threshold: Minimum relevance score (0-1 after sigmoid) to keep a query.
        use_full_doc: If True, score against full document text. If False,
            split into paragraphs and take max score — more precise for queries
            about specific sections.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.5,
        use_full_doc: bool = False,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._use_full_doc = use_full_doc
        self._model = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        """Identifier including model name and threshold (e.g., 'cross_encoder:ms-marco-MiniLM-L-6-v2:t=0.5')."""
        short_name = self._model_name.split("/")[-1]
        return f"cross_encoder:{short_name}:t={self._threshold}"

    def filter(
        self,
        queries: list[Query],
        documents: list[Document],
    ) -> list[Query]:
        """Filter queries by cross-encoder relevance score against source documents.

        Args:
            queries: List of queries to filter.
            documents: List of documents to match queries against.

        Returns:
            List of queries that pass the relevance threshold.
        """
        if not queries or not documents:
            return []

        doc_lookup = {doc.title: doc for doc in documents}
        kept: list[Query] = []

        for query in queries:
            doc = doc_lookup.get(query.source_doc_title)
            if doc is None:
                print(
                    f"WARNING: Source doc '{query.source_doc_title}' not found, "
                    f"discarding query: {query.text[:50]}",
                    file=sys.stderr,
                )
                continue

            score = self._score_query(query.text, doc.text)
            if score >= self._threshold:
                kept.append(query)

        return kept

    def _score_query(self, query_text: str, doc_text: str) -> float:
        """Score a (query, document) pair.

        Args:
            query_text: The query string.
            doc_text: The document text.

        Returns:
            Relevance score between 0 and 1 (sigmoid of model logit).
        """
        if self._use_full_doc:
            # Score against full document — model will truncate if too long
            scores = self._model.predict([(query_text, doc_text)])
            # Handle scalar, non-empty array, and empty array cases
            if hasattr(scores, '__len__') and len(scores) > 0:
                return self._sigmoid(float(scores[0]))
            elif hasattr(scores, '__len__'):
                return 0.0  # Empty result — no score available
            else:
                return self._sigmoid(float(scores))
        else:
            # Split into paragraphs and take max score
            paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [doc_text]

            pairs = [(query_text, p) for p in paragraphs]
            scores = self._model.predict(pairs)
            max_score = max(self._sigmoid(float(s)) for s in scores)
            return max_score

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid function to convert logit to 0-1 probability.

        Args:
            x: Raw logit value.

        Returns:
            Probability between 0 and 1.
        """
        import math
        return 1.0 / (1.0 + math.exp(-x))
