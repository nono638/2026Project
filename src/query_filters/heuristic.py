"""Heuristic pre-filters for generated queries.

Fast, zero-cost validation that catches degenerate queries before more
expensive filters (round-trip, cross-encoder) run. Standard practice
across InPars (arxiv:2202.05144), Promptagator (arxiv:2209.11755),
and RAGAS pipelines.

Catches ~10-20% of synthetically generated queries: garbled text,
copy-paste from source, duplicates, non-questions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.document import Document
    from src.query import Query

# Common English stopwords to exclude from overlap calculations.
# Kept small and hardcoded to avoid a dependency on nltk or spaCy.
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "with", "by", "from", "as", "it",
    "this", "that",
}

# Words that indicate a query intent (question or directive)
_QUESTION_WORDS: set[str] = {
    "who", "what", "where", "when", "why", "how", "which",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercased word tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(tokens: list[str]) -> set[str]:
    """Remove stopwords from a token list, returning a set of content words.

    Args:
        tokens: List of lowercased word tokens.

    Returns:
        Set of tokens with stopwords removed.
    """
    return {t for t in tokens if t not in _STOPWORDS}


class HeuristicFilter:
    """Fast heuristic pre-filter for generated queries.

    Implements the QueryFilter protocol from src.protocols. Applies length,
    question-word, copy-detection, and deduplication checks in order.
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 50,
        require_question_mark: bool = False,
        max_source_overlap: float = 0.8,
        deduplicate: bool = True,
        similarity_threshold: float = 0.9,
    ) -> None:
        """Configure the heuristic filter thresholds.

        Args:
            min_length: Minimum query length in words. Below = degenerate.
            max_length: Maximum query length in words. Above = likely copied passage.
            require_question_mark: If True, reject queries not ending with "?".
            max_source_overlap: Max fraction of query content words also in
                source document. Above = likely copy-paste.
            deduplicate: If True, remove near-duplicate queries.
            similarity_threshold: Jaccard similarity threshold for deduplication.
        """
        self._min_length = min_length
        self._max_length = max_length
        self._require_question_mark = require_question_mark
        self._max_source_overlap = max_source_overlap
        self._deduplicate = deduplicate
        self._similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        """Return filter identifier."""
        return "heuristic"

    def filter(
        self,
        queries: list[Query],
        documents: list[Document],
    ) -> list[Query]:
        """Apply heuristic checks to filter out low-quality queries.

        Checks applied in order: length, question word, copy detection,
        question mark (optional). Then deduplication if enabled.

        Args:
            queries: List of Query objects to validate.
            documents: Source documents for copy detection.

        Returns:
            Filtered list of queries that pass all checks.
        """
        if not queries:
            return []

        # Build document lookup for copy detection
        doc_texts: dict[str, str] = {doc.title: doc.text for doc in documents}

        passed: list[Query] = []
        for query in queries:
            tokens = _tokenize(query.text)
            word_count = len(tokens)

            # Length check
            if word_count < self._min_length or word_count > self._max_length:
                continue

            # Question word check: must contain a question word OR end with "?"
            has_question_word = bool(_QUESTION_WORDS & set(tokens))
            ends_with_question = query.text.strip().endswith("?")
            if not has_question_word and not ends_with_question:
                continue

            # Copy detection: high overlap with source document = copy-paste
            source_text = doc_texts.get(query.source_doc_title)
            if source_text is not None:
                query_content = _content_words(tokens)
                if query_content:  # Avoid division by zero
                    doc_tokens = _tokenize(source_text)
                    doc_content = _content_words(doc_tokens)
                    overlap = len(query_content & doc_content) / len(query_content)
                    if overlap > self._max_source_overlap:
                        continue

            # Question mark check (optional)
            if self._require_question_mark and not ends_with_question:
                continue

            passed.append(query)

        # Deduplication: greedy, preserves order (first instance kept)
        if self._deduplicate and len(passed) > 1:
            passed = self._deduplicate_queries(passed)

        return passed

    def _deduplicate_queries(self, queries: list[Query]) -> list[Query]:
        """Remove near-duplicate queries by Jaccard similarity on word sets.

        Greedy approach: iterate in order, compare each query against all
        previously kept queries. First instance wins, duplicates discarded.

        Args:
            queries: Pre-filtered list of queries.

        Returns:
            Deduplicated list preserving original order.
        """
        kept: list[Query] = []
        kept_word_sets: list[set[str]] = []

        for query in queries:
            words = set(_tokenize(query.text))
            is_duplicate = False
            for prev_words in kept_word_sets:
                # Jaccard similarity: |A∩B| / |A∪B|
                union = words | prev_words
                if not union:
                    continue
                jaccard = len(words & prev_words) / len(union)
                if jaccard > self._similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(query)
                kept_word_sets.append(words)

        return kept
