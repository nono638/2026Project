"""Tests for HeuristicFilter.

Tests cover length checks, question word detection, copy-paste detection,
deduplication, and edge cases.
"""

from __future__ import annotations

import pytest

from src.document import Document
from src.protocols import QueryFilter
from src.query import Query
from src.query_filters.heuristic import HeuristicFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(text: str, source: str = "Doc") -> Query:
    """Create a minimal Query for testing."""
    return Query(text=text, query_type="factoid", source_doc_title=source)


def _doc(title: str = "Doc", text: str = "Default document text.") -> Document:
    """Create a minimal Document for testing."""
    return Document(title=title, text=text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHeuristicFilter:
    """Tests for the HeuristicFilter class."""

    def test_rejects_too_short(self) -> None:
        """Query with fewer than min_length words is filtered out."""
        f = HeuristicFilter(min_length=5)
        queries = [_q("What is this?")]  # 3 words
        result = f.filter(queries, [_doc()])
        assert len(result) == 0

    def test_rejects_too_long(self) -> None:
        """Query with more than max_length words is filtered out."""
        f = HeuristicFilter(max_length=10)
        long_q = "What is " + " ".join(["word"] * 50) + "?"
        queries = [_q(long_q)]
        result = f.filter(queries, [_doc()])
        assert len(result) == 0

    def test_keeps_normal_length(self) -> None:
        """Query with word count within bounds passes."""
        f = HeuristicFilter(min_length=3, max_length=20)
        queries = [_q("What is the primary function of mitochondria in cells?")]
        result = f.filter(queries, [_doc()])
        assert len(result) == 1

    def test_rejects_no_question_word(self) -> None:
        """Statement without question words or '?' is filtered out."""
        f = HeuristicFilter(min_length=3, max_length=50)
        queries = [_q("The mitochondria powers the cell through ATP production")]
        result = f.filter(queries, [_doc()])
        assert len(result) == 0

    def test_keeps_question_word(self) -> None:
        """Query containing a question word passes."""
        f = HeuristicFilter(min_length=3, max_length=50)
        queries = [_q("What is the powerhouse of the cell?")]
        result = f.filter(queries, [_doc()])
        assert len(result) == 1

    def test_keeps_imperative_with_question_word(self) -> None:
        """Imperative containing 'how' passes (it's a question word)."""
        f = HeuristicFilter(min_length=3, max_length=50)
        queries = [_q("Explain how photosynthesis works in green plants")]
        result = f.filter(queries, [_doc()])
        assert len(result) == 1

    def test_rejects_copy_paste(self) -> None:
        """Query that is mostly copied from the source document is filtered out."""
        doc = _doc(text="Photosynthesis converts light energy into chemical energy in plants")
        f = HeuristicFilter(min_length=3, max_length=50, max_source_overlap=0.5)
        # Query uses most of the same words as the document
        queries = [_q(
            "How does photosynthesis convert light energy into chemical energy?",
            source="Doc",
        )]
        result = f.filter(queries, [doc])
        assert len(result) == 0

    def test_keeps_low_overlap(self) -> None:
        """Query with only a few words from the document passes."""
        doc = _doc(text="Photosynthesis converts light energy into chemical energy in plants")
        f = HeuristicFilter(min_length=3, max_length=50, max_source_overlap=0.8)
        queries = [_q(
            "What role does chlorophyll play in cellular metabolism?",
            source="Doc",
        )]
        result = f.filter(queries, [doc])
        assert len(result) == 1

    def test_deduplication(self) -> None:
        """Near-identical queries are deduplicated."""
        f = HeuristicFilter(min_length=3, max_length=50, similarity_threshold=0.6)
        queries = [
            _q("What is the function of mitochondria in cells?"),
            _q("What is the exact function of mitochondria in cells?"),
        ]
        result = f.filter(queries, [_doc()])
        assert len(result) == 1

    def test_deduplication_preserves_order(self) -> None:
        """The first of two duplicates is kept, not the second."""
        f = HeuristicFilter(min_length=3, max_length=50, similarity_threshold=0.6)
        q1 = _q("What is the function of mitochondria in cells?")
        q2 = _q("What is the exact function of mitochondria in cells?")
        result = f.filter([q1, q2], [_doc()])
        assert len(result) == 1
        assert result[0].text == q1.text

    def test_empty_queries(self) -> None:
        """Empty query list returns empty list."""
        f = HeuristicFilter()
        assert f.filter([], [_doc()]) == []

    def test_missing_source_doc_skips_copy_check(self) -> None:
        """Query referencing unknown doc: copy check skipped, other checks run."""
        f = HeuristicFilter(min_length=3, max_length=50)
        queries = [_q("What is quantum entanglement in physics?", source="Unknown")]
        result = f.filter(queries, [_doc(title="OtherDoc")])
        # Should pass since copy check is skipped and other checks pass
        assert len(result) == 1

    def test_protocol_compliance(self) -> None:
        """HeuristicFilter satisfies the QueryFilter protocol."""
        assert isinstance(HeuristicFilter(), QueryFilter)

    def test_name_format(self) -> None:
        """Name is 'heuristic'."""
        assert HeuristicFilter().name == "heuristic"
