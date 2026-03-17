"""Tests for CrossEncoderFilter — cross-encoder relevance scoring.

Uses the real cross-encoder model (~25MB, downloaded on first run).
No mocking needed — the model is small and local.
"""

from __future__ import annotations

import pytest

from src.document import Document
from src.protocols import QueryFilter
from src.query import Query
from src.query_filters.cross_encoder import CrossEncoderFilter


# Shared test data
PARIS_DOC = Document(
    title="Paris",
    text="Paris is the capital of France. It is known for the Eiffel Tower, "
    "the Louvre museum, and its rich cultural history. The city sits on the "
    "Seine river and has a population of about 2 million people.",
)

MULTI_PARA_DOC = Document(
    title="Science",
    text=(
        "Physics is the study of matter and energy.\n\n"
        "Chemistry deals with substances and their interactions.\n\n"
        "Biology is the study of living organisms and their processes."
    ),
)


class TestCrossEncoderFilter:
    """Tests for CrossEncoderFilter with real model."""

    def test_keeps_relevant_query(self) -> None:
        """A relevant query about Paris should pass the filter."""
        f = CrossEncoderFilter(threshold=0.5)
        queries = [
            Query(
                text="What is the capital of France?",
                query_type="factoid",
                source_doc_title="Paris",
            ),
        ]
        result = f.filter(queries, [PARIS_DOC])
        assert len(result) == 1

    def test_rejects_irrelevant_query(self) -> None:
        """An unrelated query should be filtered out."""
        f = CrossEncoderFilter(threshold=0.5)
        queries = [
            Query(
                text="How do you bake a chocolate cake?",
                query_type="factoid",
                source_doc_title="Paris",
            ),
        ]
        result = f.filter(queries, [PARIS_DOC])
        assert len(result) == 0

    def test_threshold_tuning(self) -> None:
        """Different thresholds should produce different filter results."""
        queries = [
            Query(
                text="What is the capital of France?",
                query_type="factoid",
                source_doc_title="Paris",
            ),
        ]
        # Very high threshold — may filter even relevant queries
        f_high = CrossEncoderFilter(threshold=0.999)
        result_high = f_high.filter(queries, [PARIS_DOC])

        # Very low threshold — should keep everything
        f_low = CrossEncoderFilter(threshold=0.01)
        result_low = f_low.filter(queries, [PARIS_DOC])

        assert len(result_low) >= len(result_high)

    def test_chunked_vs_full_doc(self) -> None:
        """Chunked scoring should find relevant paragraph better than full doc."""
        # Query about biology, which is in the 3rd paragraph
        query = Query(
            text="What is the study of living organisms?",
            query_type="factoid",
            source_doc_title="Science",
        )

        # Chunked mode should score higher (max over paragraphs picks biology paragraph)
        f_chunked = CrossEncoderFilter(threshold=0.5, use_full_doc=False)
        f_full = CrossEncoderFilter(threshold=0.5, use_full_doc=True)

        # Both should process without error
        result_chunked = f_chunked.filter([query], [MULTI_PARA_DOC])
        result_full = f_full.filter([query], [MULTI_PARA_DOC])

        # Chunked should be at least as permissive as full doc
        assert len(result_chunked) >= len(result_full)

    def test_empty_queries(self) -> None:
        """filter([], docs) returns []."""
        f = CrossEncoderFilter()
        assert f.filter([], [PARIS_DOC]) == []

    def test_empty_documents(self) -> None:
        """filter(queries, []) returns []."""
        f = CrossEncoderFilter()
        queries = [
            Query(text="test", query_type="factoid", source_doc_title="Paris"),
        ]
        assert f.filter(queries, []) == []

    def test_missing_source_doc(self) -> None:
        """Query referencing a doc not in the list should be filtered out."""
        f = CrossEncoderFilter()
        queries = [
            Query(
                text="What is the capital?",
                query_type="factoid",
                source_doc_title="NonExistentDoc",
            ),
        ]
        result = f.filter(queries, [PARIS_DOC])
        assert len(result) == 0

    def test_name_format(self) -> None:
        """Name includes model name and threshold."""
        f = CrossEncoderFilter(threshold=0.7)
        assert "ms-marco-MiniLM-L-6-v2" in f.name
        assert "t=0.7" in f.name

    def test_protocol_compliance(self) -> None:
        """CrossEncoderFilter satisfies the QueryFilter protocol."""
        f = CrossEncoderFilter()
        assert isinstance(f, QueryFilter)
