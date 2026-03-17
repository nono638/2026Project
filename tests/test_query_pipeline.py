"""Tests for query pipeline data models, helpers, and protocol compliance.

Covers: Document, Query, CSV loading, JSON persistence, bridge helpers,
QueryGenerator protocol, QueryFilter protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.document import (
    Document,
    documents_to_dicts,
    load_corpus_from_csv,
    sample_corpus,
)
from src.protocols import QueryFilter, QueryGenerator
from src.query import Query, load_queries, queries_to_dicts, save_queries


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------


class TestDocument:
    """Tests for the Document dataclass and helpers."""

    def test_document_creation(self) -> None:
        """Create a Document with all fields and verify access."""
        doc = Document(title="Test", text="Hello world", metadata={"source": "wiki"})
        assert doc.title == "Test"
        assert doc.text == "Hello world"
        assert doc.metadata == {"source": "wiki"}

    def test_document_metadata_default_none(self) -> None:
        """Document without metadata kwarg defaults to None."""
        doc = Document(title="T", text="X")
        assert doc.metadata is None

    def test_load_corpus_from_csv(self, tmp_path: Path) -> None:
        """Load documents from a CSV with title and text columns."""
        csv_path = tmp_path / "corpus.csv"
        csv_path.write_text(
            "title,text,word_count\n"
            "Article A,Content of A,100\n"
            "Article B,Content of B,200\n"
        )
        docs = load_corpus_from_csv(csv_path)
        assert len(docs) == 2
        assert docs[0].title == "Article A"
        assert docs[1].text == "Content of B"

    def test_load_corpus_skips_null_text(self, tmp_path: Path) -> None:
        """Rows with null/NaN text are skipped during CSV loading."""
        csv_path = tmp_path / "corpus.csv"
        csv_path.write_text("title,text\nGood,Some text\nBad,\n")
        docs = load_corpus_from_csv(csv_path)
        # Only "Good" row should be loaded; "Bad" has empty text
        # Note: empty string is not NaN, so pandas keeps it. Let's use actual NaN.
        csv_path.write_text("title,text\nGood,Some text\nBad,\n")
        # Use a CSV that genuinely has NaN (missing value)
        import pandas as pd

        df = pd.DataFrame(
            {"title": ["Good", "Bad"], "text": ["Some text", None]}
        )
        csv_nan_path = tmp_path / "corpus_nan.csv"
        df.to_csv(csv_nan_path, index=False)
        docs = load_corpus_from_csv(csv_nan_path)
        assert len(docs) == 1
        assert docs[0].title == "Good"

    def test_load_corpus_metadata_cols(self, tmp_path: Path) -> None:
        """Loading with metadata_cols populates metadata dict."""
        csv_path = tmp_path / "corpus.csv"
        csv_path.write_text("title,text,word_count\nA,Content,150\n")
        docs = load_corpus_from_csv(csv_path, metadata_cols=["word_count"])
        assert docs[0].metadata == {"word_count": 150}

    def test_load_corpus_empty_csv(self, tmp_path: Path) -> None:
        """CSV with only headers returns empty list."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("title,text\n")
        docs = load_corpus_from_csv(csv_path)
        assert docs == []

    def test_sample_corpus_returns_n(self) -> None:
        """sample_corpus returns exactly n documents."""
        docs = [Document(title=f"D{i}", text=f"Text {'x' * (i * 10)}") for i in range(50)]
        sampled = sample_corpus(docs, n=10)
        assert len(sampled) == 10

    def test_sample_corpus_deterministic(self) -> None:
        """Same seed produces identical samples."""
        docs = [Document(title=f"D{i}", text=f"Text {'x' * (i * 10)}") for i in range(50)]
        s1 = sample_corpus(docs, n=10, seed=42)
        s2 = sample_corpus(docs, n=10, seed=42)
        assert [d.title for d in s1] == [d.title for d in s2]

    def test_sample_corpus_n_exceeds_total(self) -> None:
        """When n >= len(documents), return all documents."""
        docs = [Document(title=f"D{i}", text="Text") for i in range(5)]
        sampled = sample_corpus(docs, n=100)
        assert len(sampled) == 5

    def test_sample_corpus_n_zero(self) -> None:
        """sample_corpus with n=0 returns empty list."""
        docs = [Document(title="D", text="T")]
        assert sample_corpus(docs, n=0) == []

    def test_documents_to_dicts(self) -> None:
        """Converts Document list to list of title/text dicts."""
        docs = [
            Document(title="A", text="Text A"),
            Document(title="B", text="Text B"),
        ]
        result = documents_to_dicts(docs)
        assert result == [
            {"title": "A", "text": "Text A"},
            {"title": "B", "text": "Text B"},
        ]


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestQuery:
    """Tests for the Query dataclass and helpers."""

    def test_query_creation(self) -> None:
        """Create a Query with all fields and verify access."""
        q = Query(
            text="What is X?",
            query_type="factoid",
            source_doc_title="Doc A",
            reference_answer="X is Y.",
            generator_name="ragas",
            metadata={"difficulty": "easy"},
        )
        assert q.text == "What is X?"
        assert q.query_type == "factoid"
        assert q.source_doc_title == "Doc A"
        assert q.reference_answer == "X is Y."
        assert q.generator_name == "ragas"
        assert q.metadata == {"difficulty": "easy"}

    def test_query_defaults(self) -> None:
        """Query with only required fields has correct defaults."""
        q = Query(text="Q?", query_type="factoid", source_doc_title="D")
        assert q.reference_answer is None
        assert q.generator_name is None
        assert q.metadata is None

    def test_save_and_load_queries(self, tmp_path: Path) -> None:
        """Queries survive a JSON round-trip."""
        queries = [
            Query(text="Q1?", query_type="factoid", source_doc_title="D1"),
            Query(
                text="Q2?",
                query_type="reasoning",
                source_doc_title="D2",
                reference_answer="A2",
                generator_name="template",
            ),
        ]
        path = tmp_path / "queries.json"
        save_queries(queries, path)
        loaded = load_queries(path)

        assert len(loaded) == 2
        assert loaded[0].text == "Q1?"
        assert loaded[1].reference_answer == "A2"
        assert loaded[1].generator_name == "template"

    def test_load_queries_validates_fields(self, tmp_path: Path) -> None:
        """Loading JSON missing required fields raises ValueError."""
        path = tmp_path / "bad.json"
        # Missing 'text' field
        path.write_text(json.dumps([{"query_type": "factoid", "source_doc_title": "D"}]))
        with pytest.raises(ValueError, match="missing required fields"):
            load_queries(path)

    def test_save_empty_queries(self, tmp_path: Path) -> None:
        """Saving an empty list writes a valid JSON array."""
        path = tmp_path / "empty.json"
        save_queries([], path)
        loaded = load_queries(path)
        assert loaded == []

    def test_queries_to_dicts(self) -> None:
        """Converts Query list to list of text/type dicts."""
        queries = [
            Query(text="Q1?", query_type="factoid", source_doc_title="D1"),
            Query(text="Q2?", query_type="reasoning", source_doc_title="D2"),
        ]
        result = queries_to_dicts(queries)
        assert result == [
            {"text": "Q1?", "type": "factoid"},
            {"text": "Q2?", "type": "reasoning"},
        ]


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests that minimal implementations satisfy the new protocols."""

    def test_query_generator_protocol(self) -> None:
        """A class with name property and generate method satisfies QueryGenerator."""

        class FakeGenerator:
            @property
            def name(self) -> str:
                return "fake"

            def generate(self, documents: list, queries_per_doc: int = 5) -> list:
                return []

        assert isinstance(FakeGenerator(), QueryGenerator)

    def test_query_filter_protocol(self) -> None:
        """A class with name property and filter method satisfies QueryFilter."""

        class FakeFilter:
            @property
            def name(self) -> str:
                return "fake"

            def filter(self, queries: list, documents: list) -> list:
                return queries

        assert isinstance(FakeFilter(), QueryFilter)
