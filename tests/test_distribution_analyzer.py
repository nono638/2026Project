"""Tests for DistributionAnalyzer — set-level query quality analysis.

Tests statistical analysis, coverage, diversity, and optional embedding-based
analysis.
"""

from __future__ import annotations

import io
import sys

import pytest

from src.document import Document
from src.query import Query
from src.query_analysis.distribution import DistributionAnalyzer


def _make_queries(types: list[str], doc_titles: list[str]) -> list[Query]:
    """Helper: create queries with given types and source doc titles."""
    queries = []
    for i, (qtype, title) in enumerate(zip(types, doc_titles)):
        queries.append(
            Query(
                text=f"What is the meaning of query number {i} about {title}?",
                query_type=qtype,
                source_doc_title=title,
            )
        )
    return queries


def _make_documents(titles: list[str]) -> list[Document]:
    """Helper: create documents with given titles."""
    return [
        Document(title=t, text=f"This is the document about {t}.")
        for t in titles
    ]


class TestDistributionAnalyzer:
    """Tests for DistributionAnalyzer."""

    def test_type_distribution(self) -> None:
        """Type counts and fractions should match known input."""
        queries = _make_queries(
            ["factoid", "factoid", "factoid", "reasoning", "multi_context"],
            ["A", "B", "C", "D", "E"],
        )
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        td = result["type_distribution"]
        assert td["factoid"]["count"] == 3
        assert td["reasoning"]["count"] == 1
        assert td["multi_context"]["count"] == 1
        assert abs(td["factoid"]["fraction"] - 0.6) < 0.01

    def test_length_stats(self) -> None:
        """Word-count statistics should be computed correctly."""
        # Known word counts: "one two three" = 3, "one two three four five" = 5
        queries = [
            Query(text="one two three", query_type="factoid", source_doc_title="A"),
            Query(text="one two three four five", query_type="factoid", source_doc_title="B"),
            Query(text="one two three four five six seven", query_type="factoid", source_doc_title="C"),
        ]
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        ls = result["length_stats"]
        assert ls["min"] == 3
        assert ls["max"] == 7
        assert ls["mean"] == 5.0
        assert ls["median"] == 5

    def test_document_coverage(self) -> None:
        """Should correctly count docs with and without queries."""
        documents = _make_documents(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
        queries = _make_queries(
            ["factoid"] * 8,
            ["A", "B", "C", "D", "E", "F", "G", "H"],
        )
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries, documents)

        assert result["docs_with_queries"] == 8
        assert result["docs_without_queries"] == 2

    def test_queries_per_doc(self) -> None:
        """Should compute min/max queries per document correctly."""
        queries = _make_queries(
            ["factoid"] * 4,
            ["A", "A", "A", "B"],
        )
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        qpd = result["queries_per_doc"]
        assert qpd["min"] == 1
        assert qpd["max"] == 3

    def test_duplicate_detection(self) -> None:
        """Should detect exact string duplicates."""
        queries = [
            Query(text="What is X?", query_type="factoid", source_doc_title="A"),
            Query(text="What is X?", query_type="factoid", source_doc_title="B"),
            Query(text="What is Y?", query_type="factoid", source_doc_title="C"),
        ]
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        assert result["duplicate_count"] == 1

    def test_lexical_diversity(self) -> None:
        """Type-token ratio should be computed correctly."""
        # All unique words
        queries = [
            Query(text="alpha beta gamma", query_type="factoid", source_doc_title="A"),
            Query(text="delta epsilon zeta", query_type="factoid", source_doc_title="B"),
        ]
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        # 6 unique words / 6 total words = 1.0
        assert result["lexical_diversity"] == 1.0

    def test_warnings_type_imbalance(self) -> None:
        """Should warn when one type dominates (>60%)."""
        queries = _make_queries(
            ["factoid"] * 19 + ["reasoning"],
            [f"doc{i}" for i in range(20)],
        )
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries)

        # factoid is 95%, should trigger warning
        imbalance_warnings = [w for w in result["warnings"] if "imbalance" in w]
        assert len(imbalance_warnings) > 0

    def test_warnings_low_coverage(self) -> None:
        """Should warn when coverage is below 80%."""
        documents = _make_documents([f"doc{i}" for i in range(10)])
        queries = _make_queries(
            ["factoid"] * 4,
            ["doc0", "doc1", "doc2", "doc3"],
        )
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(queries, documents)

        # 4/10 = 40% coverage, should trigger warning
        coverage_warnings = [w for w in result["warnings"] if "coverage" in w.lower()]
        assert len(coverage_warnings) > 0

    def test_no_embedder_skips_embedding_analysis(self) -> None:
        """Analyzer with embedder=None should not have embedding keys."""
        queries = _make_queries(["factoid"] * 3, ["A", "B", "C"])
        analyzer = DistributionAnalyzer(embedder=None)
        result = analyzer.analyze(queries)

        assert "embedding_diversity" not in result
        assert "corpus_coverage" not in result

    def test_with_embedder(self) -> None:
        """Analyzer with embedder should compute embedding-based metrics."""
        from src.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder()
        queries = _make_queries(
            ["factoid", "reasoning", "factoid", "multi_context", "conditional"],
            ["A", "B", "C", "D", "E"],
        )
        documents = _make_documents(["A", "B", "C", "D", "E"])

        analyzer = DistributionAnalyzer(embedder=embedder)
        result = analyzer.analyze(queries, documents)

        assert "embedding_diversity" in result
        assert "mean_pairwise_distance" in result["embedding_diversity"]
        assert "cluster_count" in result["embedding_diversity"]
        assert result["embedding_diversity"]["mean_pairwise_distance"] >= 0
        assert "corpus_coverage" in result
        assert 0 <= result["corpus_coverage"] <= 1

    def test_print_report_runs(self) -> None:
        """print_report should run without exceptions."""
        queries = _make_queries(["factoid"] * 5, [f"doc{i}" for i in range(5)])
        analyzer = DistributionAnalyzer()
        analysis = analyzer.analyze(queries)

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            analyzer.print_report(analysis)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Total queries" in output
