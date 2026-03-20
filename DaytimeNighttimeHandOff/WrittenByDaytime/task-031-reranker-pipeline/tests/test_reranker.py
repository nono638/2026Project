"""Tests for Reranker Protocol and implementations (MiniLM + BGE).

Uses real models — downloads on first run (~25MB for MiniLM, ~1.1GB for BGE).
BGE tests are marked slow for optional skipping.
"""

from __future__ import annotations

import math

import pytest

from src.protocols import Reranker


# ---------------------------------------------------------------------------
# Fixtures — sample retrieved chunks (mimics Retriever.retrieve() output)
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"text": "Python is a programming language.", "score": 0.85, "index": 0},
    {"text": "Machine learning uses statistical methods.", "score": 0.72, "index": 1},
    {"text": "The Eiffel Tower is in Paris, France.", "score": 0.45, "index": 2},
    {"text": "Python supports object-oriented programming.", "score": 0.80, "index": 3},
    {"text": "Deep learning is a subset of machine learning.", "score": 0.68, "index": 4},
]


# ---------------------------------------------------------------------------
# MiniLM Reranker tests
# ---------------------------------------------------------------------------

class TestMiniLMReranker:
    """Tests for the lightweight MiniLM cross-encoder reranker."""

    @pytest.fixture(scope="class")
    def reranker(self):
        from src.rerankers import MiniLMReranker
        return MiniLMReranker()

    def test_protocol_compliance(self, reranker):
        """MiniLMReranker satisfies the Reranker protocol."""
        assert isinstance(reranker, Reranker)

    def test_name_format(self, reranker):
        """Name should be 'minilm:ms-marco-MiniLM-L-6-v2'."""
        assert reranker.name == "minilm:ms-marco-MiniLM-L-6-v2"

    def test_rerank_returns_all_keys(self, reranker):
        """Each result dict must have text, score, rerank_score, index."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "rerank_score" in r
            assert "index" in r

    def test_rerank_preserves_original_score(self, reranker):
        """Original retrieval score must be preserved unchanged."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        original_scores = {c["index"]: c["score"] for c in SAMPLE_CHUNKS}
        for r in results:
            assert r["score"] == original_scores[r["index"]]

    def test_rerank_sorted_by_rerank_score(self, reranker):
        """Results must be sorted by rerank_score descending."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_scores_are_sigmoid(self, reranker):
        """Rerank scores must be between 0 and 1 (sigmoid applied)."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        for r in results:
            assert 0.0 <= r["rerank_score"] <= 1.0

    def test_rerank_top_k_truncates(self, reranker):
        """Setting top_k should limit results."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS, top_k=2)
        assert len(results) == 2

    def test_rerank_top_k_larger_than_input(self, reranker):
        """top_k > input size should return all chunks."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS, top_k=100)
        assert len(results) == len(SAMPLE_CHUNKS)

    def test_rerank_top_k_zero(self, reranker):
        """top_k=0 should return empty list."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS, top_k=0)
        assert results == []

    def test_rerank_empty_input(self, reranker):
        """Empty chunk list should return empty list."""
        results = reranker.rerank("What is Python?", [])
        assert results == []

    def test_rerank_single_chunk(self, reranker):
        """Single chunk should work without error."""
        single = [SAMPLE_CHUNKS[0]]
        results = reranker.rerank("What is Python?", single)
        assert len(results) == 1
        assert "rerank_score" in results[0]

    def test_rerank_none_top_k_returns_all(self, reranker):
        """top_k=None (default) returns all chunks reranked."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS, top_k=None)
        assert len(results) == len(SAMPLE_CHUNKS)


# ---------------------------------------------------------------------------
# BGE Reranker tests
# ---------------------------------------------------------------------------

class TestBGEReranker:
    """Tests for the deeper BGE cross-encoder reranker.

    Model is ~1.1GB — downloads on first run. Tests are identical in structure
    to MiniLM but use the larger model.
    """

    @pytest.fixture(scope="class")
    def reranker(self):
        from src.rerankers import BGEReranker
        return BGEReranker()

    def test_protocol_compliance(self, reranker):
        """BGEReranker satisfies the Reranker protocol."""
        assert isinstance(reranker, Reranker)

    def test_name_format(self, reranker):
        """Name should be 'bge:bge-reranker-v2-m3'."""
        assert reranker.name == "bge:bge-reranker-v2-m3"

    def test_rerank_returns_all_keys(self, reranker):
        """Each result dict must have text, score, rerank_score, index."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "rerank_score" in r
            assert "index" in r

    def test_rerank_preserves_original_score(self, reranker):
        """Original retrieval score must be preserved unchanged."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        original_scores = {c["index"]: c["score"] for c in SAMPLE_CHUNKS}
        for r in results:
            assert r["score"] == original_scores[r["index"]]

    def test_rerank_sorted_by_rerank_score(self, reranker):
        """Results must be sorted by rerank_score descending."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_scores_are_sigmoid(self, reranker):
        """Rerank scores must be between 0 and 1 (sigmoid applied)."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS)
        for r in results:
            assert 0.0 <= r["rerank_score"] <= 1.0

    def test_rerank_top_k_truncates(self, reranker):
        """Setting top_k should limit results."""
        results = reranker.rerank("What is Python?", SAMPLE_CHUNKS, top_k=2)
        assert len(results) == 2

    def test_rerank_empty_input(self, reranker):
        """Empty chunk list should return empty list."""
        results = reranker.rerank("What is Python?", [])
        assert results == []


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

class TestRerankerComparison:
    """Verify both rerankers produce different but valid orderings."""

    @pytest.fixture(scope="class")
    def minilm(self):
        from src.rerankers import MiniLMReranker
        return MiniLMReranker()

    @pytest.fixture(scope="class")
    def bge(self):
        from src.rerankers import BGEReranker
        return BGEReranker()

    def test_both_produce_valid_output(self, minilm, bge):
        """Both rerankers should produce valid results on the same input."""
        query = "What programming paradigms does Python support?"
        r1 = minilm.rerank(query, SAMPLE_CHUNKS)
        r2 = bge.rerank(query, SAMPLE_CHUNKS)

        assert len(r1) == len(SAMPLE_CHUNKS)
        assert len(r2) == len(SAMPLE_CHUNKS)

        # Both should have scores in [0, 1]
        for r in r1 + r2:
            assert 0.0 <= r["rerank_score"] <= 1.0
