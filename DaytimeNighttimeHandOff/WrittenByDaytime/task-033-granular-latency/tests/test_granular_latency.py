"""Tests for task-033: granular latency breakdown.

Verifies that _TimedRetriever works correctly and that the experiment
runner produces the new per-stage latency columns.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _TimedRetriever unit tests
# ---------------------------------------------------------------------------


class TestTimedRetriever:
    """Test the timing wrapper around Retriever."""

    def _make_timed_retriever(self):
        """Create a _TimedRetriever wrapping a mock Retriever."""
        from src.experiment import _TimedRetriever

        mock_retriever = MagicMock()
        mock_retriever.chunks = ["chunk1", "chunk2", "chunk3"]
        mock_retriever._embedder = MagicMock()
        mock_retriever._index = MagicMock()

        def slow_retrieve(query, top_k=None):
            time.sleep(0.01)  # 10ms
            return [{"text": "chunk1", "score": 0.9, "index": 0}]

        mock_retriever.retrieve.side_effect = slow_retrieve
        return _TimedRetriever(mock_retriever), mock_retriever

    def test_accumulates_time(self):
        """Multiple retrieve() calls accumulate total time."""
        timed, _ = self._make_timed_retriever()
        timed.retrieve("query 1")
        timed.retrieve("query 2")
        # Two calls at ~10ms each should give >= 15ms (conservative)
        assert timed.retrieval_ms >= 15.0

    def test_reset_timer(self):
        """reset() clears accumulated time to zero."""
        timed, _ = self._make_timed_retriever()
        timed.retrieve("query")
        assert timed.retrieval_ms > 0
        timed.reset()
        assert timed.retrieval_ms == 0.0

    def test_delegates_attribute_access(self):
        """Attributes not on _TimedRetriever delegate to inner retriever."""
        timed, mock = self._make_timed_retriever()
        assert timed.chunks == ["chunk1", "chunk2", "chunk3"]

    def test_returns_retrieve_results(self):
        """retrieve() returns the inner retriever's results unchanged."""
        timed, _ = self._make_timed_retriever()
        results = timed.retrieve("query")
        assert len(results) == 1
        assert results[0]["text"] == "chunk1"
        assert results[0]["score"] == 0.9

    def test_passes_top_k_through(self):
        """top_k parameter is forwarded to inner retriever."""
        timed, mock = self._make_timed_retriever()
        timed.retrieve("query", top_k=3)
        mock.retrieve.assert_called_with("query", top_k=3)

    def test_zero_retrieval_calls(self):
        """No retrieve() calls means retrieval_ms stays at 0."""
        timed, _ = self._make_timed_retriever()
        assert timed.retrieval_ms == 0.0


# ---------------------------------------------------------------------------
# Experiment output column tests
# ---------------------------------------------------------------------------


class _MockChunker:
    name = "mock_chunker:100"

    def chunk(self, text: str) -> list[str]:
        return ["chunk1", "chunk2"]


class _MockEmbedder:
    name = "mock_embedder"
    dimension = 4

    def embed(self, texts: list[str]):
        import numpy as np

        return np.random.rand(len(texts), self.dimension).astype("float32")


class _MockStrategy:
    name = "MockStrategy"

    def __init__(self):
        self._llm = None

    def run(self, query: str, retriever, model: str) -> str:
        # Simulate retrieval + generation
        retriever.retrieve(query)
        time.sleep(0.01)  # simulate LLM generation
        return "mock answer"


class _MockScorer:
    name = "mock_scorer"

    def score(self, query: str, document: str, answer: str) -> dict:
        return {"faithfulness": 4.0, "relevance": 3.5}


class _MockReranker:
    name = "mock_reranker"

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        time.sleep(0.01)  # simulate reranking work
        for c in chunks[:top_k]:
            c["rerank_score"] = 0.8
        return chunks[:top_k]


class TestExperimentGranularLatency:
    """Verify new latency columns appear in experiment output."""

    def _run_experiment(self, use_reranker: bool = False):
        from src.experiment import Experiment

        reranker = _MockReranker() if use_reranker else None
        reranker_top_k = 2 if use_reranker else None

        exp = Experiment(
            chunkers=[_MockChunker()],
            embedders=[_MockEmbedder()],
            models=["test-model"],
            strategies=[_MockStrategy()],
            scorer=_MockScorer(),
            retrieval_top_k=2,
            reranker=reranker,
            reranker_top_k=reranker_top_k,
        )
        exp.load_corpus(
            documents=[{"title": "Test Doc", "text": "Some test document text for chunking."}],
            queries=[{"text": "test query", "type": "lookup"}],
        )
        result = exp.run(progress=False)
        return result.df

    def test_new_columns_present(self):
        """Output has retrieval, generation, and reranking latency columns."""
        df = self._run_experiment()
        assert "retrieval_latency_ms" in df.columns
        assert "generation_latency_ms" in df.columns
        assert "reranking_latency_ms" in df.columns

    def test_generation_equals_strategy_minus_retrieval(self):
        """generation_latency_ms = strategy_latency_ms - retrieval_latency_ms."""
        df = self._run_experiment()
        row = df.iloc[0]
        expected = row["strategy_latency_ms"] - row["retrieval_latency_ms"]
        assert abs(row["generation_latency_ms"] - expected) < 0.01

    def test_reranking_none_without_reranker(self):
        """reranking_latency_ms is None when no reranker configured."""
        df = self._run_experiment(use_reranker=False)
        assert df.iloc[0]["reranking_latency_ms"] is None

    def test_reranking_positive_with_reranker(self):
        """reranking_latency_ms is a positive float when reranker is present."""
        df = self._run_experiment(use_reranker=True)
        val = df.iloc[0]["reranking_latency_ms"]
        assert val is not None
        assert val > 0.0

    def test_retrieval_latency_positive(self):
        """retrieval_latency_ms should be positive (strategy calls retrieve)."""
        df = self._run_experiment()
        assert df.iloc[0]["retrieval_latency_ms"] > 0.0

    def test_existing_columns_unchanged(self):
        """strategy_latency_ms, scorer_latency_ms, total_latency_ms still present."""
        df = self._run_experiment()
        assert "strategy_latency_ms" in df.columns
        assert "scorer_latency_ms" in df.columns
        assert "total_latency_ms" in df.columns
        row = df.iloc[0]
        assert row["strategy_latency_ms"] > 0.0
        assert row["scorer_latency_ms"] > 0.0
        expected_total = row["strategy_latency_ms"] + row["scorer_latency_ms"]
        assert abs(row["total_latency_ms"] - expected_total) < 0.01


# ---------------------------------------------------------------------------
# latency_report() includes new columns
# ---------------------------------------------------------------------------


class TestLatencyReport:
    """Verify latency_report() includes granular columns."""

    def test_report_includes_new_columns(self):
        """latency_report() output has retrieval and generation latency stats."""
        from src.experiment import ExperimentResult

        data = {
            "strategy": ["NaiveRAG", "NaiveRAG"],
            "model": ["qwen3:4b", "qwen3:4b"],
            "total_latency_ms": [1000, 1200],
            "strategy_latency_ms": [800, 1000],
            "retrieval_latency_ms": [100, 150],
            "generation_latency_ms": [700, 850],
            "reranking_latency_ms": [50, 60],
            "scorer_latency_ms": [200, 200],
        }
        result = ExperimentResult(pd.DataFrame(data))
        report = result.latency_report()
        # Report should have multi-level columns for multiple latency metrics
        assert not report.empty
        # Should include retrieval and generation stats
        report_str = report.to_string()
        assert "retrieval" in report_str.lower() or len(report.columns) > 4
