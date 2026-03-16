"""Tests for the core framework: protocols, retriever, and experiment runner.

Uses mock implementations of each Protocol to verify the framework works
end-to-end without requiring real Ollama models, FAISS embeddings, or API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.protocols import Chunker, Embedder, Strategy, Scorer
from src.retriever import Retriever
from src.experiment import Experiment, ExperimentResult


# ---------------------------------------------------------------------------
# Mock implementations — satisfy Protocol interfaces without real dependencies
# ---------------------------------------------------------------------------

class MockChunker:
    """Splits text on double newlines. No external dependencies."""

    @property
    def name(self) -> str:
        """Return chunker identifier."""
        return "mock_chunker"

    def chunk(self, text: str) -> list[str]:
        """Split on double newlines."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class MockEmbedder:
    """Produces deterministic fake embeddings based on a fixed random seed."""

    @property
    def name(self) -> str:
        """Return embedder identifier."""
        return "mock_embedder"

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        """Deterministic fake embeddings based on text length."""
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 4).astype(np.float32)


class MockStrategy:
    """Returns a canned answer referencing retrieved chunk count."""

    def __init__(self, name: str = "mock_strategy") -> None:
        """Initialize with an optional custom name for cartesian product tests."""
        self._name = name

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return self._name

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        """Return a canned answer referencing the retrieval result."""
        retrieved = retriever.retrieve(query)
        return f"Answer based on {len(retrieved)} chunks using {model}"


class MockScorer:
    """Returns fixed scores for every answer."""

    @property
    def name(self) -> str:
        """Return scorer identifier."""
        return "mock_scorer"

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        """Return fixed scores."""
        return {"faithfulness": 4.0, "relevance": 3.5, "conciseness": 4.0}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOC = {
    "title": "Test Document",
    "text": "The quick brown fox jumps over the lazy dog.\n\n"
            "Python is a popular programming language.\n\n"
            "FAISS is a library for efficient similarity search.",
}

SAMPLE_QUERY = {
    "text": "What is Python?",
    "type": "lookup",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify mock classes pass isinstance checks against Protocol definitions."""

    def test_chunker_protocol(self) -> None:
        """MockChunker should satisfy the Chunker protocol."""
        assert isinstance(MockChunker(), Chunker)

    def test_embedder_protocol(self) -> None:
        """MockEmbedder should satisfy the Embedder protocol."""
        assert isinstance(MockEmbedder(), Embedder)

    def test_strategy_protocol(self) -> None:
        """MockStrategy should satisfy the Strategy protocol."""
        assert isinstance(MockStrategy(), Strategy)

    def test_scorer_protocol(self) -> None:
        """MockScorer should satisfy the Scorer protocol."""
        assert isinstance(MockScorer(), Scorer)


class TestRetriever:
    """Test Retriever build and search functionality."""

    def test_retriever_build_and_search(self) -> None:
        """Build a Retriever with MockEmbedder, verify retrieve() returns results."""
        embedder = MockEmbedder()
        chunks = ["chunk one", "chunk two", "chunk three"]
        retriever = Retriever(chunks, embedder, top_k=2)

        results = retriever.retrieve("test query", top_k=2)

        assert len(results) <= 2
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "index" in r
            assert r["text"] in chunks

    def test_retriever_empty_chunks(self) -> None:
        """Retriever with empty chunks should return empty results."""
        embedder = MockEmbedder()
        retriever = Retriever([], embedder, top_k=5)

        results = retriever.retrieve("test query")
        assert results == []

    def test_retriever_chunks_property(self) -> None:
        """The chunks property should return the original chunks."""
        embedder = MockEmbedder()
        chunks = ["a", "b", "c"]
        retriever = Retriever(chunks, embedder)

        assert retriever.chunks == chunks


class TestExperiment:
    """Test Experiment runner with mock components."""

    def test_experiment_runs(self) -> None:
        """Create an Experiment with all mocks, run on tiny corpus, verify result."""
        exp = Experiment(
            chunkers=[MockChunker()],
            embedders=[MockEmbedder()],
            models=["test-model"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        exp.load_corpus([SAMPLE_DOC], [SAMPLE_QUERY])
        result = exp.run(progress=False)

        assert isinstance(result, ExperimentResult)
        assert len(result.df) == 1

        # Verify expected columns are present
        expected_cols = [
            "doc_title", "query_text", "query_type", "chunker", "embedder",
            "model", "strategy", "answer", "faithfulness", "relevance",
            "conciseness", "quality", "timestamp",
        ]
        for col in expected_cols:
            assert col in result.df.columns, f"Missing column: {col}"

    def test_experiment_cartesian_product(self) -> None:
        """2 strategies x 2 models = 4 rows for 1 doc x 1 query."""
        exp = Experiment(
            chunkers=[MockChunker()],
            embedders=[MockEmbedder()],
            models=["model-a", "model-b"],
            strategies=[MockStrategy("strat_a"), MockStrategy("strat_b")],
            scorer=MockScorer(),
        )
        exp.load_corpus([SAMPLE_DOC], [SAMPLE_QUERY])
        result = exp.run(progress=False)

        assert len(result.df) == 4

    def test_experiment_empty_corpus(self) -> None:
        """Experiment with no documents should return empty result."""
        exp = Experiment(
            chunkers=[MockChunker()],
            embedders=[MockEmbedder()],
            models=["test-model"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        # Don't load any corpus
        result = exp.run(progress=False)

        assert isinstance(result, ExperimentResult)
        assert result.df.empty

    def test_experiment_rejects_bad_protocol(self) -> None:
        """Experiment should reject components that don't implement Protocols."""
        with pytest.raises(TypeError, match="does not implement the Chunker protocol"):
            Experiment(
                chunkers=["not_a_chunker"],  # type: ignore
                embedders=[MockEmbedder()],
                models=["test-model"],
                strategies=[MockStrategy()],
                scorer=MockScorer(),
            )


class TestExperimentResult:
    """Test ExperimentResult analysis methods."""

    def _make_result(self) -> ExperimentResult:
        """Helper: run a small experiment and return its result."""
        exp = Experiment(
            chunkers=[MockChunker()],
            embedders=[MockEmbedder()],
            models=["model-a", "model-b"],
            strategies=[MockStrategy("strat_a"), MockStrategy("strat_b")],
            scorer=MockScorer(),
        )
        exp.load_corpus([SAMPLE_DOC], [SAMPLE_QUERY])
        return exp.run(progress=False)

    def test_compare(self) -> None:
        """compare() should return a summary DataFrame."""
        result = self._make_result()
        summary = result.compare()

        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert len(summary) == 4  # 2 strategies x 2 models

    def test_parquet_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Save to parquet, load back, verify equality."""
        result = self._make_result()
        path = tmp_path / "test_results.parquet"

        result.to_parquet(path)
        loaded = ExperimentResult.from_parquet(path)

        pd.testing.assert_frame_equal(result.df, loaded.df)

    def test_compare_empty(self) -> None:
        """compare() on empty result should not crash."""
        result = ExperimentResult(pd.DataFrame())
        summary = result.compare()
        assert summary.empty

    def test_best_config(self) -> None:
        """best_config() should return a tuple of config values."""
        result = self._make_result()
        best = result.best_config()

        # Should be a tuple of (chunker, embedder, strategy, model)
        assert isinstance(best, tuple)
        assert len(best) == 4
