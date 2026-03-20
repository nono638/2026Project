"""End-to-end smoke test — full pipeline with mock external services.

Runs the complete Experiment.run() loop with a tiny corpus, mock strategy,
and mock scorer. Verifies that the full data flow produces a valid
ExperimentResult with expected columns and reasonable values.

No external services required (Ollama, Anthropic, Google).
Uses HuggingFaceEmbedder (local, ~80MB model) for real embedding/retrieval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.chunkers import FixedSizeChunker, RecursiveChunker
from src.embedders import HuggingFaceEmbedder
from src.experiment import Experiment, ExperimentResult
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Mock implementations for external-service components
# ---------------------------------------------------------------------------

class MockStrategy:
    """Strategy that returns a canned answer without calling Ollama."""

    def __init__(self, label: str = "mock"):
        self._label = label

    @property
    def name(self) -> str:
        return f"mock:{self._label}"

    def run(self, query: str, retriever: Retriever, model: str) -> str:
        # Use retrieved context so the scorer has something to evaluate
        retrieved = retriever.retrieve(query, top_k=3)
        context = " ".join(r["text"] for r in retrieved)
        return f"Based on the context: {context[:200]}"


class MockScorer:
    """Scorer that returns deterministic scores without calling Anthropic."""

    @property
    def name(self) -> str:
        return "mock:deterministic"

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        # Return deterministic but varied scores based on input lengths
        f = min(5.0, max(1.0, len(answer) / 50))
        r = min(5.0, max(1.0, len(query) / 10))
        c = 4.0 if len(answer) < 300 else 2.0
        return {"faithfulness": f, "relevance": r, "conciseness": c}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "title": "Python Programming",
        "text": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use of "
            "significant indentation. Python is dynamically typed and garbage-collected. "
            "It supports multiple programming paradigms, including structured, "
            "object-oriented, and functional programming. Python was conceived in the "
            "late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) "
            "in the Netherlands as a successor to the ABC programming language."
        ),
    },
    {
        "title": "Machine Learning",
        "text": (
            "Machine learning is a subset of artificial intelligence that focuses on "
            "building systems that learn from data. Unlike traditional programming "
            "where rules are explicitly coded, machine learning algorithms identify "
            "patterns in training data and make predictions on new data. Common "
            "approaches include supervised learning, unsupervised learning, and "
            "reinforcement learning. Applications range from image recognition to "
            "natural language processing to recommendation systems."
        ),
    },
]

SAMPLE_QUERIES = [
    {"text": "What is Python used for?", "type": "factoid"},
    {"text": "How does machine learning differ from traditional programming?", "type": "synthesis"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ESmoke:
    """Full pipeline smoke test."""

    def test_full_experiment_run(self):
        """Run the complete experiment loop and verify output shape."""
        chunkers = [FixedSizeChunker(chunk_size=100, overlap=20)]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy("baseline")]
        scorer = MockScorer()
        models = ["mock-model-small"]

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=models,
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)

        result = exp.run(progress=False)

        # Should have docs × queries × chunkers × embedders × strategies × models rows
        expected_rows = len(SAMPLE_DOCS) * len(SAMPLE_QUERIES) * 1 * 1 * 1 * 1
        assert len(result.df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(result.df)}"
        )

        # Required columns present
        required_cols = [
            "doc_title", "query_text", "query_type",
            "chunker", "embedder", "model", "strategy",
            "answer", "faithfulness", "relevance", "conciseness", "quality",
            "query_length", "doc_length", "mean_retrieval_score",
            # Pipeline metadata columns
            "chunk_type", "chunk_size", "chunk_overlap", "num_chunks",
            "embed_provider", "embed_model", "embed_dimension",
            "retrieval_mode", "retrieval_top_k", "num_chunks_retrieved",
            "context_char_length", "reranker_model", "reranker_top_k",
            "scorer_provider", "scorer_model",
            "dataset_name", "dataset_sample_seed",
        ]
        for col in required_cols:
            assert col in result.df.columns, f"Missing column: {col}"

        # Scores are in valid range
        for metric in ["faithfulness", "relevance", "conciseness", "quality"]:
            assert result.df[metric].min() >= 1.0, f"{metric} below 1.0"
            assert result.df[metric].max() <= 5.0, f"{metric} above 5.0"

        # Answers are non-empty
        assert all(len(a) > 0 for a in result.df["answer"])

    def test_multi_config_experiment(self):
        """Run with multiple chunkers, strategies, and models."""
        chunkers = [
            FixedSizeChunker(chunk_size=80, overlap=10),
            RecursiveChunker(chunk_size=80, chunk_overlap=10),
        ]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy("naive"), MockStrategy("smart")]
        scorer = MockScorer()
        models = ["mock-small", "mock-large"]

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=models,
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)

        result = exp.run(progress=False)

        # 2 docs × 2 queries × 2 chunkers × 1 embedder × 2 strategies × 2 models = 32
        assert len(result.df) == 32

        # Each config combination should appear exactly once per doc-query pair
        configs = result.df.groupby(
            ["chunker", "strategy", "model"]
        ).size()
        # 2 docs × 2 queries = 4 rows per config
        assert (configs == 4).all()

    def test_result_analysis_on_real_output(self):
        """Analysis methods work on actual experiment output (not hand-crafted data)."""
        chunkers = [FixedSizeChunker(chunk_size=100, overlap=20)]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy("a"), MockStrategy("b")]
        scorer = MockScorer()
        models = ["small", "large"]

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=models,
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        # compare() should produce a grouped summary
        summary = result.compare()
        assert not summary.empty

        # best_config() should return a valid 4-tuple
        best = result.best_config()
        assert len(best) == 4

        # strategy_vs_size() should have strategies as rows and models as cols
        pivot = result.strategy_vs_size()
        assert pivot.shape == (2, 2)  # 2 strategies × 2 models

        # per_query() should have one row per unique query
        pq = result.per_query()
        assert len(pq) == len(SAMPLE_QUERIES)

    def test_parquet_roundtrip(self, tmp_path):
        """Results survive save/load to Parquet."""
        chunkers = [FixedSizeChunker(chunk_size=100, overlap=20)]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy()]
        scorer = MockScorer()

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=["test-model"],
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS[:1], SAMPLE_QUERIES[:1])
        result = exp.run(progress=False)

        path = tmp_path / "results.parquet"
        result.to_parquet(path)
        loaded = ExperimentResult.from_parquet(path)

        assert len(loaded.df) == len(result.df)
        assert list(loaded.df.columns) == list(result.df.columns)

    def test_csv_export(self, tmp_path):
        """Results export to CSV cleanly."""
        chunkers = [FixedSizeChunker(chunk_size=100, overlap=20)]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy()]
        scorer = MockScorer()

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=["test-model"],
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS[:1], SAMPLE_QUERIES[:1])
        result = exp.run(progress=False)

        path = tmp_path / "results.csv"
        result.to_csv(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(result.df)

    def test_empty_corpus_handled(self):
        """Experiment with no data returns empty result gracefully."""
        exp = Experiment(
            chunkers=[FixedSizeChunker()],
            embedders=[HuggingFaceEmbedder()],
            models=["m"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
        )
        # Don't load any corpus
        result = exp.run(progress=False)
        assert result.df.empty

    def test_heatmap_saves(self, tmp_path):
        """Heatmap generation works on real experiment output."""
        chunkers = [FixedSizeChunker(chunk_size=100, overlap=20)]
        embedders = [HuggingFaceEmbedder()]
        strategies = [MockStrategy("a"), MockStrategy("b")]
        scorer = MockScorer()

        exp = Experiment(
            chunkers=chunkers,
            embedders=embedders,
            models=["small", "large"],
            strategies=strategies,
            scorer=scorer,
        )
        exp.load_corpus(SAMPLE_DOCS, SAMPLE_QUERIES)
        result = exp.run(progress=False)

        path = tmp_path / "heatmap.png"
        result.heatmap("strategy", "model", save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0
