"""Integration tests — verify all modules import and wire together correctly.

These tests catch cross-branch merge issues: broken imports, missing modules,
protocol mismatches between components that were developed on separate branches.

No external services required (Ollama, Anthropic, Google). Uses HuggingFace
embedder (local) and mock implementations for everything that hits a network.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.protocols import Chunker, Embedder, Strategy, Scorer, QueryGenerator, QueryFilter


# ---------------------------------------------------------------------------
# 1. Import health — can every module be imported?
# ---------------------------------------------------------------------------

class TestImportHealth:
    """Verify all src packages import without errors."""

    def test_import_protocols(self):
        from src.protocols import Chunker, Embedder, Strategy, Scorer
        from src.protocols import QueryGenerator, QueryFilter

    def test_import_chunkers(self):
        from src.chunkers import FixedSizeChunker, RecursiveChunker, SentenceChunker
        from src.chunkers import SemanticChunker

    def test_import_embedders(self):
        from src.embedders import OllamaEmbedder, HuggingFaceEmbedder, GoogleTextEmbedder

    def test_import_strategies(self):
        from src.strategies import NaiveRAG, SelfRAG, MultiQueryRAG
        from src.strategies import CorrectiveRAG, AdaptiveRAG

    def test_import_scorers(self):
        from src.scorers import LLMScorer

    def test_import_query_pipeline(self):
        from src.document import Document, load_corpus_from_csv, sample_corpus
        from src.query import Query, save_queries, load_queries

    def test_import_query_generators(self):
        from src.query_generators import (
            RagasQueryGenerator, HumanQuerySet, BEIRQuerySet, TemplateQueryGenerator,
        )

    def test_import_query_filters(self):
        from src.query_filters import HeuristicFilter, RoundTripFilter, CrossEncoderFilter

    def test_import_query_analysis(self):
        from src.query_analysis import DistributionAnalyzer

    def test_import_experiment(self):
        from src.experiment import Experiment, ExperimentResult

    def test_import_retriever(self):
        from src.retriever import Retriever

    def test_import_rerankers(self):
        from src.rerankers import MiniLMReranker, BGEReranker

    def test_import_features(self):
        from src.features import extract_features

    def test_import_model(self):
        from src.model.train import train
        from src.model.predict import predict

    def test_import_app(self):
        from src.app import app


# ---------------------------------------------------------------------------
# 2. Protocol compliance — every implementation satisfies its Protocol
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify all implementations are runtime-checkable against their Protocols."""

    def test_chunkers(self):
        from src.chunkers import FixedSizeChunker, RecursiveChunker, SentenceChunker
        for cls in [FixedSizeChunker, RecursiveChunker, SentenceChunker]:
            instance = cls()
            assert isinstance(instance, Chunker), f"{cls.__name__} doesn't satisfy Chunker"

    def test_huggingface_embedder(self):
        from src.embedders import HuggingFaceEmbedder
        e = HuggingFaceEmbedder()
        assert isinstance(e, Embedder), "HuggingFaceEmbedder doesn't satisfy Embedder"

    def test_scorer_protocol_shape(self):
        """LLMScorer satisfies Scorer protocol (checked structurally, not instantiated
        because it hits external APIs on init)."""
        from src.scorers.llm import LLMScorer
        # Check the class has the required methods/properties
        assert hasattr(LLMScorer, "name")
        assert hasattr(LLMScorer, "score")


# ---------------------------------------------------------------------------
# 3. Cross-component wiring
# ---------------------------------------------------------------------------

class TestCrossComponentWiring:
    """Test that components work together across module boundaries."""

    def test_chunker_to_retriever(self):
        """Chunker output feeds into Retriever correctly."""
        from src.chunkers import FixedSizeChunker
        from src.embedders import HuggingFaceEmbedder
        from src.retriever import Retriever

        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        embedder = HuggingFaceEmbedder()

        text = "Python is a programming language. " * 20
        chunks = chunker.chunk(text)
        assert len(chunks) > 0

        retriever = Retriever(chunks, embedder, top_k=3)
        results = retriever.retrieve("What is Python?")
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]

    def test_feature_extraction_with_retriever(self):
        """Feature extraction works with a real Retriever."""
        from src.chunkers import RecursiveChunker
        from src.embedders import HuggingFaceEmbedder
        from src.retriever import Retriever
        from src.features import extract_features

        text = "Machine learning is a subset of AI. " * 10
        chunker = RecursiveChunker()
        chunks = chunker.chunk(text)
        embedder = HuggingFaceEmbedder()
        retriever = Retriever(chunks, embedder)

        features = extract_features("What is machine learning?", text, retriever)

        assert "query_length" in features
        assert "doc_length" in features
        assert "mean_retrieval_score" in features
        assert "var_retrieval_score" in features
        assert "doc_vocab_entropy" in features
        # Extended features (task-032)
        assert "doc_readability_score" in features
        assert "doc_embedding_spread" in features
        assert "query_doc_similarity" in features
        assert "query_doc_lexical_overlap" in features
        assert features["query_length"] > 0
        assert features["doc_length"] > 0

    def test_document_to_query_pipeline(self):
        """Document objects work with query generators and filters."""
        from src.document import Document
        from src.query import Query
        from src.query_filters.heuristic import HeuristicFilter

        docs = [
            Document(title="Test Doc", text="Python is great for data science. " * 5),
        ]
        queries = [
            Query(
                text="What is Python used for in data science?",
                query_type="factoid",
                source_doc_title="Test Doc",
                generator_name="manual",
            ),
            Query(
                text="Hi",  # Too short — should be filtered
                query_type="factoid",
                source_doc_title="Test Doc",
                generator_name="manual",
            ),
        ]

        hf = HeuristicFilter()
        filtered = hf.filter(queries, docs)
        # The short query should be filtered out
        assert len(filtered) < len(queries)
        assert all(isinstance(q, Query) for q in filtered)

    def test_experiment_result_analysis_methods(self):
        """ExperimentResult analysis works on realistic-shaped data."""
        import pandas as pd
        from src.experiment import ExperimentResult

        rows = []
        for strategy in ["naive", "self_rag"]:
            for model in ["qwen3:0.6b", "qwen3:4b"]:
                for i in range(3):
                    rows.append({
                        "doc_title": f"doc_{i}",
                        "query_text": f"question {i}",
                        "query_type": "factoid",
                        "chunker": "recursive:512/50",
                        "embedder": "hf:all-MiniLM-L6-v2",
                        "model": model,
                        "strategy": strategy,
                        "answer": "test answer",
                        "faithfulness": 4.0,
                        "relevance": 3.5,
                        "conciseness": 4.5,
                        "quality": 4.0,
                    })

        result = ExperimentResult(pd.DataFrame(rows))

        # All analysis methods should work without error
        summary = result.compare()
        assert not summary.empty

        strategies = result.compare_strategies()
        assert "naive" in strategies.index

        models = result.compare_models()
        assert "qwen3:0.6b" in models.index

        pivot = result.strategy_vs_size()
        assert pivot.shape[0] > 0

        best = result.best_config()
        assert isinstance(best, dict)
        assert "strategy" in best and "model" in best

        per_q = result.per_query()
        assert "spread" in per_q.columns
