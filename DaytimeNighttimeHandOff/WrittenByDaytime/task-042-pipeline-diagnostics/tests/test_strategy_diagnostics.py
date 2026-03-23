"""Tests for strategy diagnostics population.

Each strategy's run() method should populate a diagnostics dict when provided,
and work unchanged when diagnostics=None.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.naive import NaiveRAG
from src.strategies.corrective import CorrectiveRAG
from src.strategies.multi_query import MultiQueryRAG
from src.strategies.self_rag import SelfRAG
from src.strategies.adaptive import AdaptiveRAG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Mock LLM that returns canned responses."""
    llm = MagicMock()
    llm.generate.return_value = "The answer is 42."
    return llm


@pytest.fixture
def mock_retriever():
    """Mock retriever returning 3 chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"text": "Chunk A about science.", "score": 0.9, "index": 0},
        {"text": "Chunk B about history.", "score": 0.7, "index": 1},
        {"text": "Chunk C about math.", "score": 0.5, "index": 2},
    ]
    retriever.chunks = [
        "Chunk A about science.",
        "Chunk B about history.",
        "Chunk C about math.",
        "Chunk D about art.",
        "Chunk E about music.",
    ]
    return retriever


EXPECTED_DIAG_KEYS = {
    "retrieved_chunks",
    "filtered_chunks",
    "context_sent_to_llm",
    "retrieval_queries",
    "skipped_retrieval",
}


# ---------------------------------------------------------------------------
# NaiveRAG
# ---------------------------------------------------------------------------

class TestNaiveRAGDiagnostics:
    def test_populates_diagnostics(self, mock_llm, mock_retriever):
        strategy = NaiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("What is 6x7?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert isinstance(result, str)
        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert len(diag["retrieved_chunks"]) == 3
        # NaiveRAG doesn't filter — filtered == retrieved texts
        assert len(diag["filtered_chunks"]) == 3
        assert isinstance(diag["context_sent_to_llm"], str)
        assert len(diag["context_sent_to_llm"]) > 0
        assert diag["retrieval_queries"] == ["What is 6x7?"]
        assert diag["skipped_retrieval"] is False

    def test_none_diagnostics_works(self, mock_llm, mock_retriever):
        strategy = NaiveRAG(llm=mock_llm)
        result = strategy.run("What is 6x7?", mock_retriever, "qwen3:4b")
        assert isinstance(result, str)

    def test_explicit_none_diagnostics(self, mock_llm, mock_retriever):
        strategy = NaiveRAG(llm=mock_llm)
        result = strategy.run("What is 6x7?", mock_retriever, "qwen3:4b", diagnostics=None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# CorrectiveRAG
# ---------------------------------------------------------------------------

class TestCorrectiveRAGDiagnostics:
    def test_populates_diagnostics_no_reformulation(self, mock_llm, mock_retriever):
        """When enough chunks pass relevance filter, no reformulation happens."""
        # Make LLM return "relevant" for relevance checks, then answer
        mock_llm.generate.side_effect = [
            "relevant",      # chunk A relevance
            "relevant",      # chunk B relevance
            "relevant",      # chunk C relevance
            "The answer.",   # final generation
        ]
        strategy = CorrectiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert len(diag["retrieved_chunks"]) == 3
        assert diag["skipped_retrieval"] is False
        assert "question?" in diag["retrieval_queries"]

    def test_populates_diagnostics_with_reformulation(self, mock_llm, mock_retriever):
        """When chunks are filtered out, reformulation triggers second retrieval."""
        mock_llm.generate.side_effect = [
            "irrelevant",          # chunk A — filtered out
            "irrelevant",          # chunk B — filtered out
            "irrelevant",          # chunk C — filtered out
            "better question?",    # reformulated query
            "relevant",            # second round chunk A
            "relevant",            # second round chunk B
            "relevant",            # second round chunk C
            "The answer.",         # final generation
        ]
        strategy = CorrectiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        # Should have queries from both rounds
        assert len(diag["retrieval_queries"]) >= 2
        assert diag["skipped_retrieval"] is False

    def test_none_diagnostics_works(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = ["relevant", "relevant", "relevant", "Answer."]
        strategy = CorrectiveRAG(llm=mock_llm)
        result = strategy.run("question?", mock_retriever, "qwen3:4b")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# MultiQueryRAG
# ---------------------------------------------------------------------------

class TestMultiQueryRAGDiagnostics:
    def test_populates_diagnostics(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = [
            "Alt question 1?\nAlt question 2?\nAlt question 3?",  # rephrase
            "The answer.",  # generation
        ]
        strategy = MultiQueryRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("original?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        # Should have original + up to 3 alternatives
        assert len(diag["retrieval_queries"]) >= 2
        assert "original?" in diag["retrieval_queries"]
        assert diag["skipped_retrieval"] is False
        assert isinstance(diag["context_sent_to_llm"], str)

    def test_none_diagnostics_works(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = ["Alt 1?\nAlt 2?\nAlt 3?", "Answer."]
        strategy = MultiQueryRAG(llm=mock_llm)
        result = strategy.run("question?", mock_retriever, "qwen3:4b")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SelfRAG
# ---------------------------------------------------------------------------

class TestSelfRAGDiagnostics:
    def test_populates_diagnostics_with_retrieval(self, mock_llm, mock_retriever):
        """When SelfRAG decides retrieval IS needed."""
        mock_llm.generate.side_effect = [
            "yes",          # retrieval decision
            "relevant",     # chunk A
            "relevant",     # chunk B
            "irrelevant",   # chunk C — filtered
            "First answer.",  # generation
            "Final answer.",  # self-critique
        ]
        strategy = SelfRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert len(diag["retrieved_chunks"]) == 3
        # One chunk was filtered as irrelevant
        assert len(diag["filtered_chunks"]) == 2
        assert diag["skipped_retrieval"] is False

    def test_no_retrieval_path(self, mock_llm, mock_retriever):
        """When SelfRAG decides retrieval is NOT needed."""
        mock_llm.generate.side_effect = [
            "no",             # retrieval decision
            "Direct answer.", # generation without retrieval
        ]
        strategy = SelfRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert diag["skipped_retrieval"] is True
        assert diag["retrieved_chunks"] == []
        assert diag["filtered_chunks"] == []

    def test_none_diagnostics_works(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = ["yes", "relevant", "relevant", "relevant", "A.", "A."]
        strategy = SelfRAG(llm=mock_llm)
        result = strategy.run("question?", mock_retriever, "qwen3:4b")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AdaptiveRAG
# ---------------------------------------------------------------------------

class TestAdaptiveRAGDiagnostics:
    def test_simple_path_skips_retrieval(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = [
            "simple",       # classification
            "Direct answer.",  # generation
        ]
        strategy = AdaptiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("What is 2+2?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert diag["skipped_retrieval"] is True
        assert diag["retrieved_chunks"] == []

    def test_moderate_path(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = [
            "moderate",     # classification
            "The answer.",  # generation
        ]
        strategy = AdaptiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert diag["skipped_retrieval"] is False
        assert len(diag["retrieved_chunks"]) == 3

    def test_complex_path(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = [
            "complex",          # classification
            "Preliminary.",     # intermediate answer
            "Follow-up?",       # follow-up query
            "Final answer.",    # final generation
        ]
        strategy = AdaptiveRAG(llm=mock_llm)
        diag = {}
        result = strategy.run("complex question?", mock_retriever, "qwen3:4b", diagnostics=diag)

        assert EXPECTED_DIAG_KEYS.issubset(diag.keys())
        assert diag["skipped_retrieval"] is False
        # Complex path does two retrievals
        assert len(diag["retrieval_queries"]) >= 2

    def test_none_diagnostics_works(self, mock_llm, mock_retriever):
        mock_llm.generate.side_effect = ["moderate", "Answer."]
        strategy = AdaptiveRAG(llm=mock_llm)
        result = strategy.run("question?", mock_retriever, "qwen3:4b")
        assert isinstance(result, str)
