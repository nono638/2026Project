"""Tests for diagnostics integration in experiment_utils.generate_answer().

Verifies that generate_answer() passes diagnostics to strategies and populates
the new failure attribution columns in its return dict.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import generate_answer


@pytest.fixture
def mock_strategy():
    """Mock strategy that populates diagnostics when called."""
    strategy = MagicMock()
    strategy.name = "naive"

    def fake_run(query, retriever, model, diagnostics=None):
        if diagnostics is not None:
            diagnostics["retrieved_chunks"] = [
                {"text": "Paris is the capital.", "score": 0.9, "index": 0},
            ]
            diagnostics["filtered_chunks"] = ["Paris is the capital."]
            diagnostics["context_sent_to_llm"] = "Paris is the capital."
            diagnostics["retrieval_queries"] = [query]
            diagnostics["skipped_retrieval"] = False
        return "The capital of France is Paris."

    strategy.run.side_effect = fake_run
    return strategy


@pytest.fixture
def mock_chunker():
    chunker = MagicMock()
    chunker.chunk.return_value = [
        "Paris is the capital.",
        "France is in Europe.",
        "The Eiffel Tower is tall.",
    ]
    return chunker


@pytest.fixture
def mock_embedder():
    return MagicMock()


@pytest.fixture
def mock_query():
    query = MagicMock()
    query.text = "What is the capital of France?"
    query.reference_answer = "Paris"
    return query


@pytest.fixture
def mock_doc():
    doc = MagicMock()
    doc.text = "Paris is the capital. France is in Europe. The Eiffel Tower is tall."
    return doc


class TestGenerateAnswerDiagnosticColumns:
    """generate_answer() should return diagnostic columns."""

    def test_returns_failure_stage(self, mock_strategy, mock_chunker, mock_embedder,
                                    mock_query, mock_doc):
        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=mock_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=mock_query,
                doc=mock_doc,
                model="qwen3:4b",
            )

        assert "failure_stage" in result
        # Gold "Paris" is in the answer → "none"
        assert result["failure_stage"] == "none"

    def test_returns_gold_presence_bools(self, mock_strategy, mock_chunker, mock_embedder,
                                         mock_query, mock_doc):
        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=mock_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=mock_query,
                doc=mock_doc,
                model="qwen3:4b",
            )

        assert "gold_in_chunks" in result
        assert "gold_in_retrieved" in result
        assert "gold_in_context" in result
        # "Paris" is in chunks, retrieved, and context
        assert result["gold_in_chunks"] is True
        assert result["gold_in_retrieved"] is True
        assert result["gold_in_context"] is True

    def test_returns_context_sent_to_llm(self, mock_strategy, mock_chunker, mock_embedder,
                                          mock_query, mock_doc):
        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=mock_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=mock_query,
                doc=mock_doc,
                model="qwen3:4b",
            )

        assert "context_sent_to_llm" in result
        assert result["context_sent_to_llm"] == "Paris is the capital."

    def test_context_char_length_uses_actual_context(self, mock_strategy, mock_chunker,
                                                      mock_embedder, mock_query, mock_doc):
        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=mock_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=mock_query,
                doc=mock_doc,
                model="qwen3:4b",
            )

        # context_char_length should match actual context from diagnostics
        assert result["context_char_length"] == len("Paris is the capital.")


class TestGenerateAnswerErrorCase:
    """When strategy.run() throws, diagnostics should degrade gracefully."""

    def test_error_returns_unknown_failure_stage(self, mock_chunker, mock_embedder,
                                                  mock_query, mock_doc):
        error_strategy = MagicMock()
        error_strategy.run.side_effect = RuntimeError("Model timeout")

        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=error_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=mock_query,
                doc=mock_doc,
                model="qwen3:4b",
            )

        assert result["failure_stage"] == "unknown"
        assert result["answer"] == ""


class TestGenerateAnswerNoGold:
    """When no gold answer exists, failure_stage should be 'unknown'."""

    def test_no_reference_answer(self, mock_strategy, mock_chunker, mock_embedder, mock_doc):
        query_no_gold = MagicMock()
        query_no_gold.text = "What is the capital?"
        query_no_gold.reference_answer = None

        with patch("experiment_utils.Retriever"):
            result = generate_answer(
                strategy=mock_strategy,
                chunker=mock_chunker,
                embedder=mock_embedder,
                retrieval_mode="hybrid",
                query=query_no_gold,
                doc=mock_doc,
                model="qwen3:4b",
            )

        assert result["failure_stage"] == "unknown"
