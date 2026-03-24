"""Tests for _RerankedRetriever wrapper in experiment_utils.

Verifies that the wrapper correctly chains retrieval → reranking,
and that generate_answer() works with and without a reranker.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call


class TestRerankedRetriever:
    """Tests for the _RerankedRetriever wrapper class."""

    def test_wrapper_calls_retriever_then_reranker(self):
        """Wrapper calls underlying retriever.retrieve(), then reranker.rerank()."""
        from scripts.experiment_utils import _RerankedRetriever

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": "chunk A", "score": 0.9, "index": 0},
            {"text": "chunk B", "score": 0.7, "index": 1},
            {"text": "chunk C", "score": 0.5, "index": 2},
        ]
        mock_retriever.chunks = ["chunk A", "chunk B", "chunk C"]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"text": "chunk B", "score": 0.7, "rerank_score": 0.95, "index": 1},
            {"text": "chunk A", "score": 0.9, "rerank_score": 0.80, "index": 0},
        ]

        wrapper = _RerankedRetriever(mock_retriever, mock_reranker, top_k=2)
        results = wrapper.retrieve("test query")

        mock_retriever.retrieve.assert_called_once_with("test query", top_k=None)
        mock_reranker.rerank.assert_called_once_with(
            "test query", mock_retriever.retrieve.return_value, 2
        )
        assert len(results) == 2
        assert results[0]["text"] == "chunk B"  # reranker reordered

    def test_wrapper_passes_top_k_to_reranker(self):
        """Wrapper passes the configured top_k to the reranker, not the retriever."""
        from scripts.experiment_utils import _RerankedRetriever

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": f"chunk {i}", "score": 1.0 - i * 0.1, "index": i}
            for i in range(10)
        ]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []

        wrapper = _RerankedRetriever(mock_retriever, mock_reranker, top_k=3)
        wrapper.retrieve("query")

        # Reranker should receive top_k=3
        _, kwargs = mock_reranker.rerank.call_args
        assert mock_reranker.rerank.call_args[0][2] == 3

    def test_wrapper_exposes_chunks_from_underlying_retriever(self):
        """Wrapper's .chunks property returns the underlying retriever's chunks."""
        from scripts.experiment_utils import _RerankedRetriever

        mock_retriever = MagicMock()
        mock_retriever.chunks = ["chunk 1", "chunk 2", "chunk 3"]
        mock_reranker = MagicMock()

        wrapper = _RerankedRetriever(mock_retriever, mock_reranker, top_k=2)
        assert wrapper.chunks == ["chunk 1", "chunk 2", "chunk 3"]


class TestGenerateAnswerRerankerIntegration:
    """Tests that generate_answer() correctly uses/skips the reranker."""

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.detect_failure_stage", return_value=("correct", "high"))
    def test_no_reranker_works_as_before(self, mock_detect, mock_retriever_cls):
        """Without a reranker, generate_answer() passes the raw Retriever to strategy."""
        from scripts.experiment_utils import generate_answer

        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever

        mock_strategy = MagicMock()
        mock_strategy.run.return_value = "test answer"

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = ["chunk1"]

        mock_embedder = MagicMock()

        mock_query = MagicMock()
        mock_query.text = "test question"
        mock_query.reference_answer = "test gold"

        mock_doc = MagicMock()
        mock_doc.text = "document text"

        result = generate_answer(
            strategy=mock_strategy,
            chunker=mock_chunker,
            embedder=mock_embedder,
            retrieval_mode="hybrid",
            query=mock_query,
            doc=mock_doc,
            model="qwen3:4b",
            reranker=None,
        )

        # Strategy should receive the raw Retriever, not a wrapper
        strategy_call_args = mock_strategy.run.call_args
        retriever_arg = strategy_call_args[0][1]
        assert retriever_arg is mock_retriever

    @patch("scripts.experiment_utils.Retriever")
    @patch("scripts.experiment_utils.detect_failure_stage", return_value=("correct", "high"))
    def test_with_reranker_wraps_retriever(self, mock_detect, mock_retriever_cls):
        """With a reranker, generate_answer() wraps the Retriever."""
        from scripts.experiment_utils import _RerankedRetriever, generate_answer

        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever

        mock_strategy = MagicMock()
        mock_strategy.run.return_value = "test answer"

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = ["chunk1"]

        mock_embedder = MagicMock()
        mock_reranker = MagicMock()

        mock_query = MagicMock()
        mock_query.text = "test question"
        mock_query.reference_answer = "test gold"

        mock_doc = MagicMock()
        mock_doc.text = "document text"

        result = generate_answer(
            strategy=mock_strategy,
            chunker=mock_chunker,
            embedder=mock_embedder,
            retrieval_mode="hybrid",
            query=mock_query,
            doc=mock_doc,
            model="qwen3:4b",
            reranker=mock_reranker,
            reranker_top_k=3,
        )

        # Strategy should receive a _RerankedRetriever wrapper
        strategy_call_args = mock_strategy.run.call_args
        retriever_arg = strategy_call_args[0][1]
        assert isinstance(retriever_arg, _RerankedRetriever)
