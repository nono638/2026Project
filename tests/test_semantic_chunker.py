"""Tests for SemanticChunker.chunk() with mocked LangChain internals."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chunkers.semantic import SemanticChunker


class TestSemanticChunkerChunk:
    """Tests for SemanticChunker.chunk — the main method."""

    @patch("src.chunkers.semantic.LCSemanticChunker")
    @patch("src.chunkers.semantic.OllamaEmbeddings")
    def test_chunk_returns_page_contents(
        self, mock_embeddings_cls: MagicMock, mock_chunker_cls: MagicMock
    ) -> None:
        # Mock LangChain Document objects
        doc1 = MagicMock()
        doc1.page_content = "First semantic chunk"
        doc2 = MagicMock()
        doc2.page_content = "Second semantic chunk"

        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.create_documents.return_value = [doc1, doc2]

        chunker = SemanticChunker(embedding_model="test-model")
        result = chunker.chunk("Some long text about various topics.")

        assert result == ["First semantic chunk", "Second semantic chunk"]
        mock_embeddings_cls.assert_called_once_with(model="test-model")
        mock_chunker.create_documents.assert_called_once_with(
            ["Some long text about various topics."]
        )

    @patch("src.chunkers.semantic.LCSemanticChunker")
    @patch("src.chunkers.semantic.OllamaEmbeddings")
    def test_chunk_empty_text(
        self, mock_embeddings_cls: MagicMock, mock_chunker_cls: MagicMock
    ) -> None:
        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.create_documents.return_value = []

        chunker = SemanticChunker()
        result = chunker.chunk("")

        assert result == []

    @patch("src.chunkers.semantic.LCSemanticChunker")
    @patch("src.chunkers.semantic.OllamaEmbeddings")
    def test_chunk_single_sentence(
        self, mock_embeddings_cls: MagicMock, mock_chunker_cls: MagicMock
    ) -> None:
        doc = MagicMock()
        doc.page_content = "Just one sentence."
        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.create_documents.return_value = [doc]

        chunker = SemanticChunker()
        result = chunker.chunk("Just one sentence.")

        assert result == ["Just one sentence."]


class TestSemanticChunkerName:
    """Tests for SemanticChunker.name property."""

    def test_name_includes_model(self) -> None:
        chunker = SemanticChunker(embedding_model="nomic-embed-text")
        assert chunker.name == "semantic:nomic-embed-text"

    def test_default_name(self) -> None:
        chunker = SemanticChunker()
        assert chunker.name == "semantic:mxbai-embed-large"
