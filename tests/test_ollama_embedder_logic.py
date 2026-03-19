"""Tests for OllamaEmbedder embed/dimension logic (mocked Ollama client).

Complements test_ollama_embedder_host.py which covers host parameter wiring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedders.ollama import OllamaEmbedder


@patch("src.embedders.ollama.Client")
class TestOllamaEmbedderEmbed:
    """Tests for OllamaEmbedder.embed — the core embedding method."""

    def test_embed_returns_ndarray(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        embedder = OllamaEmbedder(model="test-model")
        result = embedder.embed(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_embed_passes_model_and_texts(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(embeddings=[[1.0, 2.0]])

        embedder = OllamaEmbedder(model="mxbai-embed-large")
        embedder.embed(["test text"])

        mock_client.embed.assert_called_once_with(
            model="mxbai-embed-large", input=["test text"]
        )

    def test_embed_sets_dimension_lazily(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3, 0.4]]
        )

        embedder = OllamaEmbedder()
        assert embedder._dimension is None

        embedder.embed(["probe"])
        assert embedder._dimension == 4

    def test_embed_single_text(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(embeddings=[[1.0, 2.0, 3.0]])

        embedder = OllamaEmbedder()
        result = embedder.embed(["single"])

        assert result.shape == (1, 3)


@patch("src.embedders.ollama.Client")
class TestOllamaEmbedderDimension:
    """Tests for OllamaEmbedder.dimension — lazy detection."""

    def test_dimension_triggers_probe_embed(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(
            embeddings=[[0.0] * 1024]
        )

        embedder = OllamaEmbedder()
        dim = embedder.dimension

        assert dim == 1024
        mock_client.embed.assert_called_once_with(
            model="mxbai-embed-large", input=["hello"]
        )

    def test_dimension_cached_after_first_call(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(embeddings=[[0.0] * 768])

        embedder = OllamaEmbedder()
        _ = embedder.dimension
        _ = embedder.dimension  # second access

        # Only one embed call — dimension is cached
        assert mock_client.embed.call_count == 1

    def test_dimension_cached_after_embed_call(self, mock_client_cls: MagicMock) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.embed.return_value = MagicMock(embeddings=[[0.0] * 512])

        embedder = OllamaEmbedder()
        embedder.embed(["text"])  # sets _dimension as side effect
        dim = embedder.dimension  # should not trigger another embed

        assert dim == 512
        assert mock_client.embed.call_count == 1


@patch("src.embedders.ollama.Client")
class TestOllamaEmbedderName:
    """Tests for OllamaEmbedder.name property."""

    def test_name_includes_model(self, mock_client_cls: MagicMock) -> None:
        embedder = OllamaEmbedder(model="nomic-embed-text")
        assert embedder.name == "ollama:nomic-embed-text"

    def test_default_name(self, mock_client_cls: MagicMock) -> None:
        embedder = OllamaEmbedder()
        assert embedder.name == "ollama:mxbai-embed-large"
