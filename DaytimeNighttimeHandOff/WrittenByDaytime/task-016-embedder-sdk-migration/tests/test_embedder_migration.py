"""Tests for Google embedder migration to google-genai SDK.

Mocks the google.genai Client to avoid network dependency.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_embedding(values: list[float] | None = None) -> MagicMock:
    """Create a mock embedding result matching the new SDK format."""
    if values is None:
        values = [0.1] * 768
    embedding = MagicMock()
    embedding.values = values
    return embedding


def _make_mock_response(values: list[float] | None = None) -> MagicMock:
    """Create a mock embed_content response."""
    response = MagicMock()
    response.embeddings = [_make_mock_embedding(values)]
    return response


def _make_mock_client() -> MagicMock:
    """Create a mock genai.Client with models.embed_content."""
    client = MagicMock()
    client.models.embed_content.return_value = _make_mock_response()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicEmbed:
    """Test basic embedding functionality with new SDK."""

    @patch("src.embedders.google_text.genai")
    def test_embed_returns_correct_shape(self, mock_genai):
        mock_client = _make_mock_client()
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        result = embedder.embed(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 768)
        assert result.dtype == np.float32

    @patch("src.embedders.google_text.genai")
    def test_embed_empty_list(self, mock_genai):
        mock_client = _make_mock_client()
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        result = embedder.embed([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 768)

    @patch("src.embedders.google_text.genai")
    def test_embed_calls_api_per_text(self, mock_genai):
        mock_client = _make_mock_client()
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        embedder.embed(["a", "b", "c"])

        assert mock_client.models.embed_content.call_count == 3


class TestEmbedQuery:
    """Test query embedding uses retrieval_query task type."""

    @patch("src.embedders.google_text.genai")
    def test_embed_query_returns_correct_shape(self, mock_genai):
        mock_client = _make_mock_client()
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        result = embedder.embed_query(["what is this?"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)

    @patch("src.embedders.google_text.genai")
    def test_embed_query_empty_list(self, mock_genai):
        mock_client = _make_mock_client()
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        result = embedder.embed_query([])

        assert result.shape == (0, 768)


class TestProperties:
    """Test name and dimension properties."""

    @patch("src.embedders.google_text.genai")
    def test_name_strips_models_prefix(self, mock_genai):
        mock_genai.Client.return_value = _make_mock_client()

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        assert embedder.name == "google:text-embedding-005"

    @patch("src.embedders.google_text.genai")
    def test_dimension_is_768(self, mock_genai):
        mock_genai.Client.return_value = _make_mock_client()

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        assert embedder.dimension == 768


class TestRateLimitRetry:
    """Test rate limit retry behavior."""

    @patch("src.embedders.google_text.time")
    @patch("src.embedders.google_text.genai")
    def test_retries_on_rate_limit(self, mock_genai, mock_time):
        mock_client = _make_mock_client()
        # First call raises rate limit, second succeeds
        mock_client.models.embed_content.side_effect = [
            Exception("429 rate limit exceeded"),
            _make_mock_response(),
        ]
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")
        result = embedder.embed(["test"])

        assert result.shape == (1, 768)
        mock_time.sleep.assert_called_once_with(1)

    @patch("src.embedders.google_text.genai")
    def test_raises_on_non_rate_limit_error(self, mock_genai):
        mock_client = _make_mock_client()
        mock_client.models.embed_content.side_effect = Exception("Internal server error")
        mock_genai.Client.return_value = mock_client

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="test-key")

        with pytest.raises(Exception, match="Internal server error"):
            embedder.embed(["test"])


class TestApiKeyHandling:
    """Test API key resolution."""

    @patch("src.embedders.google_text.genai")
    def test_no_api_key_raises(self, mock_genai):
        from src.embedders.google_text import GoogleTextEmbedder

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GoogleTextEmbedder()

    @patch("src.embedders.google_text.genai")
    def test_explicit_api_key_used(self, mock_genai):
        mock_genai.Client.return_value = _make_mock_client()

        from src.embedders.google_text import GoogleTextEmbedder
        embedder = GoogleTextEmbedder(api_key="explicit-key")
        # Should not raise — key was provided explicitly
        assert embedder is not None
