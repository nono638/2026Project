"""Tests for GoogleTextEmbedder.

All tests use mocks — no real Google API calls are made. The SDK's
genai.embed_content and genai.configure are mocked at the module level.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedders.google_text import GoogleTextEmbedder
from src.protocols import Embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_genai_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent genai.configure from making real API calls in every test."""
    monkeypatch.setattr("google.generativeai.configure", lambda **kwargs: None)


@pytest.fixture()
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake GOOGLE_API_KEY env var for tests that need it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-testing")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_set_api_key")
class TestGoogleTextEmbedder:
    """Tests for the GoogleTextEmbedder class."""

    def test_name_format(self) -> None:
        """Name property returns 'google:<model_name>' with models/ prefix stripped."""
        embedder = GoogleTextEmbedder()
        assert embedder.name == "google:text-embedding-005"

    def test_dimension(self) -> None:
        """Dimension is always 768 for text-embedding-005."""
        embedder = GoogleTextEmbedder()
        assert embedder.dimension == 768

    @patch("src.embedders.google_text.genai.embed_content")
    def test_embed_returns_correct_shape(
        self, mock_embed: MagicMock
    ) -> None:
        """embed() returns (n, 768) float32 array from mocked API responses."""
        # Each call to embed_content returns a single embedding vector
        mock_embed.return_value = {"embedding": [0.1] * 768}

        embedder = GoogleTextEmbedder()
        result = embedder.embed(["hello", "world"])

        assert result.shape == (2, 768)
        assert result.dtype == np.float32
        assert mock_embed.call_count == 2

    @patch("src.embedders.google_text.genai.embed_content")
    def test_embed_empty_list(self, mock_embed: MagicMock) -> None:
        """embed([]) returns shape (0, 768) without calling the API."""
        embedder = GoogleTextEmbedder()
        result = embedder.embed([])

        assert result.shape == (0, 768)
        assert result.dtype == np.float32
        # API should not be called for empty input
        mock_embed.assert_not_called()

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor raises ValueError when no API key is available."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleTextEmbedder()

    @patch("src.embedders.google_text.genai.embed_content")
    def test_embed_query_uses_retrieval_query_task_type(
        self, mock_embed: MagicMock
    ) -> None:
        """embed_query() calls the API with task_type='retrieval_query'."""
        mock_embed.return_value = {"embedding": [0.5] * 768}

        embedder = GoogleTextEmbedder()
        result = embedder.embed_query(["test query"])

        assert result.shape == (1, 768)
        # Verify the API was called with retrieval_query task type
        mock_embed.assert_called_once_with(
            model="models/text-embedding-005",
            content="test query",
            task_type="retrieval_query",
        )

    def test_protocol_compliance(self) -> None:
        """GoogleTextEmbedder satisfies the Embedder protocol (structural subtyping)."""
        embedder = GoogleTextEmbedder()
        assert isinstance(embedder, Embedder)
