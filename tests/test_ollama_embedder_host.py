"""Tests for OllamaEmbedder host parameter.

Verifies that the host parameter is correctly passed to ollama.Client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestOllamaEmbedderHost:
    """Tests for the host parameter on OllamaEmbedder."""

    @patch("src.embedders.ollama.Client")
    def test_embedder_default_host(self, mock_client_cls: MagicMock) -> None:
        """OllamaEmbedder() with no host calls Client() with no args."""
        from src.embedders.ollama import OllamaEmbedder

        embedder = OllamaEmbedder()

        # Client() should be called with no arguments
        mock_client_cls.assert_called_once_with()

    @patch("src.embedders.ollama.Client")
    def test_embedder_custom_host(self, mock_client_cls: MagicMock) -> None:
        """OllamaEmbedder(host=...) calls Client(host=...)."""
        from src.embedders.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(host="http://remote:11434")

        # Client should be called with the host parameter
        mock_client_cls.assert_called_once_with(host="http://remote:11434")
