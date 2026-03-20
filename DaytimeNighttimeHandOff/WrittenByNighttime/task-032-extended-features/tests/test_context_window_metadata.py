"""Tests for LLM context window metadata (task-032).

Tests build_llm_context_metadata(), get_llm_context_window(), and the
Ollama query fallback. All Ollama calls are mocked — no real server needed.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.metadata import (
    build_llm_context_metadata,
    get_llm_context_window,
    _context_window_cache,
)


# ---------------------------------------------------------------------------
# Fixtures — clear cache between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the context window cache before each test."""
    _context_window_cache.clear()
    yield
    _context_window_cache.clear()


# ---------------------------------------------------------------------------
# build_llm_context_metadata
# ---------------------------------------------------------------------------

class TestBuildLlmContextMetadata:
    """Tests for the top-level context metadata builder."""

    def test_returns_required_keys(self):
        """Output dict must have llm_context_window and context_utilization_ratio."""
        result = build_llm_context_metadata("test-model", provider="none")
        assert "llm_context_window" in result
        assert "context_utilization_ratio" in result

    @patch("src.metadata.get_llm_context_window", return_value=8192)
    def test_with_known_context_window(self, mock_get):
        """When context window is known, ratio should be computed."""
        result = build_llm_context_metadata(
            "qwen3:4b", provider="ollama", context_char_length=4096
        )
        assert result["llm_context_window"] == 8192
        # 4096 chars / 4 = 1024 tokens, 1024 / 8192 = 0.125
        assert result["context_utilization_ratio"] == pytest.approx(0.125, abs=0.001)

    @patch("src.metadata.get_llm_context_window", return_value=None)
    def test_with_unknown_context_window(self, mock_get):
        """When context window is None, ratio should also be None."""
        result = build_llm_context_metadata(
            "unknown-model", provider="unknown", context_char_length=4096
        )
        assert result["llm_context_window"] is None
        assert result["context_utilization_ratio"] is None

    @patch("src.metadata.get_llm_context_window", return_value=4096)
    def test_zero_context_length(self, mock_get):
        """Zero context length should give 0.0 ratio."""
        result = build_llm_context_metadata(
            "model", provider="ollama", context_char_length=0
        )
        assert result["context_utilization_ratio"] == 0.0

    @patch("src.metadata.get_llm_context_window", return_value=32768)
    def test_large_context_window(self, mock_get):
        """Large context windows should produce small ratios."""
        result = build_llm_context_metadata(
            "model", provider="ollama", context_char_length=2000
        )
        # 2000 / 4 = 500 tokens, 500 / 32768 ≈ 0.0153
        assert result["context_utilization_ratio"] < 0.02


# ---------------------------------------------------------------------------
# get_llm_context_window
# ---------------------------------------------------------------------------

class TestGetLlmContextWindow:
    """Tests for the context window lookup function."""

    def test_non_ollama_returns_none(self):
        """Non-Ollama providers should return None."""
        result = get_llm_context_window("gpt-4", provider="openai-compat")
        assert result is None

    def test_none_provider_returns_none(self):
        """None provider should return None."""
        result = get_llm_context_window("model", provider=None)
        assert result is None

    @patch("src.metadata._query_ollama_context_window", return_value=8192)
    def test_ollama_queries_api(self, mock_query):
        """Ollama provider should query the API."""
        result = get_llm_context_window("qwen3:4b", provider="ollama")
        assert result == 8192
        mock_query.assert_called_once_with("qwen3:4b", None)

    @patch("src.metadata._query_ollama_context_window", return_value=8192)
    def test_caching(self, mock_query):
        """Second call for same model should use cache, not re-query."""
        r1 = get_llm_context_window("qwen3:4b", provider="ollama")
        r2 = get_llm_context_window("qwen3:4b", provider="ollama")
        assert r1 == r2 == 8192
        # Should only query once
        mock_query.assert_called_once()

    @patch("src.metadata._query_ollama_context_window", return_value=4096)
    def test_different_models_not_cached(self, mock_query):
        """Different model names should each get their own lookup."""
        get_llm_context_window("model-a", provider="ollama")
        get_llm_context_window("model-b", provider="ollama")
        assert mock_query.call_count == 2

    @patch("src.metadata._query_ollama_context_window", return_value=None)
    def test_ollama_failure_returns_none(self, mock_query):
        """If Ollama query fails, should return None (not crash)."""
        result = get_llm_context_window("missing-model", provider="ollama")
        assert result is None


# ---------------------------------------------------------------------------
# _query_ollama_context_window (Ollama API interaction)
# ---------------------------------------------------------------------------

class TestQueryOllamaContextWindow:
    """Tests for the low-level Ollama API query."""

    @patch("src.metadata.Client")
    def test_ollama_not_running(self, mock_client_cls):
        """When Ollama is not running, should return None without crashing."""
        mock_client_cls.side_effect = Exception("Connection refused")
        from src.metadata import _query_ollama_context_window
        result = _query_ollama_context_window("qwen3:4b", None)
        assert result is None

    @patch("src.metadata.Client")
    def test_model_not_found(self, mock_client_cls):
        """When model doesn't exist, should return None."""
        mock_client = MagicMock()
        mock_client.show.side_effect = Exception("model not found")
        mock_client_cls.return_value = mock_client
        from src.metadata import _query_ollama_context_window
        result = _query_ollama_context_window("nonexistent:model", None)
        assert result is None
