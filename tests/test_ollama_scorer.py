"""Pre-written tests for task-055: Ollama provider adapter on LLMScorer.

These tests will FAIL until the implementation lands. They exercise the
adapter's URL handling, payload shape, response extraction, and integration
with LLMScorer's error path. No real network calls — `requests.post` is
mocked at the module level where the adapter imports it.

Night instance: copy these tests into `tests/test_llm_scorer.py` (or
`tests/test_ollama_scorer.py` — separate file is fine) before implementing.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest


_GOOD_RESPONSE = json.dumps({
    "faithfulness": 4,
    "relevance": 5,
    "conciseness": 3,
    "reasoning": {
        "faithfulness": "Well supported",
        "relevance": "Directly answers",
        "conciseness": "Could be shorter",
    },
})


def _mock_ollama_response(text: str = _GOOD_RESPONSE, status: int = 200) -> MagicMock:
    """Create a mock requests.Response for Ollama /api/chat."""
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {"message": {"content": text}}
    response.raise_for_status = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Adapter URL handling
# ---------------------------------------------------------------------------

class TestOllamaAdapterURL:
    """Verify _ollama_adapter constructs the correct /api/chat URL."""

    @patch.dict(os.environ, {}, clear=False)
    @patch("src.scorers.llm._requests.post" if False else "requests.post")
    def test_uses_default_host_when_env_unset(self, mock_post):
        # Ensure OLLAMA_HOST is unset for this test
        os.environ.pop("OLLAMA_HOST", None)
        mock_post.return_value = _mock_ollama_response()

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        # First positional arg of requests.post should be the URL
        called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert called_url == "http://localhost:11434/api/chat"

    @patch.dict(os.environ, {"OLLAMA_HOST": "gpu-pod:11434"})
    @patch("requests.post")
    def test_bare_host_gets_http_prefix(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="qwen3.5:27b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert called_url == "http://gpu-pod:11434/api/chat"

    @patch.dict(os.environ, {"OLLAMA_HOST": "https://my-host.example.com:11434/"})
    @patch("requests.post")
    def test_full_url_with_trailing_slash_is_normalized(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert called_url == "https://my-host.example.com:11434/api/chat"


# ---------------------------------------------------------------------------
# Adapter payload shape
# ---------------------------------------------------------------------------

class TestOllamaAdapterPayload:
    """Verify _ollama_adapter sends the correct JSON body."""

    @patch("requests.post")
    def test_payload_has_model_and_messages(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "gemma4:31b"
        assert isinstance(body["messages"], list)
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert isinstance(body["messages"][0]["content"], str)
        assert len(body["messages"][0]["content"]) > 0

    @patch("requests.post")
    def test_payload_disables_streaming(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        body = mock_post.call_args.kwargs["json"]
        assert body["stream"] is False

    @patch("requests.post")
    def test_payload_sets_temperature_zero(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="qwen3.5:27b", max_retries=0)
        scorer.score(query="q", context="c", answer="a")

        body = mock_post.call_args.kwargs["json"]
        assert body["options"]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Adapter response extraction
# ---------------------------------------------------------------------------

class TestOllamaAdapterResponse:
    """Verify _ollama_adapter extracts message content correctly."""

    @patch("requests.post")
    def test_extracts_message_content(self, mock_post):
        mock_post.return_value = _mock_ollama_response()
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        result = scorer.score(query="q", context="c", answer="a")

        # Score result keys & ranges (1-5 ints from _GOOD_RESPONSE)
        assert set(result.keys()) >= {"faithfulness", "relevance", "conciseness"}
        for key in ("faithfulness", "relevance", "conciseness"):
            assert 1 <= result[key] <= 5

    @patch("requests.post")
    def test_handles_fenced_json_response(self, mock_post):
        fenced = f"```json\n{_GOOD_RESPONSE}\n```"
        mock_post.return_value = _mock_ollama_response(text=fenced)
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        result = scorer.score(query="q", context="c", answer="a")
        assert result["faithfulness"] == 4


# ---------------------------------------------------------------------------
# Adapter error handling
# ---------------------------------------------------------------------------

class TestOllamaAdapterErrors:
    """Verify failures bubble up as ScorerError (matches other providers)."""

    @patch("requests.post")
    def test_connection_error_raises_scorer_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        from src.scorers.llm import LLMScorer, ScorerError
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        with pytest.raises(ScorerError):
            scorer.score(query="q", context="c", answer="a")

    @patch("requests.post")
    def test_404_model_not_found_raises_scorer_error(self, mock_post):
        import requests
        bad = MagicMock()
        bad.status_code = 404
        bad.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = bad

        from src.scorers.llm import LLMScorer, ScorerError
        scorer = LLMScorer(provider="ollama", model="gemma4:31b", max_retries=0)
        with pytest.raises(ScorerError):
            scorer.score(query="q", context="c", answer="a")


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

class TestAdapterRegistry:
    """Verify ollama is registered in _ADAPTERS."""

    def test_ollama_in_adapters_registry(self):
        from src.scorers.llm import _ADAPTERS
        assert "ollama" in _ADAPTERS

    def test_unknown_provider_still_raises(self):
        from src.scorers.llm import LLMScorer, ScorerError
        with pytest.raises(ScorerError):
            LLMScorer(provider="bogus_provider_xyz", model="any")
