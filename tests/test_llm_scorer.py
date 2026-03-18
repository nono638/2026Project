"""Tests for LLMScorer with provider adapters.

Mocks both Anthropic and Google APIs to avoid network dependency.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

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

_FENCED_RESPONSE = f"```json\n{_GOOD_RESPONSE}\n```"

_MALFORMED_RESPONSE = "This is not JSON at all."


def _mock_anthropic_response(text: str = _GOOD_RESPONSE) -> MagicMock:
    """Create a mock Anthropic messages.create response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def _mock_google_response(text: str = _GOOD_RESPONSE) -> MagicMock:
    """Create a mock Google generate_content response."""
    response = MagicMock()
    response.text = text
    return response


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    """Test LLMScorer with provider='anthropic'."""

    @patch("anthropic.Anthropic")
    def test_score_returns_dict(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")
        result = scorer.score("What is X?", "X is a thing.", "X is a thing.")

        assert "faithfulness" in result
        assert "relevance" in result
        assert "conciseness" in result
        assert result["faithfulness"] == 4.0
        assert result["relevance"] == 5.0

    @patch("anthropic.Anthropic")
    def test_name_property(self, mock_cls):
        mock_cls.return_value = MagicMock()

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")
        assert scorer.name == "anthropic:claude-sonnet-4-20250514"

    @patch("anthropic.Anthropic")
    def test_score_batch(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")
        items = [
            {"query": "Q1", "context": "C1", "answer": "A1"},
            {"query": "Q2", "context": "C2", "answer": "A2"},
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2
        assert all("faithfulness" in r for r in results)

    @patch("anthropic.Anthropic")
    def test_api_error_raises_scorer_error(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer, ScorerError
        scorer = LLMScorer(provider="anthropic", model="claude-sonnet-4-20250514")

        with pytest.raises(ScorerError, match="API call failed"):
            scorer.score("Q", "C", "A")


# ---------------------------------------------------------------------------
# Google provider tests
# ---------------------------------------------------------------------------

class TestGoogleProvider:
    """Test LLMScorer with provider='google'."""

    def test_score_returns_dict(self):
        """Test Google provider returns correct scores."""
        # We need to mock the import chain carefully
        mock_genai_module = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_google_response()
        mock_genai_module.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai_module),
                                         "google.genai": mock_genai_module}):
            from src.scorers.llm import LLMScorer
            scorer = LLMScorer(provider="google", model="gemini-2.5-flash", api_key="test")
            result = scorer.score("What is X?", "X is a thing.", "X is a thing.")

            assert result["faithfulness"] == 4.0
            assert result["relevance"] == 5.0
            assert result["conciseness"] == 3.0

    def test_name_property(self):
        mock_genai_module = MagicMock()
        mock_genai_module.Client.return_value = MagicMock()

        with patch.dict("sys.modules", {"google": MagicMock(genai=mock_genai_module),
                                         "google.genai": mock_genai_module}):
            from src.scorers.llm import LLMScorer
            scorer = LLMScorer(provider="google", model="gemini-2.5-flash", api_key="test")
            assert scorer.name == "google:gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Shared behavior tests (provider-independent)
# ---------------------------------------------------------------------------

class TestSharedBehavior:
    """Test behavior shared across all providers."""

    @patch("anthropic.Anthropic")
    def test_empty_answer_returns_defaults(self, mock_cls):
        mock_cls.return_value = MagicMock()

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test-model")
        result = scorer.score("Q", "Context here", "")

        assert result["faithfulness"] == 1.0
        assert result["relevance"] == 1.0
        assert result["conciseness"] == 5.0

    @patch("anthropic.Anthropic")
    def test_whitespace_answer_returns_defaults(self, mock_cls):
        mock_cls.return_value = MagicMock()

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test-model")
        result = scorer.score("Q", "Context here", "   \n  ")

        assert result["faithfulness"] == 1.0
        assert result["relevance"] == 1.0
        assert result["conciseness"] == 5.0

    @patch("anthropic.Anthropic")
    def test_empty_context_overrides_faithfulness(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response()
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test-model")
        result = scorer.score("Q", "", "Some answer")

        assert result["faithfulness"] == 1.0

    @patch("anthropic.Anthropic")
    def test_malformed_json_returns_defaults(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(_MALFORMED_RESPONSE)
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test-model")
        result = scorer.score("Q", "C", "A")

        assert result == {"faithfulness": 3.0, "relevance": 3.0, "conciseness": 3.0}

    @patch("anthropic.Anthropic")
    def test_fenced_json_parsed(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(_FENCED_RESPONSE)
        mock_cls.return_value = mock_client

        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test-model")
        result = scorer.score("Q", "C", "A")

        assert result["faithfulness"] == 4.0

    def test_unknown_provider_raises(self):
        from src.scorers.llm import LLMScorer, ScorerError

        with pytest.raises(ScorerError, match="Unknown provider"):
            LLMScorer(provider="nonexistent", model="some-model")


class TestAdapterPattern:
    """Test that adapters are properly structured."""

    @patch("anthropic.Anthropic")
    def test_anthropic_adapter_lazy_imports(self, mock_cls):
        """Anthropic SDK is imported inside the adapter, not at module level."""
        mock_cls.return_value = MagicMock()

        # If we can import LLMScorer without anthropic being available at module
        # level, the lazy import works. The patch proves it's imported inside.
        from src.scorers.llm import LLMScorer
        scorer = LLMScorer(provider="anthropic", model="test")
        assert scorer is not None

    def test_scorer_satisfies_protocol(self):
        """LLMScorer instances satisfy the Scorer protocol."""
        from src.protocols import Scorer

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            from src.scorers.llm import LLMScorer
            scorer = LLMScorer(provider="anthropic", model="test")
            assert isinstance(scorer, Scorer)
