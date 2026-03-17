"""Tests for ClaudeScorer — all API calls mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import helper — src.scorers may not exist until the night agent creates it
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_anthropic():
    """Mock the anthropic module so tests run without the SDK installed."""
    mock_mod = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        yield mock_mod


@pytest.fixture
def scorer(_mock_anthropic):
    """Create a ClaudeScorer with a mocked Anthropic client."""
    from src.scorers.claude import ClaudeScorer
    s = ClaudeScorer(model="claude-sonnet-4-20250514", api_key="test-key")
    return s


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_has_name(self, scorer):
        assert hasattr(scorer, "name")
        assert "claude" in scorer.name

    def test_has_score(self, scorer):
        assert callable(getattr(scorer, "score", None))

    def test_implements_scorer_protocol(self, scorer):
        from src.protocols import Scorer
        assert isinstance(scorer, Scorer)


# ---------------------------------------------------------------------------
# Name format
# ---------------------------------------------------------------------------

class TestName:
    def test_name_includes_model(self, scorer):
        assert scorer.name == "claude:claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_query(self, scorer):
        prompt = scorer._build_prompt("What is X?", "X is Y.", "X is Y.")
        assert "What is X?" in prompt

    def test_prompt_contains_context(self, scorer):
        prompt = scorer._build_prompt("q", "The context text here.", "a")
        assert "The context text here." in prompt

    def test_prompt_contains_answer(self, scorer):
        prompt = scorer._build_prompt("q", "c", "The answer text here.")
        assert "The answer text here." in prompt

    def test_prompt_contains_rubric(self, scorer):
        prompt = scorer._build_prompt("q", "c", "a")
        assert "faithfulness" in prompt.lower()
        assert "relevance" in prompt.lower()
        assert "conciseness" in prompt.lower()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_clean_json(self, scorer):
        response = json.dumps({
            "faithfulness": 4, "relevance": 5, "conciseness": 3,
            "reasoning": {"faithfulness": "good", "relevance": "great", "conciseness": "ok"}
        })
        result = scorer._parse_response(response)
        assert result["faithfulness"] == 4.0
        assert result["relevance"] == 5.0
        assert result["conciseness"] == 3.0

    def test_json_in_code_fence(self, scorer):
        response = '```json\n{"faithfulness": 4, "relevance": 5, "conciseness": 3}\n```'
        result = scorer._parse_response(response)
        assert result["faithfulness"] == 4.0

    def test_missing_key_defaults_to_3(self, scorer):
        response = json.dumps({"faithfulness": 4, "relevance": 5})
        result = scorer._parse_response(response)
        assert result["conciseness"] == 3.0

    def test_non_numeric_defaults_to_3(self, scorer):
        response = json.dumps({"faithfulness": "high", "relevance": 5, "conciseness": 3})
        result = scorer._parse_response(response)
        assert result["faithfulness"] == 3.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_answer(self, scorer, _mock_anthropic):
        """Empty answer should return 1/1/5 without calling the API."""
        result = scorer.score("What is X?", "X is Y.", "")
        assert result["faithfulness"] == 1.0
        assert result["relevance"] == 1.0
        assert result["conciseness"] == 5.0

    def test_empty_context_scores_faithfulness_1(self, scorer, _mock_anthropic):
        """Empty context should score faithfulness as 1.0."""
        # Mock the API response for relevance/conciseness
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "faithfulness": 1, "relevance": 3, "conciseness": 4,
            "reasoning": {"faithfulness": "no context", "relevance": "ok", "conciseness": "ok"}
        })
        scorer._client.messages.create.return_value = mock_response
        result = scorer.score("What is X?", "", "X is Y.")
        assert result["faithfulness"] == 1.0


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class TestBatchScoring:
    def test_batch_returns_list(self, scorer, _mock_anthropic):
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "faithfulness": 4, "relevance": 5, "conciseness": 3
        })
        scorer._client.messages.create.return_value = mock_response

        items = [
            {"query": "q1", "context": "c1", "answer": "a1"},
            {"query": "q2", "context": "c2", "answer": "a2"},
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2
        assert all("faithfulness" in r for r in results)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_api_error_raises_scorer_error(self, scorer, _mock_anthropic):
        from src.scorers.claude import ScorerError
        scorer._client.messages.create.side_effect = Exception("API down")
        with pytest.raises(ScorerError, match="API down"):
            scorer.score("q", "c", "a")

    def test_missing_api_key_raises(self, _mock_anthropic):
        """If no API key and env var not set, constructor should raise."""
        _mock_anthropic.Anthropic.side_effect = Exception("No API key")
        from src.scorers.claude import ClaudeScorer, ScorerError
        with pytest.raises((ScorerError, Exception)):
            ClaudeScorer(api_key=None)
