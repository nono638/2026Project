"""Tests for BERTScore computation in the experiment pipeline.

Tests compute_bertscores() from both experiment_utils.py and
run_experiment_0.py. Uses the actual bert_score library (no mocking)
since the function is a thin wrapper and we need to verify the real
integration works end-to-end.

The RoBERTa-large model (~1.4GB) is downloaded on first run and cached
by HuggingFace. Subsequent runs use the cache.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import compute_bertscores as utils_bertscores


# Import the run_experiment_0 copy for parity testing
def _import_exp0_bertscores():
    """Import compute_bertscores from run_experiment_0.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_experiment_0",
        PROJECT_ROOT / "scripts" / "run_experiment_0.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_bertscores


exp0_bertscores = _import_exp0_bertscores()


# ---------------------------------------------------------------------------
# Tests for experiment_utils.compute_bertscores
# ---------------------------------------------------------------------------


class TestComputeBertscores:
    """Tests for compute_bertscores from experiment_utils."""

    def test_returns_list_of_correct_length(self):
        preds = ["The capital of France is Paris.", "Water boils at 100 degrees."]
        golds = ["Paris is the capital of France.", "Water boils at 100 degrees Celsius."]
        result = utils_bertscores(preds, golds)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_scores_are_floats_between_0_and_1(self):
        preds = ["Hello world"]
        golds = ["Hello world"]
        result = utils_bertscores(preds, golds)
        assert all(isinstance(s, float) for s in result)
        assert all(0.0 <= s <= 1.0 for s in result)

    def test_identical_strings_score_near_one(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = utils_bertscores([text], [text])
        assert result[0] > 0.99

    def test_similar_strings_score_high(self):
        pred = "Paris is the capital city of France."
        gold = "The capital of France is Paris."
        result = utils_bertscores([pred], [gold])
        # Semantically similar sentences should score above 0.8
        assert result[0] > 0.8

    def test_unrelated_strings_score_lower(self):
        pred = "The weather is sunny today."
        gold = "Quantum mechanics describes subatomic particles."
        result = utils_bertscores([pred], [gold])
        identical = utils_bertscores([pred], [pred])
        # Unrelated text should score meaningfully lower than identical
        assert result[0] < identical[0]

    def test_empty_prediction_returns_zero(self):
        """Empty prediction should score 0.0, not crash."""
        result = utils_bertscores([""], ["The answer is 42."])
        assert len(result) == 1
        assert result[0] == 0.0

    def test_empty_gold_returns_zero(self):
        """Empty gold should score 0.0, not crash."""
        result = utils_bertscores(["The answer is 42."], [""])
        assert len(result) == 1
        assert result[0] == 0.0

    def test_both_empty_returns_zero(self):
        """Both empty should score 0.0, not crash."""
        result = utils_bertscores([""], [""])
        assert len(result) == 1
        assert result[0] == 0.0

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only strings should be treated as empty."""
        result = utils_bertscores(["   "], ["The answer is 42."])
        assert result[0] == 0.0

    def test_mixed_empty_and_nonempty(self):
        """Empty pairs get 0.0 without affecting non-empty pairs."""
        preds = ["The capital of France is Paris.", "", "Water boils at 100 degrees."]
        golds = ["Paris is the capital of France.", "Some answer", "Water boils at 100 degrees Celsius."]
        result = utils_bertscores(preds, golds)
        assert len(result) == 3
        assert result[0] > 0.8  # non-empty similar pair
        assert result[1] == 0.0  # empty prediction
        assert result[2] > 0.8  # non-empty similar pair

    def test_multiple_pairs(self):
        preds = [
            "Paris is the capital of France.",
            "The sky is blue.",
            "Python is a programming language.",
        ]
        golds = [
            "The capital of France is Paris.",
            "The sky appears blue during the day.",
            "Python is a widely-used programming language.",
        ]
        result = utils_bertscores(preds, golds)
        assert len(result) == 3
        # All semantically similar pairs should score reasonably high
        assert all(s > 0.8 for s in result)

    def test_single_pair(self):
        result = utils_bertscores(["hello"], ["hello"])
        assert len(result) == 1


class TestExp0BertscoresParity:
    """Verify the run_experiment_0.py copy produces identical results."""

    def test_same_output_as_utils(self):
        preds = ["The Earth revolves around the Sun."]
        golds = ["The Sun is orbited by the Earth."]
        utils_result = utils_bertscores(preds, golds)
        exp0_result = exp0_bertscores(preds, golds)
        # Both copies should produce the exact same score
        assert len(utils_result) == len(exp0_result)
        for u, e in zip(utils_result, exp0_result):
            assert abs(u - e) < 1e-6, f"Parity mismatch: utils={u}, exp0={e}"
