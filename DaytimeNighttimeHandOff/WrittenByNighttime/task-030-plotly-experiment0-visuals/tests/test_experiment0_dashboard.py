"""Tests for Experiment 0 Plotly dashboard (task-030).

All tests use mock data — no real API calls or HotpotQA downloads.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# Minimal test data — 3 examples, 2 judges (one complete, one partial)
def _make_test_scores_df() -> pd.DataFrame:
    return pd.DataFrame({
        "example_id": [0, 1, 2],
        "question": [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What year did WW2 end?",
        ],
        "gold_answer": ["Paris", "Shakespeare", "1945"],
        "rag_answer": ["Paris", "William Shakespeare wrote Hamlet", "The war ended in 1945"],
        "gold_exact_match": [True, True, True],
        "gold_f1": [1.0, 0.5, 0.8],
        "gold_bertscore": [0.99, 0.95, 0.92],
        # Judge A — complete (3/3)
        "judge_a_faithfulness": [5, 4, 5],
        "judge_a_relevance": [5, 5, 4],
        "judge_a_conciseness": [5, 3, 5],
        "judge_a_quality": [5.0, 4.0, 4.67],
        # Judge B — partial (2/3, one NaN)
        "judge_b_faithfulness": [4, float("nan"), 5],
        "judge_b_relevance": [4, float("nan"), 5],
        "judge_b_conciseness": [4, float("nan"), 4],
        "judge_b_quality": [4.0, float("nan"), 4.67],
        # Judge C — all NaN (should be skipped)
        "judge_c_faithfulness": [float("nan")] * 3,
        "judge_c_relevance": [float("nan")] * 3,
        "judge_c_conciseness": [float("nan")] * 3,
        "judge_c_quality": [float("nan")] * 3,
    })


def _make_test_answers_df() -> pd.DataFrame:
    return pd.DataFrame({
        "example_id": [0, 1, 2],
        "question": [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What year did WW2 end?",
        ],
        "gold_answer": ["Paris", "Shakespeare", "1945"],
        "rag_answer": ["Paris", "William Shakespeare wrote Hamlet", "The war ended in 1945"],
        "doc_text": ["France doc...", "Hamlet doc...", "WW2 doc..."],
    })


@pytest.fixture
def test_data(tmp_path):
    """Set up test CSV files in a temporary directory."""
    results_dir = tmp_path / "results" / "experiment_0"
    results_dir.mkdir(parents=True)
    visuals_dir = tmp_path / "visuals"
    visuals_dir.mkdir()

    scores_df = _make_test_scores_df()
    answers_df = _make_test_answers_df()
    scores_df.to_csv(results_dir / "raw_scores.csv", index=False)
    answers_df.to_csv(results_dir / "raw_answers.csv", index=False)

    return {
        "results_dir": results_dir,
        "visuals_dir": visuals_dir,
        "scores_df": scores_df,
        "answers_df": answers_df,
        "tmp_path": tmp_path,
    }


class TestDashboardGeneration:
    """Verify the dashboard HTML is created with expected structure."""

    def test_dashboard_generates_html(self, test_data):
        """Script produces an HTML file with section headers."""
        from scripts.generate_experiment0_dashboard import generate_dashboard

        output_path = test_data["visuals_dir"] / "experiment_0.html"
        generate_dashboard(
            scores_path=test_data["results_dir"] / "raw_scores.csv",
            answers_path=test_data["results_dir"] / "raw_answers.csv",
            output_path=output_path,
            skip_enrichment=True,
        )

        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "Judge vs Gold" in html or "judge" in html.lower()
        assert "<html" in html


class TestEnrichment:
    """Verify HotpotQA metadata enrichment."""

    def test_enrichment_adds_columns(self, test_data):
        """Mock HotpotQA, verify difficulty and question_type columns added."""
        from scripts.generate_experiment0_dashboard import enrich_with_hotpotqa_metadata

        # Mock queries with metadata
        mock_queries = [
            MagicMock(metadata={"difficulty": "easy", "question_type": "bridge"}),
            MagicMock(metadata={"difficulty": "medium", "question_type": "comparison"}),
            MagicMock(metadata={"difficulty": "hard", "question_type": "bridge"}),
        ]

        df = test_data["scores_df"].copy()
        enriched = enrich_with_hotpotqa_metadata(df, mock_queries)

        assert "difficulty" in enriched.columns
        assert "question_type" in enriched.columns
        assert list(enriched["difficulty"]) == ["easy", "medium", "hard"]


class TestJudgeFiltering:
    """Verify all-NaN judges are excluded."""

    def test_skips_empty_judges(self):
        """Judges with all NaN quality scores should be excluded."""
        from scripts.generate_experiment0_dashboard import get_valid_judges

        df = _make_test_scores_df()
        valid = get_valid_judges(df, min_valid=1)

        judge_prefixes = [j["prefix"] for j in valid]
        assert "judge_a" in judge_prefixes
        assert "judge_b" in judge_prefixes
        assert "judge_c" not in judge_prefixes


class TestDisplayNames:
    """Verify column prefix to display name mapping."""

    def test_judge_display_names(self):
        """Known judge prefixes map to short display names."""
        from scripts.generate_experiment0_dashboard import JUDGE_DISPLAY_NAMES

        assert "google_gemini_2_5_flash" in JUDGE_DISPLAY_NAMES
        assert "google_gemini_3_1_pro_preview" in JUDGE_DISPLAY_NAMES
        assert JUDGE_DISPLAY_NAMES["google_gemini_2_5_flash"] == "Flash"


class TestPipelineWalkthrough:
    """Verify the walkthrough has all examples."""

    def test_pipeline_walkthrough_has_all_examples(self, test_data):
        """Generated HTML should have a select option for each example."""
        from scripts.generate_experiment0_dashboard import generate_dashboard

        output_path = test_data["visuals_dir"] / "experiment_0.html"
        generate_dashboard(
            scores_path=test_data["results_dir"] / "raw_scores.csv",
            answers_path=test_data["results_dir"] / "raw_answers.csv",
            output_path=output_path,
            skip_enrichment=True,
        )

        html = output_path.read_text(encoding="utf-8")
        # Each example should appear as a selectable option
        assert "example-0" in html or "What is the capital" in html
        assert "example-1" in html or "Who wrote Hamlet" in html
        assert "example-2" in html or "What year did WW2" in html
