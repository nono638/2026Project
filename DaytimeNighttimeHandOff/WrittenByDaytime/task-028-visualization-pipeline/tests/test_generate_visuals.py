"""Tests for scripts/generate_visuals.py.

Integration tests that run each visualization generator on real Experiment 0
data and verify the expected output files are created. Uses a temp directory
so tests don't pollute the project's visuals/ folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

RESULTS_DIR = PROJECT_ROOT / "results"
EXP0_DIR = RESULTS_DIR / "experiment_0"

# Skip entire module if experiment 0 data doesn't exist
pytestmark = pytest.mark.skipif(
    not (EXP0_DIR / "raw_scores.csv").exists(),
    reason="Experiment 0 results not found — run experiment first",
)


@pytest.fixture
def scores_df() -> pd.DataFrame:
    return pd.read_csv(EXP0_DIR / "raw_scores.csv")


@pytest.fixture
def answers_df() -> pd.DataFrame:
    return pd.read_csv(EXP0_DIR / "raw_answers.csv")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Temporary output directory for generated visuals."""
    exp0_dir = tmp_path / "experiment_0"
    exp0_dir.mkdir()
    return tmp_path


class TestExplainer:
    def test_creates_png(self, answers_df, scores_df, out_dir):
        from generate_visuals import generate_explainer

        generate_explainer(answers_df, scores_df, out_dir)
        assert (out_dir / "explainer_rag_pipeline.png").exists()

    def test_skips_if_no_answers(self, scores_df, out_dir):
        """Should not crash if raw_answers.csv is unavailable."""
        from generate_visuals import generate_explainer

        # Pass None to simulate missing answers
        generate_explainer(None, scores_df, out_dir)
        assert not (out_dir / "explainer_rag_pipeline.png").exists()


class TestCorrelationHeatmap:
    def test_creates_png(self, scores_df, out_dir):
        from generate_visuals import generate_exp0_heatmap

        generate_exp0_heatmap(scores_df, out_dir / "experiment_0")
        assert (out_dir / "experiment_0" / "judge_correlation_heatmap.png").exists()

    def test_skips_if_fewer_than_2_judges(self, out_dir):
        """Heatmap needs at least 2 judges with data."""
        from generate_visuals import generate_exp0_heatmap

        # DataFrame with only one quality column
        df = pd.DataFrame({
            "anthropic_claude_sonnet_4_20250514_quality": [4.0, 5.0, 3.0],
        })
        generate_exp0_heatmap(df, out_dir / "experiment_0")
        assert not (out_dir / "experiment_0" / "judge_correlation_heatmap.png").exists()


class TestJudgeVsGoldScatter:
    def test_creates_png(self, scores_df, out_dir):
        from generate_visuals import generate_exp0_scatter

        generate_exp0_scatter(scores_df, out_dir / "experiment_0")
        assert (out_dir / "experiment_0" / "judge_vs_gold_scatter.png").exists()

    def test_skips_if_no_bertscore(self, out_dir):
        """Should skip if gold_bertscore column is missing."""
        from generate_visuals import generate_exp0_scatter

        df = pd.DataFrame({
            "anthropic_claude_sonnet_4_20250514_quality": [4.0, 5.0, 3.0],
        })
        generate_exp0_scatter(df, out_dir / "experiment_0")
        assert not (out_dir / "experiment_0" / "judge_vs_gold_scatter.png").exists()


class TestScoreDistributions:
    def test_creates_png(self, scores_df, out_dir):
        from generate_visuals import generate_exp0_distributions

        generate_exp0_distributions(scores_df, out_dir / "experiment_0")
        assert (out_dir / "experiment_0" / "score_distributions.png").exists()


class TestBertscoreHistogram:
    def test_creates_png(self, scores_df, out_dir):
        from generate_visuals import generate_exp0_bertscore_hist

        generate_exp0_bertscore_hist(scores_df, out_dir / "experiment_0")
        assert (out_dir / "experiment_0" / "bertscore_distribution.png").exists()

    def test_skips_if_no_bertscore_column(self, out_dir):
        from generate_visuals import generate_exp0_bertscore_hist

        df = pd.DataFrame({"some_col": [1, 2, 3]})
        generate_exp0_bertscore_hist(df, out_dir / "experiment_0")
        assert not (out_dir / "experiment_0" / "bertscore_distribution.png").exists()


class TestHtmlGallery:
    def test_creates_html(self, scores_df, answers_df, out_dir):
        """Generate all visuals, then check HTML references them."""
        from generate_visuals import (
            generate_explainer,
            generate_exp0_heatmap,
            generate_exp0_scatter,
            generate_exp0_distributions,
            generate_exp0_bertscore_hist,
            generate_html,
        )

        generate_explainer(answers_df, scores_df, out_dir)
        exp0_out = out_dir / "experiment_0"
        generate_exp0_heatmap(scores_df, exp0_out)
        generate_exp0_scatter(scores_df, exp0_out)
        generate_exp0_distributions(scores_df, exp0_out)
        generate_exp0_bertscore_hist(scores_df, exp0_out)
        generate_html(out_dir)

        html_path = out_dir / "index.html"
        assert html_path.exists()

        html = html_path.read_text()
        assert "explainer_rag_pipeline.png" in html
        assert "judge_correlation_heatmap.png" in html
        assert "RAGBench" in html

    def test_html_handles_missing_images(self, out_dir):
        """HTML should still generate even if no PNGs exist."""
        from generate_visuals import generate_html

        generate_html(out_dir)
        html_path = out_dir / "index.html"
        assert html_path.exists()
        # Should contain "Not yet generated" or similar for missing images
        html = html_path.read_text()
        assert "RAGBench" in html


class TestDisplayNames:
    def test_all_quality_columns_have_display_names(self, scores_df):
        """Every *_quality column in the real data should map to a display name."""
        from generate_visuals import JUDGE_DISPLAY_NAMES

        quality_cols = [c for c in scores_df.columns if c.endswith("_quality")]
        for col in quality_cols:
            prefix = col.removesuffix("_quality")
            assert prefix in JUDGE_DISPLAY_NAMES, (
                f"Column prefix '{prefix}' has no display name mapping"
            )
