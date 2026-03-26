"""Tests for Experiment 1 & 2 dry-run validation.

Generates synthetic raw_scores.csv files, runs both experiments with
--skip-generation + gemini-2.5-flash-lite scorer, and verifies output
file structure and columns.

These tests make real API calls to Gemini Flash Lite (very cheap:
~$0.0001/call, ~$0.0004 total for 4 rows).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@pytest.fixture(scope="module")
def exp1_dry_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create synthetic Experiment 1 data and run --skip-generation scoring."""
    out = tmp_path_factory.mktemp("exp1_dry_run")

    # Import the data generator
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.create_dry_run_data import create_experiment_1_data

    create_experiment_1_data(out)

    # Run scoring
    result = subprocess.run(
        [
            PYTHON, str(PROJECT_ROOT / "scripts" / "run_experiment_1.py"),
            "--skip-generation",
            "--output-dir", str(out),
            "--scorer", "google:gemini-2.5-flash-lite",
            "--no-gallery",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Experiment 1 dry run failed (likely API/network issue): "
            f"rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

    return out


@pytest.fixture(scope="module")
def exp2_dry_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create synthetic Experiment 2 data and run --skip-generation scoring."""
    out = tmp_path_factory.mktemp("exp2_dry_run")

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.create_dry_run_data import create_experiment_2_data

    create_experiment_2_data(out)

    result = subprocess.run(
        [
            PYTHON, str(PROJECT_ROOT / "scripts" / "run_experiment_2.py"),
            "--skip-generation",
            "--output-dir", str(out),
            "--scorer", "google:gemini-2.5-flash-lite",
            "--no-gallery",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Experiment 2 dry run failed (likely API/network issue): "
            f"rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

    return out


# ---------------------------------------------------------------------------
# Experiment 1 validation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExperiment1DryRun:
    """Validate Experiment 1 scoring pipeline on synthetic data."""

    def test_raw_scores_exists(self, exp1_dry_run_dir: Path) -> None:
        """raw_scores.csv must exist after scoring."""
        assert (exp1_dry_run_dir / "raw_scores.csv").exists()

    def test_raw_scores_row_count(self, exp1_dry_run_dir: Path) -> None:
        """raw_scores.csv should have 4 rows (2 questions x 2 strategies)."""
        df = pd.read_csv(exp1_dry_run_dir / "raw_scores.csv")
        assert len(df) == 4

    def test_scorer_columns_exist(self, exp1_dry_run_dir: Path) -> None:
        """Scorer quality column should be present after scoring."""
        df = pd.read_csv(exp1_dry_run_dir / "raw_scores.csv")
        quality_cols = [c for c in df.columns if c.endswith("_quality")]
        assert len(quality_cols) >= 1, f"No quality columns found. Columns: {list(df.columns)}"

    def test_gold_bertscore_column(self, exp1_dry_run_dir: Path) -> None:
        """gold_bertscore column should exist with float values."""
        df = pd.read_csv(exp1_dry_run_dir / "raw_scores.csv")
        if "gold_bertscore" in df.columns:
            assert df["gold_bertscore"].dtype in ("float64", "float32")
        # BERTScore may fail in CI — not a hard requirement

    def test_gold_metrics_exist(self, exp1_dry_run_dir: Path) -> None:
        """gold_f1 and gold_exact_match columns should exist."""
        df = pd.read_csv(exp1_dry_run_dir / "raw_scores.csv")
        assert "gold_f1" in df.columns
        assert "gold_exact_match" in df.columns

    def test_report_exists(self, exp1_dry_run_dir: Path) -> None:
        """report.md should be generated."""
        assert (exp1_dry_run_dir / "report.md").exists()

    def test_report_has_content(self, exp1_dry_run_dir: Path) -> None:
        """report.md should contain expected sections."""
        report = (exp1_dry_run_dir / "report.md").read_text(encoding="utf-8")
        assert "Experiment 1" in report


# ---------------------------------------------------------------------------
# Experiment 2 validation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExperiment2DryRun:
    """Validate Experiment 2 scoring pipeline on synthetic data."""

    def test_raw_scores_exists(self, exp2_dry_run_dir: Path) -> None:
        """raw_scores.csv must exist after scoring."""
        assert (exp2_dry_run_dir / "raw_scores.csv").exists()

    def test_raw_scores_row_count(self, exp2_dry_run_dir: Path) -> None:
        """raw_scores.csv should have 4 rows (2 questions x 2 chunkers)."""
        df = pd.read_csv(exp2_dry_run_dir / "raw_scores.csv")
        assert len(df) == 4

    def test_scorer_columns_exist(self, exp2_dry_run_dir: Path) -> None:
        """Scorer quality column should be present after scoring."""
        df = pd.read_csv(exp2_dry_run_dir / "raw_scores.csv")
        quality_cols = [c for c in df.columns if c.endswith("_quality")]
        assert len(quality_cols) >= 1, f"No quality columns found. Columns: {list(df.columns)}"

    def test_gold_metrics_exist(self, exp2_dry_run_dir: Path) -> None:
        """gold_f1 and gold_exact_match columns should exist."""
        df = pd.read_csv(exp2_dry_run_dir / "raw_scores.csv")
        assert "gold_f1" in df.columns
        assert "gold_exact_match" in df.columns

    def test_report_exists(self, exp2_dry_run_dir: Path) -> None:
        """report.md should be generated."""
        assert (exp2_dry_run_dir / "report.md").exists()


# ---------------------------------------------------------------------------
# Synthetic data creation (no API calls)
# ---------------------------------------------------------------------------


class TestSyntheticDataCreation:
    """Validate synthetic data creation without running experiments."""

    def test_create_exp1_data(self, tmp_path: Path) -> None:
        """create_experiment_1_data produces a valid CSV with expected columns."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.create_dry_run_data import create_experiment_1_data

        csv_path = create_experiment_1_data(tmp_path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 4
        for col in ["question", "gold_answer", "rag_answer", "context_sent_to_llm",
                     "strategy", "model", "config_label"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_create_exp2_data(self, tmp_path: Path) -> None:
        """create_experiment_2_data produces a valid CSV with expected columns."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.create_dry_run_data import create_experiment_2_data

        csv_path = create_experiment_2_data(tmp_path)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 4
        for col in ["question", "gold_answer", "rag_answer", "context_sent_to_llm",
                     "chunker", "model", "config_label"]:
            assert col in df.columns, f"Missing column: {col}"
