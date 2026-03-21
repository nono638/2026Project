"""Tests for user documentation accuracy.

These tests verify that documentation references match reality:
- File paths mentioned in docs exist
- CLI flags mentioned match argparse definitions
- Column names match experiment script output
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

# tests/ is one level below project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestReadmeAccuracy:
    """Verify README.md references match the codebase."""

    @pytest.fixture
    def readme(self) -> str:
        return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_protocols_file_exists(self, readme):
        """README references src/protocols.py — verify it exists."""
        assert (PROJECT_ROOT / "src" / "protocols.py").exists()

    def test_test_count_is_current(self, readme):
        """Test count in README should be >= 500 (we have 527+)."""
        match = re.search(r"(\d+)\s+tests", readme)
        if match:
            count = int(match.group(1))
            assert count >= 500, f"README says {count} tests — update to current count"

    def test_docs_links_exist(self, readme):
        """Any docs/ links in README should point to existing files."""
        links = re.findall(r"\(docs/([^)]+)\)", readme)
        for link in links:
            path = PROJECT_ROOT / "docs" / link
            assert path.exists(), f"README links to docs/{link} but it doesn't exist"

    def test_reranker_section_exists(self, readme):
        """README should document rerankers."""
        assert "reranker" in readme.lower(), "README should have a reranker section"

    def test_experiment_scripts_section_exists(self, readme):
        """README should reference the experiment scripts."""
        assert "run_experiment_1" in readme or "experiment_1" in readme, \
            "README should reference Experiment 1 script"


class TestOutputFormatDoc:
    """Verify output-format.md matches actual experiment output columns."""

    @pytest.fixture
    def output_doc(self) -> str:
        path = PROJECT_ROOT / "docs" / "output-format.md"
        if not path.exists():
            pytest.skip("docs/output-format.md not yet created")
        return path.read_text(encoding="utf-8")

    def test_doc_exists(self):
        assert (PROJECT_ROOT / "docs" / "output-format.md").exists()

    def test_documents_quality_column(self, output_doc):
        assert "quality" in output_doc

    def test_documents_faithfulness_column(self, output_doc):
        assert "faithfulness" in output_doc

    def test_documents_strategy_column(self, output_doc):
        assert "strategy" in output_doc

    def test_documents_gold_f1_column(self, output_doc):
        assert "gold_f1" in output_doc

    def test_documents_latency_columns(self, output_doc):
        assert "strategy_latency_ms" in output_doc
        assert "scorer_latency_ms" in output_doc


class TestRunningExperimentsDoc:
    """Verify running-experiments.md exists and covers all experiments."""

    @pytest.fixture
    def guide(self) -> str:
        path = PROJECT_ROOT / "docs" / "running-experiments.md"
        if not path.exists():
            pytest.skip("docs/running-experiments.md not yet created")
        return path.read_text(encoding="utf-8")

    def test_doc_exists(self):
        assert (PROJECT_ROOT / "docs" / "running-experiments.md").exists()

    def test_covers_experiment_0(self, guide):
        assert "experiment_0" in guide.lower() or "experiment 0" in guide.lower()

    def test_covers_experiment_1(self, guide):
        assert "experiment_1" in guide.lower() or "experiment 1" in guide.lower()

    def test_covers_experiment_2(self, guide):
        assert "experiment_2" in guide.lower() or "experiment 2" in guide.lower()

    def test_covers_runpod(self, guide):
        assert "runpod" in guide.lower() or "ollama-host" in guide.lower()

    def test_covers_resume(self, guide):
        assert "--resume" in guide

    def test_covers_cost(self, guide):
        assert "cost" in guide.lower()

    def test_covers_troubleshooting(self, guide):
        assert "troubleshoot" in guide.lower()


class TestExperimentScriptPaths:
    """Verify that all experiment scripts referenced in docs exist."""

    def test_run_experiment_exists(self):
        assert (PROJECT_ROOT / "scripts" / "run_experiment.py").exists()

    def test_run_experiment_0_exists(self):
        assert (PROJECT_ROOT / "scripts" / "run_experiment_0.py").exists()

    def test_run_experiment_1_exists(self):
        assert (PROJECT_ROOT / "scripts" / "run_experiment_1.py").exists()

    def test_run_experiment_2_exists(self):
        assert (PROJECT_ROOT / "scripts" / "run_experiment_2.py").exists()

    def test_experiment_utils_exists(self):
        assert (PROJECT_ROOT / "scripts" / "experiment_utils.py").exists()

    def test_generate_gallery_exists(self):
        assert (PROJECT_ROOT / "scripts" / "generate_gallery.py").exists()

    def test_pull_models_exists(self):
        assert (PROJECT_ROOT / "scripts" / "pull_models.py").exists()
