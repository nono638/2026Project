"""Tests for the findings gallery static site generator (task-035).

Tests the gallery generator's page building, data handling, and output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_exp0_csv(tmp_path: Path) -> Path:
    """Create a minimal Experiment 0 CSV for testing."""
    rng = np.random.RandomState(42)
    n = 10
    data = {
        "example_id": list(range(n)),
        "question": [f"Question {i}?" for i in range(n)],
        "gold_answer": [f"Answer {i}" for i in range(n)],
        "rag_answer": [f"RAG answer {i}" for i in range(n)],
        "gold_exact_match": rng.choice([0, 1], n),
        "gold_f1": rng.uniform(0.3, 1.0, n),
        "google_gemini_2_5_flash_faithfulness": rng.uniform(3, 5, n),
        "google_gemini_2_5_flash_relevance": rng.uniform(3, 5, n),
        "google_gemini_2_5_flash_conciseness": rng.uniform(3, 5, n),
        "google_gemini_2_5_flash_quality": rng.uniform(3, 5, n),
        "anthropic_claude_haiku_4_5_20251001_faithfulness": rng.uniform(3, 5, n),
        "anthropic_claude_haiku_4_5_20251001_relevance": rng.uniform(3, 5, n),
        "anthropic_claude_haiku_4_5_20251001_conciseness": rng.uniform(3, 5, n),
        "anthropic_claude_haiku_4_5_20251001_quality": rng.uniform(3, 5, n),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "results" / "experiment_0" / "raw_scores.csv"
    csv_path.parent.mkdir(parents=True)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def results_dir(sample_exp0_csv: Path) -> Path:
    """Return the results/ directory containing experiment data."""
    return sample_exp0_csv.parent.parent


# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------

class TestPageTemplate:
    """Tests for the shared page template."""

    def test_template_contains_title(self) -> None:
        """Page template includes the given title in <title> and <h1>."""
        from scripts.generate_gallery import _build_page_template

        html = _build_page_template("Test Page", nav_active="home", content_html="<p>Hello</p>")
        assert "<title>" in html
        assert "Test Page" in html

    def test_template_contains_nav(self) -> None:
        """Page template includes navigation links."""
        from scripts.generate_gallery import _build_page_template

        html = _build_page_template("Test", nav_active="home", content_html="")
        assert "index.html" in html or "Home" in html
        assert "experiment_0" in html.lower() or "Exp 0" in html

    def test_template_contains_content(self) -> None:
        """Page template includes the provided content HTML."""
        from scripts.generate_gallery import _build_page_template

        content = "<div class='test-content'>Custom content here</div>"
        html = _build_page_template("Test", nav_active="home", content_html=content)
        assert "Custom content here" in html

    def test_template_has_inline_css(self) -> None:
        """Page template has inline CSS (no external stylesheet links)."""
        from scripts.generate_gallery import _build_page_template

        html = _build_page_template("Test", nav_active="home", content_html="")
        assert "<style" in html
        # Should NOT have external CSS links
        assert 'rel="stylesheet"' not in html


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

class TestIndexPage:
    """Tests for the index/landing page."""

    def test_index_has_experiment_links(self) -> None:
        """Index page links to each experiment dashboard."""
        from scripts.generate_gallery import _generate_index

        experiments_info = [
            {"num": 0, "title": "Scorer Validation", "status": "ready"},
            {"num": 1, "title": "Strategy × Model", "status": "placeholder"},
        ]
        html = _generate_index(experiments_info)
        # Exp 0 is linked via the hero section (v3 page), not a card
        assert "experiment_0_v3.html" in html
        assert "experiment_1.html" in html

    def test_index_shows_project_description(self) -> None:
        """Index page contains a project summary."""
        from scripts.generate_gallery import _generate_index

        html = _generate_index([])
        # Should contain some mention of RAGBench or the project
        assert "RAGBench" in html or "RAG" in html


# ---------------------------------------------------------------------------
# Experiment 0 dashboard
# ---------------------------------------------------------------------------

class TestExperiment0Dashboard:
    """Tests for Experiment 0 dashboard generation."""

    def test_generates_html_from_csv(self, sample_exp0_csv: Path) -> None:
        """Dashboard generates valid HTML from Exp 0 CSV."""
        from scripts.generate_gallery import _generate_experiment_0

        html = _generate_experiment_0(sample_exp0_csv)
        assert "<html" in html.lower() or "<div" in html.lower()
        # Should contain Plotly chart divs
        assert "plotly" in html.lower()

    def test_contains_judge_analysis(self, sample_exp0_csv: Path) -> None:
        """Dashboard includes judge comparison content."""
        from scripts.generate_gallery import _generate_experiment_0

        html = _generate_experiment_0(sample_exp0_csv)
        # Should reference judges or scorers
        assert "judge" in html.lower() or "scorer" in html.lower() or "flash" in html.lower()


# ---------------------------------------------------------------------------
# Placeholder pages
# ---------------------------------------------------------------------------

class TestPlaceholderPages:
    """Tests for placeholder pages when experiment data doesn't exist."""

    def test_placeholder_has_coming_soon(self) -> None:
        """Placeholder page indicates the experiment is not yet complete."""
        from scripts.generate_gallery import _generate_placeholder

        html = _generate_placeholder(
            experiment_num=1,
            description="5 strategies × 6 models = 30 configurations",
        )
        assert "coming soon" in html.lower() or "not yet" in html.lower() or "planned" in html.lower()

    def test_placeholder_includes_description(self) -> None:
        """Placeholder page shows the planned experiment description."""
        from scripts.generate_gallery import _generate_placeholder

        desc = "5 strategies × 6 models = 30 configurations"
        html = _generate_placeholder(experiment_num=1, description=desc)
        assert "30 configurations" in html or "5 strategies" in html


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------

class TestFullGeneration:
    """Tests for end-to-end gallery generation."""

    def test_generates_output_directory(self, results_dir: Path, tmp_path: Path) -> None:
        """Gallery generator creates the output directory and index.html."""
        from scripts.generate_gallery import main as generate_main

        output_dir = tmp_path / "site"
        generate_main(results_dir=results_dir, output_dir=output_dir)

        assert output_dir.exists()
        assert (output_dir / "index.html").exists()

    def test_generates_experiment_0_page(self, results_dir: Path, tmp_path: Path) -> None:
        """Gallery generator creates experiment_0.html when data exists."""
        from scripts.generate_gallery import main as generate_main

        output_dir = tmp_path / "site"
        generate_main(results_dir=results_dir, output_dir=output_dir)

        assert (output_dir / "experiment_0.html").exists()
        content = (output_dir / "experiment_0.html").read_text()
        assert "plotly" in content.lower()

    def test_generates_placeholder_for_missing_experiment(self, results_dir: Path, tmp_path: Path) -> None:
        """Gallery generates placeholder when experiment data doesn't exist."""
        from scripts.generate_gallery import main as generate_main

        output_dir = tmp_path / "site"
        generate_main(results_dir=results_dir, output_dir=output_dir)

        # Experiment 1 data doesn't exist in the fixture
        assert (output_dir / "experiment_1.html").exists()
        content = (output_dir / "experiment_1.html").read_text()
        assert "coming soon" in content.lower() or "planned" in content.lower() or "not yet" in content.lower()

    def test_output_dir_created_if_missing(self, results_dir: Path, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        from scripts.generate_gallery import main as generate_main

        output_dir = tmp_path / "new_site_dir"
        assert not output_dir.exists()
        generate_main(results_dir=results_dir, output_dir=output_dir)
        assert output_dir.exists()


# ---------------------------------------------------------------------------
# build_experiment0_figures (refactored from existing dashboard)
# ---------------------------------------------------------------------------

class TestBuildExperiment0Figures:
    """Tests for the extracted figure-building function."""

    def test_returns_list_of_tuples(self, sample_exp0_csv: Path) -> None:
        """build_experiment0_figures returns list of (title, figure) tuples."""
        from scripts.generate_experiment0_dashboard import build_experiment0_figures

        df = pd.read_csv(sample_exp0_csv)
        figures = build_experiment0_figures(df)

        assert isinstance(figures, list)
        assert len(figures) > 0
        for title, fig in figures:
            assert isinstance(title, str)
            assert len(title) > 0
            # Should be a Plotly figure
            assert hasattr(fig, "to_html")
