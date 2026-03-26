"""Tests for task-044: Gallery MVP Polish.

Validates the upgraded landing page, new methodology page, and navigation changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GALLERY_SCRIPT = PROJECT_ROOT / "scripts" / "generate_gallery.py"
SITE_DIR = PROJECT_ROOT / "site"


@pytest.fixture(scope="module")
def generated_site(tmp_path_factory):
    """Run the gallery generator and return the output directory."""
    out = tmp_path_factory.mktemp("site")
    result = subprocess.run(
        [sys.executable, str(GALLERY_SCRIPT), "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, f"Gallery generation failed:\n{result.stderr}"
    return out


# ---------------------------------------------------------------------------
# Landing page (index.html)
# ---------------------------------------------------------------------------


class TestLandingPage:
    """Tests for the upgraded index.html."""

    @pytest.fixture(autouse=True)
    def _load_index(self, generated_site):
        self.html = (generated_site / "index.html").read_text(encoding="utf-8")

    def test_hero_section_exists(self):
        """Hero section has project title and tagline."""
        assert "RAGBench" in self.html
        assert "Retrieval-Augmented Generation" in self.html

    def test_hero_has_description(self):
        """Hero section has a project description paragraph."""
        assert "cartesian product" in self.html.lower() or "evaluation pipeline" in self.html.lower()

    def test_key_findings_section(self):
        """Key findings from Experiment 0 are displayed."""
        assert "Claude Sonnet" in self.html or "most accurate" in self.html.lower()
        assert "Gemini Flash" in self.html or "budget" in self.html.lower()

    def test_findings_count(self):
        """At least 3 key findings are shown."""
        # Each finding should be in its own card/container — count by checking
        # for distinct finding keywords
        findings_markers = [
            "most accurate",
            "budget",
            "inter-judge",
            "answer quality",
        ]
        found = sum(1 for m in findings_markers if m in self.html.lower())
        assert found >= 3, f"Expected at least 3 findings, found {found}"

    def test_experiment_cards_still_present(self):
        """Experiment cards for 0, 1, 2 are still on the page."""
        assert "Experiment 0" in self.html or "experiment_0" in self.html
        assert "Experiment 1" in self.html or "experiment_1" in self.html
        assert "Experiment 2" in self.html or "experiment_2" in self.html

    def test_footer_present(self):
        """Footer with attribution exists."""
        assert "Noah" in self.html or "CUNY" in self.html

    def test_methodology_link(self):
        """Landing page links to the methodology page."""
        assert "methodology.html" in self.html


# ---------------------------------------------------------------------------
# Methodology page
# ---------------------------------------------------------------------------


class TestMethodologyPage:
    """Tests for the new methodology.html page."""

    @pytest.fixture(autouse=True)
    def _load_methodology(self, generated_site):
        path = generated_site / "methodology.html"
        assert path.exists(), "methodology.html was not generated"
        self.html = path.read_text(encoding="utf-8")

    def test_pipeline_diagram(self):
        """Pipeline overview includes a flow diagram."""
        assert "QueryGenerator" in self.html or "Scorer" in self.html

    def test_four_axes_explained(self):
        """All four axes (chunker, embedder, strategy, model) are described."""
        html_lower = self.html.lower()
        assert "chunker" in html_lower
        assert "embedder" in html_lower
        assert "strategy" in html_lower
        assert "model" in html_lower

    def test_evaluation_approach(self):
        """Dual evaluation (intrinsic + extrinsic) is explained."""
        html_lower = self.html.lower()
        assert "intrinsic" in html_lower or "faithfulness" in html_lower
        assert "extrinsic" in html_lower or "gold" in html_lower

    def test_experiment_design_table(self):
        """Experiment design section covers all three experiments."""
        assert "Experiment 0" in self.html or "Scorer Validation" in self.html
        assert "Experiment 1" in self.html or "Strategy" in self.html
        assert "Experiment 2" in self.html or "Chunking" in self.html

    def test_dataset_mention(self):
        """HotpotQA is mentioned as the primary dataset."""
        assert "HotpotQA" in self.html

    def test_shared_nav(self):
        """Methodology page uses the shared navigation bar."""
        assert "nav" in self.html.lower()
        assert "index.html" in self.html


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    """Tests for updated navigation across all pages."""

    @pytest.fixture(autouse=True)
    def _load_pages(self, generated_site):
        self.pages = {}
        for name in ["index.html", "methodology.html", "experiment_0.html"]:
            path = generated_site / name
            if path.exists():
                self.pages[name] = path.read_text(encoding="utf-8")

    def test_nav_has_methodology_link(self):
        """All pages link to methodology from the nav."""
        for name, html in self.pages.items():
            assert "methodology" in html.lower(), (
                f"{name} missing methodology link in nav"
            )

    def test_nav_has_home_link(self):
        """All pages link back to index."""
        for name, html in self.pages.items():
            assert "index.html" in html or 'href="./"' in html or 'href="/"' in html, (
                f"{name} missing home link in nav"
            )

    def test_nav_has_experiment_links(self):
        """Nav includes links to experiment pages."""
        for name, html in self.pages.items():
            assert "experiment_0" in html, f"{name} missing exp 0 link"


# ---------------------------------------------------------------------------
# CSS / Responsive
# ---------------------------------------------------------------------------


class TestCSS:
    """Tests for CSS polish."""

    @pytest.fixture(autouse=True)
    def _load_index(self, generated_site):
        self.html = (generated_site / "index.html").read_text(encoding="utf-8")

    def test_hero_has_gradient_or_dark_bg(self):
        """Hero section has a dark/gradient background style."""
        # Check for gradient or dark background color in CSS
        assert "gradient" in self.html.lower() or "#1a1a2e" in self.html

    def test_responsive_media_query(self):
        """CSS includes a responsive breakpoint."""
        assert "@media" in self.html

    def test_finding_cards_have_shadow_or_border(self):
        """Finding cards have visual distinction (shadow or border)."""
        assert "box-shadow" in self.html or "border-left" in self.html


# ---------------------------------------------------------------------------
# Generation flags
# ---------------------------------------------------------------------------


class TestGenerationFlags:
    """Methodology page generates regardless of --experiments flag."""

    def test_methodology_generates_with_experiments_flag(self, tmp_path):
        """--experiments 0 still generates methodology.html."""
        result = subprocess.run(
            [
                sys.executable,
                str(GALLERY_SCRIPT),
                "--output",
                str(tmp_path),
                "--experiments",
                "0",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert (tmp_path / "methodology.html").exists(), (
            "methodology.html not generated when --experiments 0 is used"
        )
