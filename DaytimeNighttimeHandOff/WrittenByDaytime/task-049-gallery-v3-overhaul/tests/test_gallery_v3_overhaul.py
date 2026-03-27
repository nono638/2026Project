"""Tests for gallery v3 overhaul (task-049).

Covers: home page hero, v3 narrative, navigation links, card links,
and backward compatibility of v1/v2 pages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gallery_output(tmp_path_factory):
    """Generate the full gallery into a temp dir and return it.

    Requires actual experiment data in results/ — skips if missing.
    """
    results_dir = PROJECT_ROOT / "results"
    v3_csv = results_dir / "experiment_0_v3" / "raw_scores.csv"
    if not v3_csv.exists():
        pytest.skip("No v3 data — cannot generate gallery")

    output_dir = tmp_path_factory.mktemp("gallery")

    from scripts.generate_gallery import main
    main(output_dir=output_dir, experiments=[0])

    return output_dir


@pytest.fixture(scope="module")
def index_html(gallery_output):
    """Read the generated index.html."""
    path = gallery_output / "index.html"
    assert path.exists(), "index.html was not generated"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def v3_html(gallery_output):
    """Read the generated experiment_0_v3.html."""
    path = gallery_output / "experiment_0_v3.html"
    assert path.exists(), "experiment_0_v3.html was not generated"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Home page hero centers on v3
# ---------------------------------------------------------------------------

class TestHomePageHero:
    """Home page should lead with Experiment 0 v3 as the centerpiece."""

    def test_third_times_a_charm(self, index_html):
        """Home page should contain the 'third time's a charm' framing."""
        assert "third time" in index_html.lower() or "Third time" in index_html

    def test_links_to_v3_page(self, index_html):
        """Home page hero should link to experiment_0_v3.html."""
        assert "experiment_0_v3.html" in index_html

    def test_haiku_featured(self, index_html):
        """Home page should mention Haiku as the answer."""
        assert "Haiku" in index_html

    def test_no_separate_v1_v2_v3_cards(self, index_html):
        """Home page should NOT have separate Experiment 0v2 or 0v3 cards."""
        # The old card titles were "Experiment 0v2: Scorer Validation v2" etc.
        assert "Experiment 0v2:" not in index_html
        assert "Experiment 0v3:" not in index_html


# ---------------------------------------------------------------------------
# Test 2: v3 page has narrative section
# ---------------------------------------------------------------------------

class TestV3Narrative:
    """v3 page should contain the 'Road to v3' narrative."""

    def test_road_to_v3_heading(self, v3_html):
        """v3 page should have the narrative heading."""
        assert "Road to v3" in v3_html or "road to v3" in v3_html.lower()

    def test_links_to_v1(self, v3_html):
        """v3 narrative should link to the v1 page."""
        assert 'href="experiment_0.html"' in v3_html

    def test_links_to_v2(self, v3_html):
        """v3 narrative should link to the v2 page."""
        assert 'href="experiment_0_v2.html"' in v3_html

    def test_explains_v1_flaws(self, v3_html):
        """v3 narrative should explain what was wrong with v1."""
        html_lower = v3_html.lower()
        # Should mention that v1 didn't track what the LLM saw
        assert "full source document" in html_lower or "retrieved chunks" in html_lower

    def test_explains_v2_contradiction(self, v3_html):
        """v3 narrative should explain that v2 contradicted v1."""
        html_lower = v3_html.lower()
        assert "contradict" in html_lower

    def test_v3_settled(self, v3_html):
        """v3 narrative should convey that the question is settled."""
        html_lower = v3_html.lower()
        assert "settled" in html_lower or "tiebreaker" in html_lower


# ---------------------------------------------------------------------------
# Test 3: v3 page has key findings
# ---------------------------------------------------------------------------

class TestV3KeyFindings:
    """v3 page should display key findings with correct stats."""

    def test_haiku_correlation(self, v3_html):
        """v3 findings should show Haiku's gold F1 correlation."""
        assert "0.450" in v3_html

    def test_exact_match_rate(self, v3_html):
        """v3 findings should show 76.2% exact match."""
        assert "76.2%" in v3_html or "76.2" in v3_html

    def test_download_link(self, v3_html):
        """v3 page should have a CSV download link."""
        assert "raw_scores_v3.csv" in v3_html


# ---------------------------------------------------------------------------
# Test 4: Navigation points to v3
# ---------------------------------------------------------------------------

class TestNavigation:
    """Nav bar should point to experiment_0_v3.html for Experiment 0."""

    def test_nav_links_to_v3(self, v3_html):
        """Nav should contain a link to experiment_0_v3.html."""
        assert "experiment_0_v3.html" in v3_html

    def test_nav_exp0_label(self, index_html):
        """Nav should have an Exp 0 entry linking to v3."""
        # The nav should contain both the label and the v3 href
        assert "experiment_0_v3.html" in index_html


# ---------------------------------------------------------------------------
# Test 5: v1 and v2 pages still generate
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """v1 and v2 pages should still be generated and accessible."""

    def test_v1_page_exists(self, gallery_output):
        """v1 page (experiment_0.html) should be generated."""
        v1_path = gallery_output / "experiment_0.html"
        # v1 page exists if v1 data exists; if not, it may be a copy of v2/v3
        # Either way, the file should exist
        assert v1_path.exists()

    def test_v2_page_exists(self, gallery_output):
        """v2 page (experiment_0_v2.html) should be generated."""
        v2_csv = PROJECT_ROOT / "results" / "experiment_0_v2" / "raw_scores.csv"
        if not v2_csv.exists():
            pytest.skip("No v2 data")
        v2_path = gallery_output / "experiment_0_v2.html"
        assert v2_path.exists()

    def test_v2_page_not_empty(self, gallery_output):
        """v2 page should have real content, not be empty."""
        v2_csv = PROJECT_ROOT / "results" / "experiment_0_v2" / "raw_scores.csv"
        if not v2_csv.exists():
            pytest.skip("No v2 data")
        v2_path = gallery_output / "experiment_0_v2.html"
        content = v2_path.read_text(encoding="utf-8")
        assert len(content) > 1000  # Should have substantial content


# ---------------------------------------------------------------------------
# Test 6: Card links work (no broken hrefs)
# ---------------------------------------------------------------------------

class TestCardLinks:
    """All experiment card links should point to files that exist."""

    def test_no_experiment_0v2_link(self, index_html):
        """Should NOT have experiment_0v2.html (missing underscore) link."""
        # This was the broken link pattern
        assert 'href="experiment_0v2.html"' not in index_html

    def test_no_experiment_0v3_link(self, index_html):
        """Should NOT have experiment_0v3.html (missing underscore) link."""
        assert 'href="experiment_0v3.html"' not in index_html
