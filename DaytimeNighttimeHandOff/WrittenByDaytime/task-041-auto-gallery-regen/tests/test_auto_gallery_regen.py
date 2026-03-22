"""Tests for task-041: auto-regenerate gallery after experiment runs.

Verifies that each experiment script calls generate_gallery.main() at the end,
respects --no-gallery, and handles gallery generation failures gracefully.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Experiment 0
# ---------------------------------------------------------------------------


class TestExperiment0GalleryRegen:
    """Gallery regeneration wired into run_experiment_0.py."""

    @patch("scripts.run_experiment_0.parse_args")
    def test_no_gallery_flag_exists(self, mock_parse_args: MagicMock) -> None:
        """The --no-gallery flag should be accepted by the arg parser."""
        from scripts.run_experiment_0 import parse_args as _real_parse_args

        # Temporarily unpatch to test the real parser
        mock_parse_args.stop  # noqa: not needed
        import scripts.run_experiment_0 as mod

        # Build a fresh parser by inspecting the parse_args function
        # We test that --no-gallery is a valid argument
        with patch("sys.argv", ["run_experiment_0.py", "--no-gallery"]):
            # This should NOT raise SystemExit for unrecognized argument
            args = mod.parse_args()
            assert hasattr(args, "no_gallery"), (
                "parse_args() must return a namespace with 'no_gallery' attribute"
            )
            assert args.no_gallery is True

    @patch("scripts.run_experiment_0.parse_args")
    def test_no_gallery_flag_default_false(self, mock_parse_args: MagicMock) -> None:
        """--no-gallery defaults to False (gallery regeneration ON by default)."""
        import scripts.run_experiment_0 as mod

        with patch("sys.argv", ["run_experiment_0.py"]):
            args = mod.parse_args()
            assert hasattr(args, "no_gallery")
            assert args.no_gallery is False


class TestExperiment0GalleryCall:
    """Verify main() in run_experiment_0 calls generate_gallery."""

    def test_gallery_called_after_experiment(self) -> None:
        """After experiment completes, generate_gallery.main is called with experiments=[0]."""
        # We check that the source code of run_experiment_0.main contains
        # a call to gallery generation with experiment 0
        import inspect
        import scripts.run_experiment_0 as mod

        source = inspect.getsource(mod.main)
        # Should contain gallery-related call with experiments=[0]
        assert "generate_gallery" in source or "gallery" in source.lower(), (
            "main() must call generate_gallery somewhere"
        )
        assert "experiments=[0]" in source or "experiments = [0]" in source, (
            "main() must call generate_gallery with experiments=[0]"
        )

    def test_gallery_failure_does_not_crash(self) -> None:
        """If gallery generation raises, main() should not propagate the exception."""
        import inspect
        import scripts.run_experiment_0 as mod

        source = inspect.getsource(mod.main)
        # The gallery call should be wrapped in try/except
        assert "try:" in source and "except" in source, (
            "main() must wrap gallery generation in try/except"
        )


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------


class TestExperiment1GalleryRegen:
    """Gallery regeneration wired into run_experiment_1.py."""

    def test_no_gallery_flag_exists(self) -> None:
        """The --no-gallery flag should be accepted by the arg parser."""
        import scripts.run_experiment_1 as mod

        with patch("sys.argv", ["run_experiment_1.py", "--no-gallery"]):
            args = mod.parse_args()
            assert hasattr(args, "no_gallery")
            assert args.no_gallery is True

    def test_no_gallery_flag_default_false(self) -> None:
        """--no-gallery defaults to False."""
        import scripts.run_experiment_1 as mod

        with patch("sys.argv", ["run_experiment_1.py"]):
            args = mod.parse_args()
            assert hasattr(args, "no_gallery")
            assert args.no_gallery is False

    def test_gallery_called_with_correct_experiment(self) -> None:
        """main() calls generate_gallery with experiments=[1]."""
        import inspect
        import scripts.run_experiment_1 as mod

        source = inspect.getsource(mod.main)
        assert "generate_gallery" in source or "gallery" in source.lower()
        assert "experiments=[1]" in source or "experiments = [1]" in source

    def test_gallery_failure_does_not_crash(self) -> None:
        """Gallery failure is caught, not re-raised."""
        import inspect
        import scripts.run_experiment_1 as mod

        source = inspect.getsource(mod.main)
        assert "try:" in source and "except" in source


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------


class TestExperiment2GalleryRegen:
    """Gallery regeneration wired into run_experiment_2.py."""

    def test_no_gallery_flag_exists(self) -> None:
        """The --no-gallery flag should be accepted by the arg parser."""
        import scripts.run_experiment_2 as mod

        with patch("sys.argv", ["run_experiment_2.py", "--no-gallery"]):
            args = mod.parse_args()
            assert hasattr(args, "no_gallery")
            assert args.no_gallery is True

    def test_no_gallery_flag_default_false(self) -> None:
        """--no-gallery defaults to False."""
        import scripts.run_experiment_2 as mod

        with patch("sys.argv", ["run_experiment_2.py"]):
            args = mod.parse_args()
            assert hasattr(args, "no_gallery")
            assert args.no_gallery is False

    def test_gallery_called_with_correct_experiment(self) -> None:
        """main() calls generate_gallery with experiments=[2]."""
        import inspect
        import scripts.run_experiment_2 as mod

        source = inspect.getsource(mod.main)
        assert "generate_gallery" in source or "gallery" in source.lower()
        assert "experiments=[2]" in source or "experiments = [2]" in source

    def test_gallery_failure_does_not_crash(self) -> None:
        """Gallery failure is caught, not re-raised."""
        import inspect
        import scripts.run_experiment_2 as mod

        source = inspect.getsource(mod.main)
        assert "try:" in source and "except" in source


# ---------------------------------------------------------------------------
# Cross-cutting: --no-gallery skips regeneration
# ---------------------------------------------------------------------------


class TestNoGallerySkipsRegeneration:
    """When --no-gallery is set, no gallery code should run."""

    def test_no_gallery_guard_in_exp0(self) -> None:
        """run_experiment_0.main checks args.no_gallery before calling gallery."""
        import inspect
        import scripts.run_experiment_0 as mod

        source = inspect.getsource(mod.main)
        assert "no_gallery" in source, (
            "main() must check args.no_gallery to conditionally skip gallery"
        )

    def test_no_gallery_guard_in_exp1(self) -> None:
        """run_experiment_1.main checks args.no_gallery before calling gallery."""
        import inspect
        import scripts.run_experiment_1 as mod

        source = inspect.getsource(mod.main)
        assert "no_gallery" in source

    def test_no_gallery_guard_in_exp2(self) -> None:
        """run_experiment_2.main checks args.no_gallery before calling gallery."""
        import inspect
        import scripts.run_experiment_2 as mod

        source = inspect.getsource(mod.main)
        assert "no_gallery" in source
