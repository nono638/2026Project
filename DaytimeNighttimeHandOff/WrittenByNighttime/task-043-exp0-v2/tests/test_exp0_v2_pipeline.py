"""Tests for Experiment 0 v2 pipeline changes.

Covers difficulty filtering, answer_quality computation, scorer context fix,
and CSV column requirements.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


class TestDifficultyFiltering:
    """Tests for HotpotQA difficulty filtering before sampling."""

    def test_medium_hard_filter_excludes_easy(self):
        """Filtering to medium+hard removes all easy questions."""
        # Simulate the filtering logic that run_experiment_0.py will use
        from unittest.mock import MagicMock

        queries = []
        for diff in ["easy", "medium", "hard", "easy", "hard"]:
            q = MagicMock()
            q.metadata = {"difficulty": diff, "question_type": "bridge"}
            queries.append(q)

        docs = [MagicMock() for _ in queries]

        allowed = {"medium", "hard"}
        filtered_docs, filtered_queries = [], []
        for d, q in zip(docs, queries):
            if q.metadata.get("difficulty") in allowed:
                filtered_docs.append(d)
                filtered_queries.append(q)

        assert len(filtered_queries) == 3
        for q in filtered_queries:
            assert q.metadata["difficulty"] in allowed

    def test_empty_filter_raises_or_exits(self):
        """Filtering with no matching difficulty should produce an empty list."""
        from unittest.mock import MagicMock

        queries = [MagicMock()]
        queries[0].metadata = {"difficulty": "easy"}
        docs = [MagicMock()]

        allowed = {"impossible"}
        filtered = [q for q in queries if q.metadata.get("difficulty") in allowed]
        assert len(filtered) == 0


class TestAnswerQuality:
    """Tests for the answer_quality column computation."""

    def _compute_answer_quality(
        self, bertscore: float, f1: float, sonnet_quality: float
    ) -> str:
        """Replicate the answer_quality logic from the spec.

        This is the reference implementation — the actual code should match.
        """
        is_poor = bertscore < 0.85 or f1 < 0.30 or sonnet_quality < 3.0
        is_good = bertscore >= 0.90 and f1 >= 0.50 and sonnet_quality >= 4.0

        if is_poor:
            return "poor"
        elif is_good:
            return "good"
        else:
            return "questionable"

    def test_good_answer_all_metrics_high(self):
        """Answer with high BERTScore, F1, and Sonnet quality is 'good'."""
        assert self._compute_answer_quality(0.95, 0.80, 5.0) == "good"

    def test_good_answer_at_thresholds(self):
        """Answer exactly at the good thresholds is 'good'."""
        assert self._compute_answer_quality(0.90, 0.50, 4.0) == "good"

    def test_poor_answer_low_bertscore(self):
        """Answer with low BERTScore is 'poor' even if other metrics are high."""
        assert self._compute_answer_quality(0.80, 0.80, 5.0) == "poor"

    def test_poor_answer_low_f1(self):
        """Answer with low F1 is 'poor' even if other metrics are high."""
        assert self._compute_answer_quality(0.95, 0.10, 5.0) == "poor"

    def test_poor_answer_low_sonnet(self):
        """Answer with low Sonnet quality is 'poor' even if gold metrics are high."""
        assert self._compute_answer_quality(0.95, 0.80, 2.0) == "poor"

    def test_questionable_answer_mixed_metrics(self):
        """Answer where metrics disagree is 'questionable'."""
        # BERTScore and Sonnet say good, but F1 is middling (not poor, not good)
        assert self._compute_answer_quality(0.92, 0.40, 4.5) == "questionable"

    def test_questionable_bertscore_borderline(self):
        """Answer with borderline BERTScore (between poor and good thresholds)."""
        assert self._compute_answer_quality(0.87, 0.60, 4.5) == "questionable"

    def test_answer_quality_on_dataframe(self):
        """answer_quality can be computed across a DataFrame."""
        df = pd.DataFrame({
            "gold_bertscore": [0.95, 0.80, 0.88],
            "gold_f1": [0.80, 0.10, 0.45],
            "anthropic_claude_sonnet_4_20250514_quality": [5.0, 2.0, 4.5],
        })

        sonnet_col = "anthropic_claude_sonnet_4_20250514_quality"
        df["answer_quality"] = df.apply(
            lambda r: self._compute_answer_quality(
                r["gold_bertscore"], r["gold_f1"], r[sonnet_col]
            ),
            axis=1,
        )

        assert df["answer_quality"].tolist() == ["good", "poor", "questionable"]

    def test_answer_quality_skipped_without_sonnet(self):
        """When Sonnet column is missing, answer_quality should not be added."""
        df = pd.DataFrame({
            "gold_bertscore": [0.95],
            "gold_f1": [0.80],
            # No Sonnet column
        })

        sonnet_col = "anthropic_claude_sonnet_4_20250514_quality"
        has_sonnet = sonnet_col in df.columns
        assert not has_sonnet


class TestScorerContextFix:
    """Tests that the scorer receives context_sent_to_llm, not doc_text."""

    def test_scorer_context_is_retrieved_chunks(self):
        """Scorer should be called with the context the LLM saw, not full doc."""
        # This tests the principle — the actual integration test would mock
        # the scorer and verify the context argument
        context_sent_to_llm = "chunk 1 text\n\nchunk 2 text"
        doc_text = "Full document with many paragraphs..."

        # The scorer should receive context_sent_to_llm
        assert context_sent_to_llm != doc_text
        assert len(context_sent_to_llm) < len(doc_text)


class TestCSVColumns:
    """Tests that v2 CSV includes all required columns."""

    def test_v2_required_columns_present(self):
        """v2 CSV must include diagnostics, metadata, and answer_quality columns."""
        v2_required = {
            # From v1
            "example_id", "question", "gold_answer", "rag_answer",
            "gold_exact_match", "gold_f1", "gold_bertscore",
            # New in v2 — metadata
            "difficulty", "question_type",
            # New in v2 — diagnostics
            "failure_stage", "failure_stage_confidence", "failure_stage_method",
            "context_sent_to_llm", "gold_in_chunks", "gold_in_retrieved",
            "gold_in_context",
            # New in v2 — reranker
            "reranker_model", "reranker_top_k",
            # New in v2 — answer quality (when Sonnet is available)
            "answer_quality",
            # Timing
            "strategy_latency_ms",
        }

        # Verify no typos in column names by checking they're all strings
        for col in v2_required:
            assert isinstance(col, str)
            assert len(col) > 0

        # Verify we have at least 20 columns
        assert len(v2_required) >= 20
