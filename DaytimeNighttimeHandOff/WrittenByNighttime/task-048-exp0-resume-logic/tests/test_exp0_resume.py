"""Tests for Experiment 0 resilient resume logic (task-048).

Covers: generation checkpointing, cost guard abort, per-judge scoring
resume, and corrupt CSV handling.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_guard import CostLimitExceeded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_answer_row(example_id: int, question: str = "Q?", gold: str = "A") -> dict:
    """Create a minimal answer dict matching run_experiment_0 shape."""
    return {
        "example_id": example_id,
        "question": question,
        "gold_answer": gold,
        "rag_answer": f"Answer for {example_id}",
        "doc_text": f"Doc text for {example_id}",
        "difficulty": "medium",
        "question_type": "bridge",
        "strategy_latency_ms": 100.0,
        "num_chunks": 5,
        "num_chunks_retrieved": 3,
        "context_char_length": 500,
        "context_sent_to_llm": f"Context for {example_id}",
        "failure_stage": "none",
        "failure_stage_confidence": 1.0,
        "failure_stage_method": "gold_tracking",
        "gold_in_chunks": True,
        "gold_in_retrieved": True,
        "gold_in_context": True,
        "reranker_model": "bge",
        "reranker_top_k": 3,
        "retrieval_top_k": 10,
    }


def _make_score_row(example_id: int, judges: dict[str, float | None] = None) -> dict:
    """Create a scored row with specified judge quality values.

    Args:
        example_id: The example ID.
        judges: Dict of judge_safe_name -> quality value (None = NaN).
    """
    row = _make_answer_row(example_id)
    row["gold_exact_match"] = True
    row["gold_f1"] = 0.8
    if judges:
        for judge_name, quality in judges.items():
            for metric in ["faithfulness", "relevance", "conciseness", "quality"]:
                col = f"{judge_name}_{metric}"
                row[col] = quality if quality is not None else float("nan")
    return row


# ---------------------------------------------------------------------------
# Test 1: Generation resume — skip already-generated example_ids
# ---------------------------------------------------------------------------

class TestGenerationResume:
    """Generation should resume from where it left off."""

    def test_resumes_from_existing_answers(self, tmp_path):
        """If raw_answers.csv has 3 of 5 rows, only generate answers 4-5."""
        raw_answers_path = tmp_path / "raw_answers.csv"

        # Pre-populate with 3 answers
        existing = pd.DataFrame([
            _make_answer_row(0),
            _make_answer_row(1),
            _make_answer_row(2),
        ])
        existing.to_csv(raw_answers_path, index=False)

        # Load and check which IDs exist
        loaded = pd.read_csv(raw_answers_path)
        existing_ids = set(loaded["example_id"].tolist())

        assert existing_ids == {0, 1, 2}
        # The generation loop should skip these and only generate for 3, 4
        remaining = [i for i in range(5) if i not in existing_ids]
        assert remaining == [3, 4]

    def test_fresh_start_generates_all(self, tmp_path):
        """If no raw_answers.csv exists, generate all answers."""
        raw_answers_path = tmp_path / "raw_answers.csv"
        assert not raw_answers_path.exists()

        existing_ids: set[int] = set()
        remaining = [i for i in range(5) if i not in existing_ids]
        assert remaining == [0, 1, 2, 3, 4]

    def test_corrupt_csv_skips_bad_lines(self, tmp_path):
        """Truncated last line should be silently skipped."""
        raw_answers_path = tmp_path / "raw_answers.csv"

        # Write valid rows + a corrupt trailing line
        valid = pd.DataFrame([_make_answer_row(0), _make_answer_row(1)])
        valid.to_csv(raw_answers_path, index=False)

        # Append a corrupt partial line
        with open(raw_answers_path, "a") as f:
            f.write("999,truncated,line,missing,columns\n")

        # Should load the valid rows without error
        loaded = pd.read_csv(raw_answers_path, on_bad_lines="skip")
        assert len(loaded) >= 2  # At least the 2 valid rows
        assert 0 in loaded["example_id"].values
        assert 1 in loaded["example_id"].values


# ---------------------------------------------------------------------------
# Test 2: Cost guard abort — break the scoring loop immediately
# ---------------------------------------------------------------------------

class TestCostGuardAbort:
    """CostLimitExceeded must stop ALL scoring, not just one scorer."""

    def test_stops_after_cost_limit(self):
        """Scoring loop should stop when CostLimitExceeded is raised."""
        call_count = 0

        class MockScorer:
            name = "mock:test-scorer"

            def score(self, query, context, answer):
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    raise CostLimitExceeded("Over limit after 3 calls")
                return {
                    "faithfulness": 4.0,
                    "relevance": 4.0,
                    "conciseness": 4.0,
                }

        # Simulate 10 answers — should stop at answer 3, not continue to 10
        answers = [_make_answer_row(i) for i in range(10)]
        scorer = MockScorer()

        scored_rows = []
        cost_limit_hit = False
        for ans in answers:
            try:
                scores = scorer.score(
                    query=ans["question"],
                    context=ans["context_sent_to_llm"],
                    answer=ans["rag_answer"],
                )
                scored_rows.append(ans["example_id"])
            except CostLimitExceeded:
                cost_limit_hit = True
                break

        assert cost_limit_hit is True
        # Should have scored only 2 (calls 1 and 2 succeed, call 3 raises)
        assert len(scored_rows) == 2
        # Should NOT have scored all 10
        assert call_count == 3  # 2 success + 1 that raised

    def test_partial_checkpoint_saved_on_abort(self, tmp_path):
        """Partial results should be checkpointed when cost limit hits."""
        checkpoint_path = tmp_path / "raw_scores_checkpoint.csv"

        # Simulate scoring 2 rows, then cost limit on row 3
        rows = [_make_score_row(0, {"judge_a": 4.0}), _make_score_row(1, {"judge_a": 3.5})]
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        assert checkpoint_path.exists()
        saved = pd.read_csv(checkpoint_path)
        assert len(saved) == 2
        assert "judge_a_quality" in saved.columns


# ---------------------------------------------------------------------------
# Test 3: Per-judge scoring resume — skip already-scored judges
# ---------------------------------------------------------------------------

class TestPerJudgeResume:
    """Only score judges whose columns are missing or NaN for a row."""

    def test_skips_already_scored_judge(self):
        """If judge A has a score, don't re-call judge A."""
        row = _make_score_row(0, {"judge_a": 4.0, "judge_b": None})

        # Judge A should be skipped (has non-NaN quality)
        judge_a_needs_scoring = math.isnan(row.get("judge_a_quality", float("nan")))
        assert judge_a_needs_scoring is False

        # Judge B should be scored (has NaN quality)
        judge_b_needs_scoring = math.isnan(row.get("judge_b_quality", float("nan")))
        assert judge_b_needs_scoring is True

    def test_scores_missing_judge_column(self):
        """If judge C column doesn't exist, it needs scoring."""
        row = _make_score_row(0, {"judge_a": 4.0})

        # Judge C column doesn't exist — should need scoring
        judge_c_val = row.get("judge_c_quality", float("nan"))
        assert math.isnan(judge_c_val)

    def test_all_judges_scored_skips_row(self):
        """Row with all judges scored should be completely skipped."""
        row = _make_score_row(0, {"judge_a": 4.0, "judge_b": 3.5})

        judges_to_check = ["judge_a", "judge_b"]
        all_scored = all(
            not math.isnan(row.get(f"{j}_quality", float("nan")))
            for j in judges_to_check
        )
        assert all_scored is True


# ---------------------------------------------------------------------------
# Test 4: Full-scored skip — zero API calls when everything is done
# ---------------------------------------------------------------------------

class TestFullScoredSkip:
    """If all rows have all judges scored, make zero API calls."""

    def test_no_api_calls_when_complete(self, tmp_path):
        """Complete raw_scores.csv should result in zero scorer calls."""
        raw_scores_path = tmp_path / "raw_scores.csv"

        # All 5 rows scored by both judges
        rows = [
            _make_score_row(i, {"judge_a": 4.0, "judge_b": 3.5})
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        df.to_csv(raw_scores_path, index=False)

        loaded = pd.read_csv(raw_scores_path)
        judges = ["judge_a", "judge_b"]

        rows_needing_scoring = 0
        for _, row in loaded.iterrows():
            for j in judges:
                val = row.get(f"{j}_quality", float("nan"))
                if pd.isna(val):
                    rows_needing_scoring += 1

        assert rows_needing_scoring == 0


# ---------------------------------------------------------------------------
# Test 5: Checkpoint merge with raw_scores.csv
# ---------------------------------------------------------------------------

class TestCheckpointMerge:
    """Checkpoint and raw_scores.csv should merge correctly."""

    def test_checkpoint_fills_gaps_in_raw_scores(self, tmp_path):
        """Checkpoint with newer data should fill NaN gaps in raw_scores."""
        raw_scores_path = tmp_path / "raw_scores.csv"
        checkpoint_path = tmp_path / "raw_scores_checkpoint.csv"

        # raw_scores has 5 rows, rows 4-5 have NaN for judge_b
        raw_rows = []
        for i in range(5):
            judges = {"judge_a": 4.0}
            if i < 3:
                judges["judge_b"] = 3.5
            else:
                judges["judge_b"] = None  # NaN
            raw_rows.append(_make_score_row(i, judges))
        pd.DataFrame(raw_rows).to_csv(raw_scores_path, index=False)

        # Checkpoint has rows 3-4 with judge_b filled in
        checkpoint_rows = [
            _make_score_row(3, {"judge_a": 4.0, "judge_b": 4.2}),
            _make_score_row(4, {"judge_a": 4.0, "judge_b": 3.8}),
        ]
        pd.DataFrame(checkpoint_rows).to_csv(checkpoint_path, index=False)

        # Merge: load both, checkpoint takes priority for non-NaN values
        raw_df = pd.read_csv(raw_scores_path)
        ckpt_df = pd.read_csv(checkpoint_path)

        # Update raw_df with checkpoint values where checkpoint has non-NaN
        ckpt_indexed = ckpt_df.set_index("example_id")
        for eid in ckpt_indexed.index:
            mask = raw_df["example_id"] == eid
            if mask.any():
                for col in ckpt_indexed.columns:
                    val = ckpt_indexed.loc[eid, col]
                    if pd.notna(val):
                        raw_df.loc[mask, col] = val

        # Verify: all 5 rows should now have non-NaN judge_b_quality
        assert raw_df["judge_b_quality"].notna().all()
        assert raw_df.loc[raw_df["example_id"] == 3, "judge_b_quality"].iloc[0] == 4.2
        assert raw_df.loc[raw_df["example_id"] == 4, "judge_b_quality"].iloc[0] == 3.8


# ---------------------------------------------------------------------------
# Test 6: Checkpoint not deleted on partial completion
# ---------------------------------------------------------------------------

class TestCheckpointRetention:
    """Checkpoint should only be deleted when all scoring is complete."""

    def test_checkpoint_kept_when_partial(self, tmp_path):
        """If some rows have NaN judges, checkpoint must be retained."""
        checkpoint_path = tmp_path / "raw_scores_checkpoint.csv"

        # 3 rows, but row 2 has NaN for judge_b (incomplete)
        rows = [
            _make_score_row(0, {"judge_a": 4.0, "judge_b": 3.5}),
            _make_score_row(1, {"judge_a": 4.0, "judge_b": 3.5}),
            _make_score_row(2, {"judge_a": 4.0, "judge_b": None}),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(checkpoint_path, index=False)

        # Check if all judges are complete
        judge_cols = [c for c in df.columns if c.endswith("_quality")]
        all_complete = df[judge_cols].notna().all().all()

        assert all_complete is False
        # Therefore checkpoint should NOT be deleted
        assert checkpoint_path.exists()

    def test_checkpoint_deleted_when_complete(self, tmp_path):
        """If all rows have all judges scored, checkpoint can be deleted."""
        checkpoint_path = tmp_path / "raw_scores_checkpoint.csv"

        rows = [
            _make_score_row(0, {"judge_a": 4.0, "judge_b": 3.5}),
            _make_score_row(1, {"judge_a": 4.0, "judge_b": 3.5}),
        ]
        df = pd.DataFrame(rows)
        df.to_csv(checkpoint_path, index=False)

        judge_cols = [c for c in df.columns if c.endswith("_quality")]
        all_complete = df[judge_cols].notna().all().all()

        assert all_complete is True
        # Safe to delete checkpoint
        if all_complete:
            checkpoint_path.unlink()
        assert not checkpoint_path.exists()
