"""Tests for row-level resume in experiment_utils.

Covers the changes made when raw_scores.csv flushing was tightened from
once-per-config to once-per-row, so a power loss mid-config no longer
discards the entire ~200-row config's work.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.experiment_utils import append_rows, load_checkpoint


def _read_back(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_load_checkpoint_returns_empty_set_when_file_missing(tmp_path: Path) -> None:
    assert load_checkpoint(tmp_path / "missing.csv") == set()


def test_load_checkpoint_two_keys_default(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    append_rows(path, [
        {"strategy": "naive", "model": "m1", "question": "q1", "x": 1},
        {"strategy": "naive", "model": "m1", "question": "q2", "x": 2},
        {"strategy": "naive", "model": "m2", "question": "q1", "x": 3},
    ])
    assert load_checkpoint(path) == {("naive", "m1"), ("naive", "m2")}


def test_load_checkpoint_three_keys_for_row_level_resume(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    append_rows(path, [
        {"strategy": "naive", "model": "m1", "question": "q1", "x": 1},
        {"strategy": "naive", "model": "m1", "question": "q2", "x": 2},
        {"strategy": "naive", "model": "m2", "question": "q1", "x": 3},
    ])
    triples = load_checkpoint(
        path, key_cols=("strategy", "model", "question"),
    )
    assert triples == {
        ("naive", "m1", "q1"),
        ("naive", "m1", "q2"),
        ("naive", "m2", "q1"),
    }


def test_load_checkpoint_handles_question_text_with_commas(tmp_path: Path) -> None:
    """Real HotpotQA questions contain commas; CSV quoting must round-trip."""
    path = tmp_path / "raw.csv"
    tricky_q = 'Are both Tim McIlrath and Spike Slawson, the punk rock musicians, American?'
    append_rows(path, [
        {"strategy": "naive", "model": "m1", "question": tricky_q, "x": 1},
    ])
    triples = load_checkpoint(
        path, key_cols=("strategy", "model", "question"),
    )
    assert triples == {("naive", "m1", tricky_q)}


def test_load_checkpoint_returns_empty_when_key_column_missing(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    append_rows(path, [{"a": 1, "b": 2}])
    assert load_checkpoint(path, key_cols=("strategy", "model")) == set()


def test_append_rows_per_row_call_preserves_header_and_appends(tmp_path: Path) -> None:
    """Per-row flush mode: each call writes one row, header only on first."""
    path = tmp_path / "raw.csv"
    append_rows(path, [{"a": 1, "b": 2}])
    append_rows(path, [{"a": 3, "b": 4}])
    append_rows(path, [{"a": 5, "b": 6}])

    rows = _read_back(path)
    assert rows == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
        {"a": "5", "b": "6"},
    ]


def test_append_rows_empty_list_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    append_rows(path, [])
    assert not path.exists()


def test_resume_skip_set_intersects_correctly(tmp_path: Path) -> None:
    """Simulate the run_experiment_1 resume path: load triples, then check
    membership for each (strategy, model, question) about to run."""
    path = tmp_path / "raw.csv"
    append_rows(path, [
        {"strategy": "naive", "model": "m1", "question": "q1", "x": 1},
        {"strategy": "naive", "model": "m1", "question": "q2", "x": 2},
    ])
    completed = load_checkpoint(
        path, key_cols=("strategy", "model", "question"),
    )

    planned = [
        ("naive", "m1", "q1"),
        ("naive", "m1", "q2"),
        ("naive", "m1", "q3"),
        ("naive", "m2", "q1"),
    ]
    to_run = [t for t in planned if t not in completed]
    assert to_run == [("naive", "m1", "q3"), ("naive", "m2", "q1")]
