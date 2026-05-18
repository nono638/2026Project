"""Smoke tests for src.monitoring.

These cover the failure-open contract that telemetry must never abort an
experiment: a missing nvidia-smi, a malformed line, or an unserializable
event payload must all be handled silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.monitoring import EventLog, snapshot
from src.monitoring.gpu_telemetry import _FIELDS


class TestGpuSnapshot:
    """gpu_telemetry.snapshot is the one nvidia-smi shell-out."""

    def test_returns_empty_when_nvidia_smi_missing(self):
        with patch("src.monitoring.gpu_telemetry.shutil.which", return_value=None):
            assert snapshot() == []

    def test_returns_empty_on_timeout(self):
        import subprocess
        with patch("src.monitoring.gpu_telemetry.shutil.which",
                   return_value="/fake/nvidia-smi"), \
             patch("src.monitoring.gpu_telemetry.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("nvidia-smi", 3)):
            assert snapshot() == []

    def test_returns_empty_on_nonzero_exit(self):
        fake = type("R", (), {"returncode": 1, "stdout": ""})()
        with patch("src.monitoring.gpu_telemetry.shutil.which",
                   return_value="/fake/nvidia-smi"), \
             patch("src.monitoring.gpu_telemetry.subprocess.run", return_value=fake):
            assert snapshot() == []

    def test_parses_csv_row(self):
        # nvidia-smi --query-gpu emits CSV in the order of _FIELDS.
        values = ["0", "NVIDIA GeForce RTX 5090 Laptop GPU",
                  "45", "12", "5", "24463", "300", "24163",
                  "12.34", "1800", "9000"]
        assert len(values) == len(_FIELDS)
        fake = type("R", (), {
            "returncode": 0,
            "stdout": ", ".join(values) + "\n",
        })()
        with patch("src.monitoring.gpu_telemetry.shutil.which",
                   return_value="/fake/nvidia-smi"), \
             patch("src.monitoring.gpu_telemetry.subprocess.run", return_value=fake):
            result = snapshot()
        assert len(result) == 1
        row = result[0]
        assert row["index"] == "0"
        assert "RTX 5090" in row["name"]
        assert row["temperature.gpu"] == 45
        assert row["power.draw"] == 12.34
        assert row["memory.total"] == 24463


class TestEventLog:
    """EventLog must be append-only, durable, and fail-open."""

    def test_writes_jsonl_line_per_event(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        log.write("run_start", experiment="test", n=3)
        log.write("config_start", strategy="naive", model="qwen3:0.8b")

        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "run_start"
        assert first["experiment"] == "test"
        assert first["n"] == 3
        assert "timestamp_iso" in first

    def test_unserializable_field_is_stringified_not_raised(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        # An arbitrary object isn't JSON-serializable on its own — the
        # writer falls back to its repr so the line is still parseable.
        log.write("weird", bad=object())
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])  # must parse
        assert record["event"] == "weird"
        assert isinstance(record["bad"], str)
        assert "object" in record["bad"]

    def test_unwritable_path_disables_log_silently(self, tmp_path: Path):
        # Pointing at a directory makes open(... 'a') fail.
        bad = tmp_path / "is_a_dir"
        bad.mkdir()
        log = EventLog(bad)
        log.write("anything", x=1)  # must not raise
        # Subsequent writes should also be no-ops.
        log.write("anything", x=2)
