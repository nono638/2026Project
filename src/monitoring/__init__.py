"""Run-time telemetry helpers for long-running experiments.

The system has crashed six times in 10 days during sustained GPU load
(BSODs 0x133, 0x4E, 0x3B, 0x1A, 0xA — a mix consistent with a
documented RTX 5090 driver-stability issue, not a Python bug). RagBench
can't fix the driver, but it can record enough state that future crashes
can be correlated to specific workload phases.

This module provides:
- ``gpu_telemetry.snapshot()`` — one-shot nvidia-smi snapshot as a dict.
- ``event_log.EventLog`` — append-only JSONL writer for structured run events.

Both fail open: if nvidia-smi is missing or the log file can't be opened,
the experiment continues — telemetry must never abort a run.
"""

from src.monitoring.event_log import EventLog
from src.monitoring.gpu_telemetry import snapshot

__all__ = ["EventLog", "snapshot"]
