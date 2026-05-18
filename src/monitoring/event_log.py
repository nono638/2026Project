"""Structured JSONL event log for long-running experiments.

The plaintext run log (experiment_1_run.log) is dominated by per-request
HTTP lines — useful for live tailing, useless for post-mortem. This log
captures only the structured events that matter when reconstructing a
crashed run: config boundaries, GPU snapshots, errors, retries, shutdown.

Each line is one JSON object with a ``timestamp_iso`` and an ``event``
discriminator. New event types can be added without changing the writer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventLog:
    """Append-only JSONL writer with one process-wide instance per file.

    Thread-safe via a per-instance lock. Each call to ``write`` is
    flushed and fsynced so a system crash loses at most the in-flight
    event — same durability guarantee as the row-level CSV checkpoint.

    Writes never raise: if the file can't be opened or fsync fails the
    error is logged once and subsequent writes become no-ops, so a
    monitoring failure can't take down an experiment.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._disabled = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("EventLog disabled — cannot create dir %s: %s",
                           self._path.parent, exc)
            self._disabled = True

    def write(self, event: str, **fields: Any) -> None:
        """Append one event record.

        Args:
            event: Short discriminator like ``"config_start"`` or
                ``"gpu_snapshot"``.
            **fields: Arbitrary JSON-serializable payload merged into the
                record. Reserved keys ``timestamp_iso`` and ``event`` are
                set by the writer and will overwrite any caller-supplied
                values with the same names.
        """
        if self._disabled:
            return
        record = dict(fields)
        record["event"] = event
        record["timestamp_iso"] = datetime.now().isoformat(timespec="seconds")
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("EventLog skipped unserializable record (%s): %r",
                           exc, event)
            return
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:
                logger.warning("EventLog disabled — write failed: %s", exc)
                self._disabled = True

    @property
    def path(self) -> Path:
        return self._path
