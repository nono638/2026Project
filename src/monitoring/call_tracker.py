"""Process-wide tracker for the most-recent Ollama call.

Why this exists: when a 5090 BSOD kills the Python process mid-call, the
last `events.jsonl` entry is whatever fired most recently — usually a
GPU snapshot from up to N rows ago. We want finer post-mortem: "the
process was 4.2s into an embed call against embeddinggemma:300m when
the kernel pulled the rug." The OllamaLLM / OllamaEmbedder wrappers
call ``record_start`` / ``record_end`` around every HTTP call; the
runner snapshots this tracker into every row-level heartbeat.

Module-level state is deliberate. There's exactly one Ollama server
this process talks to, calls are serial (no asyncio/threading in the
strategies), and adding a class would force every call site to thread
an instance through. A heartbeat that lies under contention is a bug
worth knowing about, not worth designing around.

Fails open in all directions: a malformed call_end without a matching
call_start, a snapshot during no in-flight call — all return sensible
defaults rather than raising. Telemetry must not crash the run.
"""

from __future__ import annotations

import time
from typing import Any

_state: dict[str, Any] = {
    "last_call_type": None,        # "embed" | "chat" | None
    "last_call_model": None,       # model tag string
    "last_call_started_at": None,  # monotonic seconds
    "last_call_ended_at": None,    # monotonic seconds, None while in-flight
    "last_call_duration_s": None,  # filled on call_end
    "in_flight": False,
    "calls_total": 0,
}


def record_start(call_type: str, model: str) -> None:
    """Mark a new call as in-flight.

    Args:
        call_type: ``"embed"`` or ``"chat"``.
        model: Ollama model tag (e.g., ``"embeddinggemma:300m"``).
    """
    _state["last_call_type"] = call_type
    _state["last_call_model"] = model
    _state["last_call_started_at"] = time.monotonic()
    _state["last_call_ended_at"] = None
    _state["last_call_duration_s"] = None
    _state["in_flight"] = True


def record_end() -> None:
    """Mark the in-flight call as finished and record its duration."""
    now = time.monotonic()
    start = _state.get("last_call_started_at")
    if start is not None:
        _state["last_call_duration_s"] = round(now - start, 3)
    _state["last_call_ended_at"] = now
    _state["in_flight"] = False
    _state["calls_total"] += 1


def snapshot() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the tracker state.

    ``in_flight_for_s`` is computed live so the runner can tell how long
    a hung call has been running when the heartbeat fires.
    """
    out = {
        "last_call_type": _state["last_call_type"],
        "last_call_model": _state["last_call_model"],
        "last_call_duration_s": _state["last_call_duration_s"],
        "in_flight": _state["in_flight"],
        "calls_total": _state["calls_total"],
    }
    if _state["in_flight"] and _state["last_call_started_at"] is not None:
        out["in_flight_for_s"] = round(
            time.monotonic() - _state["last_call_started_at"], 3
        )
    return out
