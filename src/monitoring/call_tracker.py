"""Process-wide tracker for Ollama calls — heartbeat + per-row trace.

Two complementary roles in one module:

1. **Live heartbeat** (``record_start`` / ``record_end`` / ``snapshot``).
   When a 5090 BSOD kills the process mid-call, the last events.jsonl
   entry is whatever fired most recently. The runner snapshots this
   tracker into every row-level heartbeat so post-mortems can answer
   "the process was 4.2 s into an embed call against
   embeddinggemma:300m when the kernel pulled the rug."

2. **Per-row trace** (``begin_row`` / ``record_complete`` / ``end_row``).
   While a row is "open", every successful chat / embed call also pushes
   a full record (prompt, response, intent, latency) into a list. The
   runner pulls that list at row end and writes it to a sidecar
   ``traces.jsonl`` so future analysis can answer "for this exact row,
   what prompts were sent and what came back?" — a question that
   Experiments 0–2 originally had no way of answering.

Module-level state is deliberate. There's exactly one Ollama server
this process talks to, calls are serial (no asyncio / threading in the
strategies), and adding a class would force every call site to thread
an instance through. A heartbeat that lies under contention is a bug
worth knowing about, not worth designing around.

Fails open in all directions: a malformed ``record_end`` without a
matching ``record_start``, a snapshot during no in-flight call,
``record_complete`` outside an open row — all return sensible defaults
rather than raising. Telemetry must not crash the run.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
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

# Per-row capture window. ``None`` means no row is currently open and
# ``record_complete`` is a no-op; an empty list means a row is open and
# trace records will be appended to it.
_row_trace: list[dict] | None = None

# Repo SHA cached at first request — stamped on every row so future
# analysis can pin which version of src/strategies / src/llms / etc.
# was actually running when this row was generated.
_code_sha_cache: dict[str, str | None] = {"full": None, "short": None}


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


# ---------------------------------------------------------------------------
# Per-row trace
# ---------------------------------------------------------------------------

def begin_row() -> None:
    """Open a new per-row capture window. Any prior unclosed window is dropped.

    The runner calls this immediately before invoking ``strategy.run`` for
    one (strategy, model, question) row, then calls ``end_row`` after the
    strategy returns. Between those two calls, every successful
    ``record_complete`` appends to the row trace.
    """
    global _row_trace
    _row_trace = []


def end_row() -> list[dict]:
    """Close the current row's capture window and return the trace.

    Returns:
        Ordered list of call records logged inside this row. Empty if no
        row was open or no calls fired.
    """
    global _row_trace
    out = _row_trace if _row_trace is not None else []
    _row_trace = None
    return out


def record_complete(
    call_type: str,
    model: str,
    prompt: str,
    response: str,
    latency_s: float,
    intent: str | None = None,
) -> None:
    """Append a finished call to the active row trace (if any is open).

    This is the "what happened" record — distinct from ``record_start`` /
    ``record_end`` which only tell the post-mortem heartbeat what was
    in-flight when the process died. Both are fired around the same call.

    Args:
        call_type: ``"chat"`` for generation, ``"embed"`` for retrieval.
        model: Ollama model tag.
        prompt: The full prompt text sent to the model. For embed calls,
            the input text (truncated by the caller if it was large).
        response: The full model response. For embed calls, a short
            descriptor like ``"<vector len=768>"`` is fine — vectors
            don't carry per-row diagnostic value.
        latency_s: Wall-clock seconds the call took, rounded by caller.
        intent: Semantic label provided by the caller for what this call
            is *for* — e.g., ``"rate_relevance"``, ``"reformulate_query"``,
            ``"generate_answer"``, ``"classify_complexity"``. ``None``
            becomes ``"unknown"``.
    """
    if _row_trace is None:
        return
    _row_trace.append({
        "call_type": call_type,
        "intent": intent or "unknown",
        "model": model,
        "prompt": prompt,
        "response": response,
        "latency_s": round(float(latency_s), 4),
        "ts": time.time(),
    })


# ---------------------------------------------------------------------------
# Code-version stamp
# ---------------------------------------------------------------------------

def code_sha(repo_root: Path | None = None) -> dict[str, str | None]:
    """Return the repo's HEAD commit SHA, cached after the first lookup.

    Going-forward provenance — stamped on every row so a backfill or
    re-analysis can say with certainty which source-tree state produced
    a given row. ``None`` values mean we couldn't determine the SHA
    (git not on PATH, not a git repo, detached state with no useful
    output). Telemetry must not raise.

    Args:
        repo_root: Directory containing the .git dir. Defaults to the
            project root inferred from this file's location.

    Returns:
        ``{"full": "<40-char SHA>", "short": "<7-char SHA>"}``. Either
        field may be ``None`` if the lookup failed.
    """
    if _code_sha_cache["full"] is not None or _code_sha_cache["short"] is not None:
        return dict(_code_sha_cache)
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        full = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if full.returncode == 0:
            _code_sha_cache["full"] = full.stdout.strip()
            _code_sha_cache["short"] = full.stdout.strip()[:7]
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        # Git not installed, or the project isn't a repo, or the call
        # timed out. Either way, leave the cache as Nones.
        pass
    return dict(_code_sha_cache)
