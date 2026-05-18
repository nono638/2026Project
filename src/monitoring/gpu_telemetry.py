"""One-shot GPU telemetry snapshots via nvidia-smi.

Shells out to nvidia-smi rather than using pynvml because nvidia-smi is
already on PATH on every machine we run on (Windows laptop and the 5090),
while pynvml adds a dependency and version-pinning hassle for one query.

The function fails open: a missing nvidia-smi, a non-NVIDIA system, or a
malformed output all return an empty list rather than raising — telemetry
must never abort an experiment.
"""

from __future__ import annotations

import shutil
import subprocess


# Fields we record per snapshot. Order matters — must match the CSV column
# order produced by nvidia-smi --format=csv,noheader.
# We deliberately omit power.limit because it returns "[N/A]" on RTX 5090
# Laptop GPUs (mobile GPUs don't expose a power limit query), which would
# poison int parsing for no useful signal.
_FIELDS = (
    "index",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
)


def snapshot(timeout_s: float = 3.0) -> list[dict]:
    """Return a list of GPU stat dicts, one per visible NVIDIA GPU.

    Returns an empty list if nvidia-smi is unavailable, times out, or
    returns no parseable lines. Numeric fields are returned as ints/floats
    when they parse cleanly; "[N/A]" and other non-numeric values pass
    through as strings so downstream JSON serialization still works.

    Args:
        timeout_s: Subprocess timeout. nvidia-smi normally returns in
            ~50ms; a 3s ceiling lets us survive a transient driver hiccup
            without hanging the experiment.

    Returns:
        List of per-GPU dicts. Empty list if probe failed.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    query = ",".join(_FIELDS)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    if result.returncode != 0:
        return []

    gpus: list[dict] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_FIELDS):
            continue
        row: dict = {}
        for field, value in zip(_FIELDS, parts):
            # Try float first (handles power.draw like "12.34"), then int,
            # else keep as string ("[N/A]", GPU name, etc.).
            if field in ("index", "name"):
                row[field] = value
                continue
            try:
                row[field] = float(value) if "." in value else int(value)
            except ValueError:
                row[field] = value
        gpus.append(row)
    return gpus
