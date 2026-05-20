"""Watchdog wrapper for run_experiment_1.py — auto-resume on crash/BSOD.

The 5090 has BSOD'd 6 times in the last 11 days (nvlddmkm,
DPC_WATCHDOG_VIOLATION / VIDEO_TDR_ERROR / SYSTEM_SERVICE_EXCEPTION /
IRQL_NOT_LESS_OR_EQUAL). Code-level pacing reduces the rate but cannot
eliminate it — it's a driver/firmware-level instability cluster.

This wrapper closes the human-in-the-loop gap: whenever the inner
process exits with a non-zero code, or the OS reboots (when launched
from a scheduled task — see scripts/install_auto_resume_task.ps1), the
wrapper re-launches the experiment with --resume so it picks up from the
last per-row checkpoint.

Safety rails:
  - --max-restarts caps the loop so a deterministic bug (e.g., a real
    Python crash on the same row) cannot spin forever.
  - A growing backoff after each restart gives the GPU driver time to
    settle and prevents tight restart storms.
  - The wrapper reads results/experiment_1/raw_scores.csv after each
    run and exits cleanly once the experiment has completed all 30
    configs at the full 200 rows each (= 6000 rows). This avoids
    infinitely re-resuming a finished run.

Usage:
    python scripts/run_experiment_1_watchdog.py
    python scripts/run_experiment_1_watchdog.py --max-restarts 20
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# raw_scores.csv carries a context_sent_to_llm column that can run into
# tens of KB per row; the default csv field limit (131,072 on 64-bit) trips
# on the long-context configs. Raise to sys.maxsize so _count_rows can read
# any row this experiment can emit.
csv.field_size_limit(2**31 - 1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "results" / "experiment_1_run.log"
DEFAULT_WATCHDOG_LOG = PROJECT_ROOT / "results" / "experiment_1_watchdog.log"
RAW_SCORES = PROJECT_ROOT / "results" / "experiment_1" / "raw_scores.csv"

# Full matrix size: 5 strategies x 6 models x 200 questions = 6000 rows.
# Matches ALL_STRATEGIES / ALL_MODELS / default --n in run_experiment_1.py.
EXPECTED_ROWS = 5 * 6 * 200


def parse_args() -> argparse.Namespace:
    """Parse watchdog CLI arguments."""
    p = argparse.ArgumentParser(
        description="Watchdog wrapper for run_experiment_1.py — auto-resume on crash.",
    )
    p.add_argument("--max-restarts", type=int, default=30,
                   help="Cap the restart loop (default: 30).")
    p.add_argument("--initial-backoff-s", type=float, default=15.0,
                   help="Sleep before first restart attempt (default: 15 s).")
    p.add_argument("--max-backoff-s", type=float, default=120.0,
                   help="Cap exponential backoff (default: 120 s).")
    p.add_argument("--max-cost", type=str, default="30",
                   help="Pass-through --max-cost for the experiment.")
    p.add_argument("--no-gallery", action="store_true", default=True,
                   help="Pass --no-gallery to the experiment (default: on).")
    p.add_argument("--log-path", type=str, default=str(DEFAULT_LOG),
                   help="Append experiment stdout/stderr to this file.")
    p.add_argument("--watchdog-log", type=str, default=str(DEFAULT_WATCHDOG_LOG),
                   help="Append watchdog decisions to this file.")
    return p.parse_args()


def _log(watchdog_log: Path, msg: str) -> None:
    """Append a timestamped line to the watchdog log and stderr.

    Args:
        watchdog_log: Path to the watchdog audit log.
        msg: Single-line message.
    """
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"{stamp} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        watchdog_log.parent.mkdir(parents=True, exist_ok=True)
        with watchdog_log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        # Watchdog logging must never break the loop.
        print(f"  (watchdog log write failed: {exc})", file=sys.stderr)


def _count_rows(path: Path) -> int:
    """Return number of data rows in a CSV (0 if missing)."""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            return sum(1 for _ in reader)
    except OSError:
        return 0


def main() -> int:
    """Run the experiment in a restart loop and return its final exit code."""
    args = parse_args()
    watchdog_log = Path(args.watchdog_log)
    inner_log = Path(args.log_path)

    python_exe = sys.executable  # use this process's interpreter (the venv)
    inner_script = PROJECT_ROOT / "scripts" / "run_experiment_1.py"

    base_cmd = [
        python_exe, "-u", str(inner_script),
        "--resume", "--max-cost", str(args.max_cost),
    ]
    if args.no_gallery:
        base_cmd.append("--no-gallery")

    _log(watchdog_log,
         f"watchdog start (pid={os.getpid()}) cmd={' '.join(base_cmd)} "
         f"max_restarts={args.max_restarts}")

    backoff = args.initial_backoff_s
    last_exit = 0
    for attempt in range(args.max_restarts + 1):  # attempt 0 = first run
        rows_before = _count_rows(RAW_SCORES)
        if rows_before >= EXPECTED_ROWS:
            _log(watchdog_log,
                 f"experiment already complete ({rows_before} rows >= "
                 f"{EXPECTED_ROWS}); exiting clean.")
            return 0

        _log(watchdog_log,
             f"attempt {attempt + 1}/{args.max_restarts + 1} "
             f"(rows_before={rows_before}/{EXPECTED_ROWS})")

        # Open the inner log in append mode so successive runs concatenate.
        inner_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with inner_log.open("a", encoding="utf-8", errors="replace") as log_fh:
                log_fh.write(f"\n========== watchdog attempt {attempt + 1} "
                             f"at {datetime.now().isoformat(timespec='seconds')} ==========\n")
                log_fh.flush()
                proc = subprocess.run(
                    base_cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            last_exit = proc.returncode
        except KeyboardInterrupt:
            _log(watchdog_log, "KeyboardInterrupt — stopping watchdog.")
            return 130
        except OSError as exc:
            _log(watchdog_log, f"failed to spawn inner process: {exc}")
            return 1

        rows_after = _count_rows(RAW_SCORES)
        progress = rows_after - rows_before
        _log(watchdog_log,
             f"attempt {attempt + 1} exited code={last_exit} "
             f"rows_after={rows_after} progress={progress}")

        if last_exit == 0:
            # Clean exit AND complete? Done. Clean exit but incomplete
            # (e.g. cost-limit hit) — also stop, the run made its own call.
            if rows_after >= EXPECTED_ROWS:
                _log(watchdog_log, "clean exit, full coverage — done.")
            else:
                _log(watchdog_log,
                     f"clean exit at {rows_after}/{EXPECTED_ROWS} rows "
                     "(cost limit or user stop) — not restarting.")
            return 0

        # Non-zero exit. If we made no progress AND we've tried before,
        # we're likely looping on the same deterministic failure. Bail out
        # rather than burn through the restart budget.
        if attempt > 0 and progress == 0:
            _log(watchdog_log,
                 "two consecutive failures with zero new rows — "
                 "likely deterministic bug, stopping.")
            return last_exit

        if attempt >= args.max_restarts:
            _log(watchdog_log,
                 f"max_restarts ({args.max_restarts}) reached, giving up.")
            return last_exit

        _log(watchdog_log, f"sleeping {backoff:.1f}s before restart...")
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            _log(watchdog_log, "KeyboardInterrupt during backoff — stopping.")
            return 130
        backoff = min(backoff * 1.5, args.max_backoff_s)

    return last_exit


if __name__ == "__main__":
    sys.exit(main())
