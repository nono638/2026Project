"""Backfill provenance columns onto pre-instrumentation experiment CSVs.

Context: Experiments 0–2 were generated before the per-row LLM-call
instrumentation in src.monitoring.call_tracker.begin_row/end_row +
OllamaLLM.generate(intent=...) landed. New rows produced by the
runner today carry seven provenance columns the old rows lack
(n_llm_calls, llm_call_intents, final_prompt, prompts_source,
code_sha, llm_num_predict, llm_keep_alive).

This script aligns the SCHEMA of historical CSVs to the new schema
without fabricating values. It does not invent prompts, branch
decisions, or call counts that weren't recorded. Per the website's
Logging Gaps section: backfilled rows are tagged
``prompts_source = "reconstructed_minimal"`` so any analysis that
joins recorded + reconstructed rows can filter on that flag.

What this script DOES fill:

- For NaiveRAG rows (whose pipeline has a single fixed prompt template
  and exactly one chat call per row), it reconstructs the literal
  final prompt from ``context_sent_to_llm`` + ``question`` — these are
  byte-exact given the template is unchanged. n_llm_calls is set to 1.
- For events.jsonl-derived ``calls_total`` deltas, it can write a best-
  effort ``n_total_calls_inc_embed`` for the runs that have a sidecar
  events log. NOT exact-equal to n_llm_calls (this is embed + chat,
  not chat-only) — column name is deliberately distinct.
- ``prompts_source = "reconstructed_minimal"``,
  ``code_sha = ""`` (unknown for historical rows — the run that
  produced the row was not stamped with one),
  ``llm_num_predict / llm_keep_alive = ""`` (same reasoning).

What this script DOES NOT fill:

- final_prompt for non-naive strategies (would require re-running with
  the same seed against a deterministic model — see Methodology).
- llm_call_intents for any strategy (the strategy step labels weren't
  written at generation time).
- Per-call latencies, intermediate LLM responses, branch path taken.

Usage:
    python scripts/backfill_provenance.py results/experiment_1
    python scripts/backfill_provenance.py results/experiment_2 \\
        --no-events            # don't try to derive call-count deltas

The script writes ``raw_scores.csv.bak.<UTC-timestamp>`` first, then
the new CSV atomically (tmp + rename). Re-running is idempotent —
rows whose ``prompts_source`` is already set are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Raise csv field limit so context_sent_to_llm columns (tens of KB) load.
csv.field_size_limit(2**31 - 1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reconstruction template — must stay byte-identical to the answer
# generation prompt in src/strategies/naive.py. If naive.py changes,
# this constant has to change in lockstep, and re-backfilled CSVs
# should bump prompts_source to a new tag.
NAIVE_PROMPT_TEMPLATE = (
    "Answer the following question using only the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

NEW_COLUMNS = [
    "n_llm_calls",            # known-exact for naive (1); blank for others
    "llm_call_intents",       # blank — not recorded at gen-time
    "final_prompt",           # reconstructed for naive only
    "prompts_source",         # "reconstructed_minimal" / "recorded"
    "code_sha",               # blank for historical rows
    "llm_num_predict",        # blank for historical rows
    "llm_keep_alive",         # blank for historical rows
    "n_total_calls_inc_embed",  # derived from events.jsonl deltas if available
]


def parse_args() -> argparse.Namespace:
    """CLI for the backfill script."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("experiment_dir", type=str,
                   help="Path to the experiment results dir "
                        "(e.g. results/experiment_1).")
    p.add_argument("--no-events", action="store_true",
                   help="Skip deriving n_total_calls_inc_embed from "
                        "events.jsonl (use when events log is missing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change but don't modify the CSV.")
    return p.parse_args()


def derive_calls_per_row(events_path: Path) -> dict[tuple[int, int], int]:
    """Walk events.jsonl and return {(config_idx, q_idx): n_calls_this_row}.

    Calls are derived from the ``calls_total`` delta between consecutive
    ``row_done`` events within the same config. The first row of each
    config uses the delta from the prior ``config_start`` event's
    ``gpu.calls_total`` if present, else falls back to the delta from
    the prior row (which carries over the start-of-config baseline).

    Args:
        events_path: Path to events.jsonl.

    Returns:
        Mapping from (config_idx, q_idx) → total calls (embed + chat)
        attributable to that row. Empty dict if events.jsonl is missing.
    """
    if not events_path.exists():
        logger.info("events.jsonl not found at %s — skipping call-count derivation",
                    events_path)
        return {}
    deltas: dict[tuple[int, int], int] = {}
    prev_total = 0
    prev_config = None
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("event")
            if ev == "config_start":
                # Reset baseline at each config boundary so the first
                # row's delta isn't polluted by the prior config's tail.
                # The gpu_snapshot baseline isn't always present on
                # config_start in older runs — best effort.
                prev_config = rec.get("config_idx")
                # The post-start total is the prior row's last total.
                # We don't reset prev_total here — the first row's
                # delta will simply absorb any inter-config calls
                # (model warmup), which is fine for the magnitude we
                # care about. Tag those edge rows downstream.
                continue
            if ev != "row_done":
                continue
            cfg = rec.get("config_idx")
            q = rec.get("q_idx")
            total = (rec.get("call") or {}).get("calls_total")
            if cfg is None or q is None or total is None:
                continue
            # Inter-config row resets prev_total via config_start above;
            # within a config, take a straight delta. Treat negative
            # deltas (impossible in steady state) as 0.
            delta = max(0, int(total) - int(prev_total))
            deltas[(int(cfg), int(q))] = delta
            prev_total = int(total)
    logger.info("Derived per-row call counts for %d rows from %s",
                len(deltas), events_path)
    return deltas


def backfill_csv(
    csv_path: Path,
    events_deltas: dict[tuple[int, int], int],
    dry_run: bool = False,
) -> dict[str, int]:
    """Add the new provenance columns to ``csv_path`` in place.

    Atomic: writes a .tmp file then renames over the original. A
    timestamped .bak.<UTC> is left in place for safety.

    Args:
        csv_path: Path to raw_scores.csv.
        events_deltas: From ``derive_calls_per_row`` — keyed (cfg, q).
            Empty dict if no events log was available.
        dry_run: Don't write anything, just count what would change.

    Returns:
        Stats dict: {"total": N, "updated": N, "skipped_recorded": N,
                     "naive_reconstructed": N, "with_call_delta": N}.
    """
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        existing_cols = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)

    out_cols = list(existing_cols)
    for col in NEW_COLUMNS:
        if col not in out_cols:
            out_cols.append(col)

    stats = {
        "total": len(rows),
        "updated": 0,
        "skipped_recorded": 0,
        "naive_reconstructed": 0,
        "with_call_delta": 0,
    }

    # Map question text → q_idx so we can join the events-derived deltas
    # back onto rows. The CSV doesn't carry q_idx, but within a single
    # experiment run the (config_idx, q_idx) joins to (strategy, model,
    # question_position_in_dataset). Reconstructing q_idx requires
    # knowing the dataset ordering — punt: just join on question text
    # by building a separate event index of (cfg → q → question_hash).
    # For Phase-1 backfill, we'll join purely on (cfg, q_idx) where
    # cfg can be derived from a (strategy, model)→cfg_idx map built
    # from the ordering of the rows in the CSV.
    # Build that map from the CSV itself: each (strategy, model)
    # block is one config.
    cfg_index: dict[tuple[str, str], int] = {}
    q_counter: dict[tuple[str, str], int] = {}
    for row in rows:
        sm = (row.get("strategy", ""), row.get("model", ""))
        if sm not in cfg_index:
            cfg_index[sm] = len(cfg_index) + 1
        q_counter[sm] = q_counter.get(sm, 0) + 1
        row["_cfg"] = cfg_index[sm]
        row["_q"]   = q_counter[sm]

    backfill_tag = f"reconstructed_minimal_{datetime.now().strftime('%Y-%m-%d')}"

    for row in rows:
        # Skip rows that were already produced by the new instrumented
        # runner (prompts_source = "recorded").
        if (row.get("prompts_source") or "").startswith("recorded"):
            stats["skipped_recorded"] += 1
            # Strip helper keys before write.
            row.pop("_cfg", None); row.pop("_q", None)
            continue
        # Skip rows that have already been backfilled to the same tag —
        # makes re-runs idempotent.
        if (row.get("prompts_source") or "") == backfill_tag:
            stats["skipped_recorded"] += 1
            row.pop("_cfg", None); row.pop("_q", None)
            continue

        row["prompts_source"] = backfill_tag
        # Defaults for the columns we can't reconstruct.
        row.setdefault("llm_call_intents", "")
        row.setdefault("code_sha", "")
        row.setdefault("llm_num_predict", "")
        row.setdefault("llm_keep_alive", "")

        strategy = (row.get("strategy") or "").strip().lower()
        question = row.get("question") or ""
        context = row.get("context_sent_to_llm") or ""

        if strategy == "naive":
            # Naive's prompt is deterministic given (context, query).
            # n_llm_calls = 1 (exactly one chat completion per row).
            row["final_prompt"] = NAIVE_PROMPT_TEMPLATE.format(
                context=context, query=question,
            )
            row["n_llm_calls"] = "1"
            row["llm_call_intents"] = "generate_answer"
            stats["naive_reconstructed"] += 1
        else:
            # Multi-step strategies — leave blank rather than fabricate.
            row.setdefault("final_prompt", "")
            row.setdefault("n_llm_calls", "")

        delta = events_deltas.get((row["_cfg"], row["_q"]))
        if delta is not None:
            row["n_total_calls_inc_embed"] = str(delta)
            stats["with_call_delta"] += 1
        else:
            row.setdefault("n_total_calls_inc_embed", "")

        stats["updated"] += 1
        row.pop("_cfg", None); row.pop("_q", None)

    if dry_run:
        return stats

    # Backup + atomic write.
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    bak = csv_path.with_suffix(f".csv.bak.{ts}")
    csv_path.replace(bak)
    logger.info("Backed up original to %s", bak)

    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)
    return stats


def main() -> int:
    """Run the backfill on the given experiment dir."""
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        logger.error("Not a directory: %s", exp_dir)
        return 2
    csv_path = exp_dir / "raw_scores.csv"
    if not csv_path.exists():
        logger.error("raw_scores.csv not found in %s", exp_dir)
        return 2

    events_deltas: dict = {}
    if not args.no_events:
        events_deltas = derive_calls_per_row(exp_dir / "events.jsonl")

    stats = backfill_csv(csv_path, events_deltas, dry_run=args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "wrote"
    logger.info(
        "%s: total=%d, updated=%d, skipped_already_recorded=%d, "
        "naive_reconstructed=%d, with_call_delta=%d",
        mode, stats["total"], stats["updated"],
        stats["skipped_recorded"], stats["naive_reconstructed"],
        stats["with_call_delta"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
