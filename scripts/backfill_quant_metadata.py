"""task-053 one-shot backfill: add Ollama model_details to legacy metadata.json.

Reads an experiment directory's metadata.json, looks up Ollama /api/show details
for the model(s) it references, and writes them into a top-level "model_details"
field. Adds a "backfill_note" recording the caveat that the resolved values
reflect *backfill time*, not the original run time.

Idempotent: if "model_details" is already present, the file is left unchanged.

Usage:
    python scripts/backfill_quant_metadata.py results/experiment_0
    python scripts/backfill_quant_metadata.py results/experiment_0_v2 results/experiment_0_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import get_ollama_model_details

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BACKFILL_NOTE = (
    "Backfilled retroactively on {date}. The values reflect what the tag "
    "resolves to at backfill time, not necessarily at original run time. "
    "Original Exp 0 runs predate this audit; quantization is most likely "
    "Q4_K_M (Ollama default) but cannot be confirmed from primary evidence."
)


def _collect_model_tags(metadata: dict) -> list[str]:
    """Pull every Ollama model tag referenced in a metadata.json.

    Looks under config.model (single string), config.models (list), and
    extra.test_models (list). De-duplicated, order preserved.
    """
    tags: list[str] = []
    config = metadata.get("config") or {}
    single = config.get("model")
    if isinstance(single, str) and single:
        tags.append(single)
    for key in ("models", "test_models"):
        candidates = config.get(key) or metadata.get(key) or []
        if isinstance(candidates, list):
            tags.extend(c for c in candidates if isinstance(c, str))
    # Also check top-level extra slot used by some experiments
    extra_models = (metadata.get("extra") or {}).get("test_models") or []
    if isinstance(extra_models, list):
        tags.extend(c for c in extra_models if isinstance(c, str))

    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def backfill_one(experiment_dir: Path, host: str | None = None) -> bool:
    """Backfill model_details into metadata.json for one experiment dir.

    Returns True if metadata was written (i.e. changes were made), False if
    skipped (idempotent — model_details already present, or metadata missing).
    """
    metadata_path = experiment_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning("No metadata.json at %s — skipping", metadata_path)
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON at %s: %s", metadata_path, exc)
        return False

    if "model_details" in metadata:
        logger.info("model_details already present in %s — skipping (idempotent)", metadata_path)
        return False

    tags = _collect_model_tags(metadata)
    if not tags:
        logger.warning("No model tags found in %s — nothing to backfill", metadata_path)
        return False

    logger.info("Backfilling %d model tag(s) in %s: %s",
                len(tags), metadata_path, ", ".join(tags))
    model_details = {tag: get_ollama_model_details(tag, host=host) for tag in tags}
    metadata["model_details"] = model_details
    metadata["backfill_note"] = BACKFILL_NOTE.format(
        date=datetime.now().strftime("%Y-%m-%d"),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Updated %s with model_details + backfill_note", metadata_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add Ollama model_details (quantization, digest, ...) to legacy "
            "experiment metadata.json files. Idempotent."
        ),
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="One or more experiment directories containing metadata.json",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server URL (default: http://localhost:11434)",
    )
    args = parser.parse_args()

    written_any = False
    for d in args.experiment_dirs:
        if backfill_one(d, host=args.ollama_host):
            written_any = True
    sys.exit(0 if written_any or args.experiment_dirs else 1)


if __name__ == "__main__":
    main()
