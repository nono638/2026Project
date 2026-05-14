"""Summarize Exp 2-E embedder size sweep across all completed sweep runs.

Reads raw_scores.csv from each embedder-specific results dir, computes mean
consensus_quality per (embedder, chunker, model), and writes:

  results/experiment_2_e/summary.csv  -- one row per (embedder, chunker, model)
  results/experiment_2_e/summary.md   -- human-readable tables

The original Experiment 2 main run (results/experiment_2/, qwen3-embedding:4b)
is included as the largest embedder point. The all-minilm:22m run only has
3/4 chunkers (semantic was skipped due to 512-token context); it's reported
as-is and flagged in the markdown.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ordered small-to-large for the sweep dimension. (name, results_dir, params_M)
EMBEDDER_RUNS = [
    ("all-minilm:22m",      "results/experiment_2_22m",   22),
    ("nomic-embed-text",    "results/experiment_2_137m",  137),
    ("embeddinggemma:300m", "results/experiment_2_300m",  300),
    ("qwen3-embedding:0.6b","results/experiment_2_600m",  600),
    ("qwen3-embedding:4b",  "results/experiment_2",       2500),  # ~2.5B effective
]


def fnum(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize() -> None:
    out_dir = PROJECT_ROOT / "results" / "experiment_2_e"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for embedder_name, rel_dir, params_M in EMBEDDER_RUNS:
        csv_path = PROJECT_ROOT / rel_dir / "raw_scores.csv"
        rows = read_rows(csv_path)
        if not rows:
            print(f"[skip] {embedder_name}: {csv_path} not found")
            continue

        # Group by (chunker, model). Normalize "semantic:<embed_model>" -> "semantic"
        # since SemanticChunker.name embeds the embedder tag; for a cross-embedder
        # comparison we want to aggregate the semantic chunker across embedders.
        cells: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            chunker = r.get("chunk_type", "")
            if chunker.startswith("semantic:"):
                chunker = "semantic"
            model = r.get("model", "")
            q = fnum(r.get("consensus_quality", ""))
            if math.isnan(q):
                continue
            cells.setdefault((chunker, model), []).append(q)

        for (chunker, model), qs in sorted(cells.items()):
            summary_rows.append({
                "embedder": embedder_name,
                "embedder_params_M": params_M,
                "chunker": chunker,
                "model": model,
                "n": len(qs),
                "mean_quality": round(statistics.mean(qs), 3),
                "stdev_quality": round(statistics.stdev(qs), 3) if len(qs) > 1 else 0.0,
            })

        n_configs = len(cells)
        n_rows = sum(len(v) for v in cells.values())
        print(f"[ok] {embedder_name:25s} configs={n_configs:>3d} rows={n_rows}")

    # Write CSV
    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "embedder", "embedder_params_M", "chunker", "model",
            "n", "mean_quality", "stdev_quality",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")

    # Write markdown: per-embedder overall mean + per-embedder x chunker table
    md_lines: list[str] = []
    md_lines.append("# Experiment 2-E: Embedder Size Sweep — Summary\n")
    md_lines.append("Same chunker x model matrix from Experiment 2 (4 chunkers x 4 "
                    "Qwen 3.5 models x 200 HotpotQA queries) re-run across 5 "
                    "embedders to isolate the embedder's contribution.\n")

    # Per-embedder overall mean (using each embedder's full completed set)
    md_lines.append("## Overall mean quality by embedder (each embedder's completed configs)\n")
    md_lines.append("| Embedder | Params (M) | Configs run | Mean consensus quality | Stdev |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for embedder_name, _, params_M in EMBEDDER_RUNS:
        rows_e = [r for r in summary_rows if r["embedder"] == embedder_name]
        if not rows_e:
            continue
        all_means = [r["mean_quality"] for r in rows_e]
        configs = len(rows_e)
        flag = ""
        if configs < 16:
            flag = f" (partial; {16 - configs} configs missing)"
        md_lines.append(
            f"| {embedder_name} | {params_M} | {configs}{flag} "
            f"| {statistics.mean(all_means):.3f} | {statistics.stdev(all_means):.3f} |"
        )

    # Fair-comparison subset: only (chunker, model) cells that ALL embedders covered.
    common_cells: set[tuple[str, str]] | None = None
    for embedder_name, _, _ in EMBEDDER_RUNS:
        cells_e = {(r["chunker"], r["model"]) for r in summary_rows if r["embedder"] == embedder_name}
        common_cells = cells_e if common_cells is None else common_cells & cells_e
    common_cells = common_cells or set()

    md_lines.append(
        "\n## Apples-to-apples mean (only the "
        f"{len(common_cells)} (chunker, model) cells present in every embedder's data)\n"
    )
    md_lines.append("| Embedder | Params (M) | Mean | Stdev |")
    md_lines.append("|---|---:|---:|---:|")
    for embedder_name, _, params_M in EMBEDDER_RUNS:
        rows_e = [r for r in summary_rows
                  if r["embedder"] == embedder_name
                  and (r["chunker"], r["model"]) in common_cells]
        if not rows_e:
            continue
        means = [r["mean_quality"] for r in rows_e]
        md_lines.append(
            f"| {embedder_name} | {params_M} | "
            f"{statistics.mean(means):.3f} | "
            f"{statistics.stdev(means) if len(means) > 1 else 0:.3f} |"
        )

    # Per-embedder x chunker heatmap-style table
    md_lines.append("\n## Mean quality by (embedder, chunker), averaged across 4 Qwen 3.5 models\n")
    chunkers_seen = sorted(set(r["chunker"] for r in summary_rows))
    md_lines.append("| Embedder | " + " | ".join(chunkers_seen) + " |")
    md_lines.append("|---" + "|---:" * len(chunkers_seen) + "|")
    for embedder_name, _, _ in EMBEDDER_RUNS:
        cells_row = [embedder_name]
        for ch in chunkers_seen:
            rows_ec = [r["mean_quality"] for r in summary_rows
                       if r["embedder"] == embedder_name and r["chunker"] == ch]
            cells_row.append(f"{statistics.mean(rows_ec):.3f}" if rows_ec else "—")
        md_lines.append("| " + " | ".join(cells_row) + " |")

    out_md = out_dir / "summary.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    summarize()
