#!/usr/bin/env python3
"""Generate synthetic raw_scores.csv files for Experiment 1 & 2 dry runs.

Creates minimal but structurally complete CSV files that exercise the
--skip-generation scoring path in both experiment scripts.  Uses 2 real
HotpotQA questions (seed=42) for realistic text content.

Usage:
    python scripts/create_dry_run_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _get_sample_questions() -> list[dict[str, str]]:
    """Load 2 HotpotQA questions for synthetic data.

    Returns:
        List of dicts with 'question', 'gold_answer', 'doc_text' keys.
    """
    try:
        from src.datasets.hotpotqa import load_hotpotqa
        docs, queries = load_hotpotqa(n=2, seed=42)
        samples = []
        for q, d in zip(queries, docs):
            samples.append({
                "question": q.text,
                "gold_answer": q.reference_answer or "unknown",
                "doc_text": d.text[:500],
            })
        return samples
    except Exception:
        # Fallback if HotpotQA dataset not available locally
        return [
            {
                "question": "What is the capital of France?",
                "gold_answer": "Paris",
                "doc_text": "France is a country in Western Europe. Its capital city is Paris.",
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "gold_answer": "William Shakespeare",
                "doc_text": "Romeo and Juliet is a tragedy written by William Shakespeare.",
            },
        ]


def _build_base_row(sample: dict[str, str]) -> dict:
    """Build a base answer row with common columns.

    Args:
        sample: Dict with question, gold_answer, doc_text.

    Returns:
        Dict with all common columns populated with plausible values.
    """
    return {
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        # Synthetic answer: copy gold with minor variation
        "rag_answer": sample["gold_answer"] + " is the answer.",
        "context_sent_to_llm": sample["doc_text"],
        "gold_f1": 0.85,
        "gold_exact_match": False,
        # Metadata columns that the report and BERTScore paths may reference
        "chunk_type": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "num_chunks": 10,
        "embed_provider": "ollama",
        "embed_model": "mxbai-embed-large",
        "embed_dimension": 1024,
        "retrieval_mode": "hybrid",
        "retrieval_top_k": 10,
        "num_chunks_retrieved": 5,
        "context_char_length": len(sample["doc_text"]),
        "reranker_model": "bge",
        "reranker_top_k": 3,
        "llm_provider": "ollama",
        "llm_host": "http://localhost:11434",
        "dataset_name": "hotpotqa",
        "dataset_sample_seed": 42,
        "strategy_latency_ms": 1200.0,
        "scorer_latency_ms": 500.0,
        "total_latency_ms": 1700.0,
        "failure_stage": "correct",
        "failure_stage_confidence": 1.0,
        "failure_stage_method": "gold_match",
        "gold_in_chunks": True,
        "gold_in_retrieved": True,
        "gold_in_context": True,
    }


def create_experiment_1_data(output_dir: Path) -> Path:
    """Create synthetic raw_scores.csv for Experiment 1 dry run.

    2 questions x 2 strategies x 1 model = 4 rows.

    Args:
        output_dir: Directory to write raw_scores.csv.

    Returns:
        Path to the created CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _get_sample_questions()
    strategies = ["naive", "self_rag"]
    model = "qwen3:4b"

    rows = []
    for sample in samples:
        for strategy in strategies:
            row = _build_base_row(sample)
            row["strategy"] = strategy
            row["model"] = model
            row["llm_model"] = model
            row["config_label"] = f"{strategy}__{model.replace(':', '_')}"
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "raw_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created Experiment 1 dry-run data: {csv_path} ({len(df)} rows)")
    return csv_path


def create_experiment_2_data(output_dir: Path) -> Path:
    """Create synthetic raw_scores.csv for Experiment 2 dry run.

    2 questions x 2 chunkers x 1 model = 4 rows.

    Args:
        output_dir: Directory to write raw_scores.csv.

    Returns:
        Path to the created CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _get_sample_questions()
    chunkers = ["fixed_512", "recursive_500_100"]
    model = "qwen3:4b"

    rows = []
    for sample in samples:
        for chunker in chunkers:
            row = _build_base_row(sample)
            row["chunker"] = chunker
            row["strategy"] = "naive"
            row["model"] = model
            row["llm_model"] = model
            row["config_label"] = f"{chunker}__{model.replace(':', '_')}"
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "raw_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created Experiment 2 dry-run data: {csv_path} ({len(df)} rows)")
    return csv_path


def main() -> None:
    """Generate synthetic data for both experiment dry runs."""
    results_dir = PROJECT_ROOT / "results"
    create_experiment_1_data(results_dir / "experiment_1_dry_run")
    create_experiment_2_data(results_dir / "experiment_2_dry_run")
    print("\nDry-run data created. Now run:")
    print("  python scripts/run_experiment_1.py --skip-generation "
          "--output-dir results/experiment_1_dry_run "
          "--scorer google:gemini-2.5-flash-lite --no-gallery")
    print("  python scripts/run_experiment_2.py --skip-generation "
          "--output-dir results/experiment_2_dry_run "
          "--scorer google:gemini-2.5-flash-lite --no-gallery")


if __name__ == "__main__":
    main()
