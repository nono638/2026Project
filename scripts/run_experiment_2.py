"""Experiment 2: Chunking x Model Size — 4 chunkers x 4 Qwen3 models on 200 HotpotQA.

Isolates the effect of chunking strategy on RAG quality across model sizes,
holding the RAG strategy constant (NaiveRAG). This complements Experiment 1
which varies strategy while holding chunking constant.

Matrix: 4 chunkers (RecursiveChunker, FixedSizeChunker, SentenceChunker,
SemanticChunker) x 4 models (qwen3:0.6b, qwen3:1.7b, qwen3:4b, qwen3:8b)
= 16 configurations.

Held constant: NaiveRAG strategy, OllamaEmbedder(mxbai-embed-large),
hybrid retrieval, retrieval_top_k=5, no reranker.

Scorer: Gemini 2.5 Flash — best cost/quality from Experiment 0.

Why Qwen3 only: Exp 2 isolates chunking x model size within one model
family. Gemma is the cross-family variable in Exp 1. Mixing families
would confound the chunking effect with architecture differences.

Usage:
    python scripts/run_experiment_2.py                          # full run
    python scripts/run_experiment_2.py --resume                 # resume interrupted
    python scripts/run_experiment_2.py --models qwen3:4b --chunkers recursive  # subset
    python scripts/run_experiment_2.py --skip-generation        # re-score only
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Import shared utilities (created by task-036)
from experiment_utils import (
    compute_f1,
    exact_match,
    compute_bertscores,
    ensure_model,
    load_hotpotqa_examples,
    generate_answer,
    score_answer,
    load_checkpoint,
    append_rows,
    format_duration,
    build_scorer,
)


# ---------------------------------------------------------------------------
# Experiment matrix — the 16 configurations
# ---------------------------------------------------------------------------

# Chunker keys mapped to human-readable names
# Why dict not list: enables --chunkers filtering by name
ALL_CHUNKERS = {
    "recursive": "RecursiveChunker(500, 100)",
    "fixed": "FixedSizeChunker(500)",
    "sentence": "SentenceChunker()",
    "semantic": "SemanticChunker()",
}

# Qwen3 only — isolates chunking effect within one model family
ALL_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
]

# NaiveRAG held constant — simplest strategy for most direct chunking measurement
STRATEGY = "naive"


def _make_chunker(name: str, ollama_host: str | None = None) -> object:
    """Instantiate a chunker by its short name.

    Args:
        name: Chunker key from ALL_CHUNKERS.
        ollama_host: Ollama host for SemanticChunker (needs embeddings).

    Returns:
        A Chunker instance.
    """
    from src.chunkers.recursive import RecursiveChunker
    from src.chunkers.fixed import FixedSizeChunker
    from src.chunkers.sentence import SentenceChunker
    from src.chunkers.semantic import SemanticChunker

    if name == "recursive":
        return RecursiveChunker(500, 100)
    elif name == "fixed":
        return FixedSizeChunker(500)
    elif name == "sentence":
        return SentenceChunker()
    elif name == "semantic":
        return SemanticChunker()
    else:
        raise ValueError(f"Unknown chunker: {name}")


def _chunker_metadata(chunker: object) -> dict:
    """Extract chunk metadata from a chunker instance.

    Different chunker types have different parameter structures. This
    normalizes them into the standard pipeline metadata columns.

    Args:
        chunker: A Chunker instance with a .name property.

    Returns:
        Dict with chunk_type, chunk_size, chunk_overlap keys.
    """
    name = chunker.name
    if "recursive" in name.lower():
        return {"chunk_type": "recursive", "chunk_size": 500, "chunk_overlap": 100}
    elif "fixed" in name.lower():
        return {"chunk_type": "fixed", "chunk_size": 500, "chunk_overlap": 0}
    elif "sentence" in name.lower():
        return {"chunk_type": "sentence", "chunk_size": None, "chunk_overlap": None}
    elif "semantic" in name.lower():
        return {"chunk_type": name, "chunk_size": None, "chunk_overlap": None}
    else:
        return {"chunk_type": name, "chunk_size": None, "chunk_overlap": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Experiment 2.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 2: Chunking x Model Size — 4 chunkers x 4 Qwen3 models on HotpotQA.",
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of HotpotQA examples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results/experiment_2",
                        help="Output directory (default: results/experiment_2)")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama server URL (default: localhost:11434)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in raw_scores.csv")
    parser.add_argument("--max-cost", type=float, default=10.0,
                        help="Maximum estimated API spend in USD (default: $10.00)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model subset (e.g., 'qwen3:4b,qwen3:8b')")
    parser.add_argument("--chunkers", type=str, default=None,
                        help="Comma-separated chunker subset (e.g., 'recursive,sentence')")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Re-score existing answers without re-generating")
    parser.add_argument("--scorer", type=str, default="google:gemini-2.5-flash",
                        help="Scorer as provider:model (default: google:gemini-2.5-flash)")
    return parser.parse_args()


def validate_models(model_str: str | None) -> list[str]:
    """Validate and filter the --models flag.

    Args:
        model_str: Comma-separated model names, or None for all.

    Returns:
        List of valid model names.

    Raises:
        SystemExit: If any model name is invalid.
    """
    if model_str is None:
        return list(ALL_MODELS)
    requested = [m.strip() for m in model_str.split(",")]
    invalid = [m for m in requested if m not in ALL_MODELS]
    if invalid:
        print(f"ERROR: Invalid model(s): {', '.join(invalid)}")
        print(f"Valid models: {', '.join(ALL_MODELS)}")
        sys.exit(1)
    return requested


def validate_chunkers(chunker_str: str | None) -> list[str]:
    """Validate and filter the --chunkers flag.

    Args:
        chunker_str: Comma-separated chunker names, or None for all.

    Returns:
        List of valid chunker names.

    Raises:
        SystemExit: If any chunker name is invalid.
    """
    if chunker_str is None:
        return list(ALL_CHUNKERS.keys())
    requested = [c.strip() for c in chunker_str.split(",")]
    invalid = [c for c in requested if c not in ALL_CHUNKERS]
    if invalid:
        print(f"ERROR: Invalid chunker(s): {', '.join(invalid)}")
        print(f"Valid chunkers: {', '.join(ALL_CHUNKERS.keys())}")
        sys.exit(1)
    return requested


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame) -> str:
    """Generate a markdown summary report for Experiment 2.

    Includes chunker x model quality heatmap, per-chunker and per-model
    rankings, chunking impact analysis, latency summary, and cost.

    Args:
        df: Results DataFrame with all scores and metadata.

    Returns:
        Markdown report string.
    """
    if df.empty:
        return "# Experiment 2: Chunking x Model Size\n\nNo data available.\n"

    lines = ["# Experiment 2: Chunking x Model Size Report\n"]

    # --- Chunker x Model quality heatmap ---
    lines.append("## Chunker x Model Quality Heatmap\n")
    if "quality" in df.columns and "chunker" in df.columns and "model" in df.columns:
        pivot = df.pivot_table(
            values="quality", index="chunker", columns="model", aggfunc="mean",
        )
        pivot = pivot.round(3)
        lines.append(pivot.to_markdown())
        lines.append("")
    else:
        lines.append("*Missing required columns for heatmap.*\n")

    # --- Per-chunker ranking ---
    lines.append("## Per-Chunker Ranking\n")
    if "quality" in df.columns:
        chunk_stats = df.groupby("chunker")["quality"].agg(["mean", "std", "count"])
        chunk_stats = chunk_stats.sort_values("mean", ascending=False).round(3)
        lines.append(chunk_stats.to_markdown())
        lines.append("")

    # --- Per-model ranking ---
    lines.append("## Per-Model Ranking\n")
    if "quality" in df.columns:
        model_stats = df.groupby("model")["quality"].agg(["mean", "std", "count"])
        model_stats = model_stats.sort_values("mean", ascending=False).round(3)
        lines.append(model_stats.to_markdown())
        lines.append("")

    # --- Chunking impact analysis ---
    # Which chunker gives the biggest lift on small vs large models?
    lines.append("## Chunking Impact Analysis\n")
    lines.append("Quality delta (chunker mean - overall mean) by model size:\n")

    if "quality" in df.columns and "chunker" in df.columns and "model" in df.columns:
        config_means = df.groupby(["chunker", "model"])["quality"].mean()
        overall_mean = df["quality"].mean()

        # Model sizes for ordering
        model_order = {"qwen3:0.6b": 0.6, "qwen3:1.7b": 1.7, "qwen3:4b": 4.0, "qwen3:8b": 8.0}

        lines.append("| Chunker | Model | Mean Quality | Delta vs Overall |")
        lines.append("|---------|-------|-------------|------------------|")

        for chunker in sorted(config_means.index.get_level_values(0).unique()):
            for model in sorted(
                config_means.index.get_level_values(1).unique(),
                key=lambda m: model_order.get(m, 0)
            ):
                mean_q = config_means.get((chunker, model), float("nan"))
                if not math.isnan(mean_q):
                    delta = mean_q - overall_mean
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"| {chunker} | {model} | {mean_q:.3f} | {sign}{delta:.3f} |"
                    )
        lines.append("")

        # Best chunker per model
        lines.append("### Best Chunker per Model\n")
        for model in sorted(
            config_means.index.get_level_values(1).unique(),
            key=lambda m: model_order.get(m, 0)
        ):
            model_configs = config_means.xs(model, level="model")
            if not model_configs.empty:
                best_chunker = model_configs.idxmax()
                best_quality = model_configs.max()
                lines.append(f"- **{model}**: {best_chunker} ({best_quality:.3f})")
        lines.append("")

    # --- Latency summary ---
    lines.append("## Latency Summary\n")
    if "strategy_latency_ms" in df.columns:
        lat_stats = df.groupby(["chunker", "model"])["strategy_latency_ms"].agg(
            ["mean", "median", "std"]
        ).round(0)
        lines.append(lat_stats.to_markdown())
        lines.append("")

    # --- Gold metrics summary ---
    lines.append("## Gold Metrics Summary\n")
    if "gold_f1" in df.columns:
        lines.append(f"- Mean gold F1: {df['gold_f1'].mean():.3f}")
    if "gold_exact_match" in df.columns:
        lines.append(f"- Exact match rate: {df['gold_exact_match'].mean():.1%}")
    if "gold_bertscore" in df.columns:
        lines.append(f"- Mean BERTScore F1: {df['gold_bertscore'].mean():.3f}")
    lines.append("")

    # --- Cost summary ---
    lines.append("## Cost Summary\n")
    n_scored = len(df.dropna(subset=["quality"])) if "quality" in df.columns else 0
    est_cost = n_scored * 0.0001  # Gemini Flash
    lines.append(f"- Total scored answers: {n_scored}")
    lines.append(f"- Estimated scorer cost: ${est_cost:.2f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Experiment 2 pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores_path = output_dir / "raw_scores.csv"
    report_path = output_dir / "report.md"

    # Validate CLI filters
    models = validate_models(args.models)
    chunkers = validate_chunkers(args.chunkers)

    print("=" * 60)
    print("Experiment 2: Chunking x Model Size")
    print("=" * 60)
    print(f"  Chunkers:         {', '.join(chunkers)}")
    print(f"  Models:           {', '.join(models)}")
    print(f"  Total configs:    {len(chunkers) * len(models)}")
    print(f"  Strategy:         {STRATEGY} (held constant)")
    print(f"  HotpotQA examples: {args.n}")
    print(f"  Seed:             {args.seed}")
    print(f"  Scorer:           {args.scorer}")
    print(f"  Max API cost:     ${args.max_cost:.2f}")
    print(f"  Output:           {output_dir}")
    print(f"  Resume:           {args.resume}")
    print(f"  Skip generation:  {args.skip_generation}")
    if args.ollama_host:
        print(f"  Ollama host:      {args.ollama_host}")
    print()

    # Check for resume — checkpoint key is (chunker_name, model)
    completed_configs = set()
    if args.resume:
        completed_configs = load_checkpoint(raw_scores_path)
        if completed_configs:
            logger.info("Resuming — %d configs already completed: %s",
                        len(completed_configs),
                        ", ".join(f"{c}+{m}" for c, m in completed_configs))

    # Build scorer
    from src.cost_guard import CostLimitExceeded

    scorer = build_scorer(args.scorer, max_cost=args.max_cost)

    if args.skip_generation:
        if not raw_scores_path.exists():
            print(f"\nERROR: {raw_scores_path} not found. Run without --skip-generation first.")
            sys.exit(1)
        logger.info("Loading existing answers for re-scoring...")
        existing_df = pd.read_csv(raw_scores_path)

        logger.info("Re-scoring %d answers...", len(existing_df))
        for idx, row in existing_df.iterrows():
            scores = score_answer(
                scorer, row["question"], row.get("doc_text", ""), row["rag_answer"]
            )
            for k, v in scores.items():
                existing_df.at[idx, k] = v

        existing_df.to_csv(raw_scores_path, index=False)
        logger.info("Re-scored and saved to %s", raw_scores_path)
    else:
        # Full generation + scoring run
        logger.info("Loading HotpotQA (n=%d, seed=%d)...", args.n, args.seed)
        docs, queries = load_hotpotqa_examples(n=args.n, seed=args.seed)

        # Connect to Ollama
        try:
            from ollama import Client
            client = Client(host=args.ollama_host) if args.ollama_host else Client()
            client.list()
        except Exception as exc:
            print(f"\nERROR: Cannot connect to Ollama: {exc}")
            print("Please start Ollama and try again.")
            sys.exit(1)

        # Set up held-constant components
        from src.embedders import OllamaEmbedder
        from src.llms import OllamaLLM
        from src.strategies.naive import NaiveRAG

        embedder = OllamaEmbedder(host=args.ollama_host)
        llm = OllamaLLM(host=args.ollama_host)
        strategy = NaiveRAG(llm=llm)

        # Build the config matrix
        config_list = [
            (chunker_name, model_name)
            for chunker_name in chunkers
            for model_name in models
        ]
        total_configs = len(config_list)
        configs_done = 0
        experiment_start = time.perf_counter()
        cost_limit_hit = False

        for config_idx, (chunker_name, model_name) in enumerate(config_list, 1):
            # Build chunker to get its .name property for checkpoint matching
            chunker = _make_chunker(chunker_name, ollama_host=args.ollama_host)
            chunker_full_name = chunker.name
            chunk_meta = _chunker_metadata(chunker)

            # Check if this config is already completed
            # Checkpoint uses (strategy, model) in experiment_utils — for Exp 2,
            # we use (chunker_name, model) since strategy is constant
            if (chunker_name, model_name) in completed_configs:
                logger.info("[config %d/%d] SKIPPING %s x %s (already completed)",
                            config_idx, total_configs, chunker_name, model_name)
                configs_done += 1
                continue

            logger.info("[config %d/%d] %s x %s",
                        config_idx, total_configs, chunker_name, model_name)

            # Ensure model is available
            try:
                ensure_model(client, model_name)
            except Exception as exc:
                logger.error("Failed to pull model %s: %s — skipping model",
                             model_name, exc)
                continue

            # Run all queries for this config
            config_rows = []
            config_start = time.perf_counter()

            for q_idx, (doc, query) in enumerate(zip(docs, queries)):
                # Progress display with ETA
                elapsed = time.perf_counter() - experiment_start
                if configs_done > 0:
                    avg_per_config = elapsed / configs_done
                    remaining_configs = total_configs - configs_done
                    eta = format_duration(avg_per_config * remaining_configs)
                else:
                    eta = "calculating..."

                print(f"\r  [config {config_idx}/{total_configs}] "
                      f"{chunker_name} x {model_name} — "
                      f"query {q_idx + 1}/{len(docs)} "
                      f"(ETA: {eta})", end="", flush=True)

                # Generate answer
                result = generate_answer(
                    strategy=strategy,
                    chunker=chunker,
                    embedder=embedder,
                    retrieval_mode="hybrid",
                    query=query,
                    doc=doc,
                    model=model_name,
                    ollama_host=args.ollama_host,
                )

                # Score answer
                try:
                    scores = score_answer(
                        scorer, query.text, doc.text, result["answer"]
                    )
                except CostLimitExceeded as exc:
                    logger.error("\nCOST LIMIT REACHED: %s", exc)
                    logger.error("Saving partial results for this config...")
                    cost_limit_hit = True
                    scores = {
                        "faithfulness": float("nan"),
                        "relevance": float("nan"),
                        "conciseness": float("nan"),
                        "quality": float("nan"),
                        "scorer_latency_ms": float("nan"),
                    }

                # Build row with all columns
                gold_answer = query.reference_answer or ""
                row = {
                    # Checkpoint key uses chunker short name for resume matching
                    "strategy": STRATEGY,
                    "chunker": chunker_name,
                    "model": model_name,
                    "question": query.text,
                    "gold_answer": gold_answer,
                    "rag_answer": result["answer"],
                    # Gold metrics
                    "gold_f1": result.get("gold_f1", float("nan")),
                    "gold_exact_match": result.get("gold_exact_match", False),
                    # Scorer metrics
                    "faithfulness": scores.get("faithfulness", float("nan")),
                    "relevance": scores.get("relevance", float("nan")),
                    "conciseness": scores.get("conciseness", float("nan")),
                    "quality": scores.get("quality", float("nan")),
                    # Latency
                    "strategy_latency_ms": result.get("strategy_latency_ms", float("nan")),
                    "scorer_latency_ms": scores.get("scorer_latency_ms", float("nan")),
                    "total_latency_ms": (
                        result.get("strategy_latency_ms", 0) +
                        scores.get("scorer_latency_ms", 0)
                    ),
                    # Pipeline metadata
                    **chunk_meta,
                    "num_chunks": result.get("num_chunks", 0),
                    "embed_provider": "ollama",
                    "embed_model": "mxbai-embed-large",
                    "embed_dimension": 1024,
                    "retrieval_mode": "hybrid",
                    "retrieval_top_k": 5,
                    "num_chunks_retrieved": result.get("num_chunks_retrieved", 0),
                    "context_char_length": result.get("context_char_length", 0),
                    "reranker_model": None,
                    "reranker_top_k": None,
                    "llm_provider": "ollama",
                    "llm_host": args.ollama_host or "local",
                    "llm_model": model_name,
                    "dataset_name": "hotpotqa",
                    "dataset_sample_seed": args.seed,
                }
                config_rows.append(row)

                if cost_limit_hit:
                    break

            # Checkpoint: flush this config's rows to CSV
            print()  # newline after progress display
            config_elapsed = time.perf_counter() - config_start
            logger.info("Config %s x %s done: %d queries in %s",
                        chunker_name, model_name, len(config_rows),
                        format_duration(config_elapsed))

            append_rows(raw_scores_path, config_rows)
            configs_done += 1

            if cost_limit_hit:
                logger.error("Stopping experiment due to cost limit.")
                break

    # Compute BERTScore in batch at the end
    if raw_scores_path.exists():
        logger.info("Computing BERTScore (batch, local model)...")
        results_df = pd.read_csv(raw_scores_path)

        try:
            preds = results_df["rag_answer"].fillna("").tolist()
            golds = results_df["gold_answer"].fillna("").tolist()
            results_df["gold_bertscore"] = compute_bertscores(preds, golds)
            results_df.to_csv(raw_scores_path, index=False)
            logger.info("BERTScore computed for %d answers.", len(results_df))
        except Exception as exc:
            logger.warning("BERTScore computation failed: %s — skipping column.", exc)

        # Generate report
        report = generate_report(results_df)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Saved report to %s", report_path)
        print("\n" + report)
    else:
        logger.warning("No results file found — skipping report.")

    print("\n" + "=" * 60)
    print("Experiment 2 complete.")
    print(f"  Raw scores: {raw_scores_path}")
    print(f"  Report:     {report_path}")
    if cost_limit_hit:
        print("  WARNING: Cost limit was reached — results are partial.")
    print("=" * 60)


if __name__ == "__main__":
    main()
