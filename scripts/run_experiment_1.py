"""Experiment 1: Strategy x Model Size — 5 strategies x 6 models on 200 HotpotQA.

This is the project's core research question: does a smart RAG strategy on a
small model beat a naive strategy on a large model?

Matrix: 5 strategies (NaiveRAG, SelfRAG, MultiQueryRAG, CorrectiveRAG,
AdaptiveRAG) x 6 models (qwen3:0.6b, qwen3:1.7b, qwen3:4b, qwen3:8b,
gemma3:1b, gemma3:4b) = 30 configurations.

Held constant: RecursiveChunker(500, 100), OllamaEmbedder(mxbai-embed-large),
hybrid retrieval, retrieval_top_k=5, no reranker.

Scorer: Gemini 2.5 Flash — best cost/quality from Experiment 0.

Checkpoint/resume: after each (strategy, model) config completes, rows are
flushed to raw_scores.csv. On restart with --resume, completed configs are
skipped. This is critical — 6,000 runs will take hours.

Usage:
    python scripts/run_experiment_1.py                              # full run
    python scripts/run_experiment_1.py --resume                     # resume interrupted
    python scripts/run_experiment_1.py --models qwen3:4b --strategies naive  # subset
    python scripts/run_experiment_1.py --skip-generation            # re-score only
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

# Import shared utilities
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
# Experiment matrix — the 30 configurations
# ---------------------------------------------------------------------------

# Strategy keys mapped to constructor functions
# Why dict not list: enables --strategies filtering by name
ALL_STRATEGIES = {
    "naive": "NaiveRAG",
    "self_rag": "SelfRAG",
    "multi_query": "MultiQueryRAG",
    "corrective": "CorrectiveRAG",
    "adaptive": "AdaptiveRAG",
}

ALL_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "gemma3:1b",
    "gemma3:4b",
]


def _make_strategy(name: str, llm: object) -> object:
    """Instantiate a strategy by its short name.

    Args:
        name: Strategy key from ALL_STRATEGIES.
        llm: An LLM instance to pass to the strategy constructor.

    Returns:
        A Strategy instance.
    """
    from src.strategies.naive import NaiveRAG
    from src.strategies.self_rag import SelfRAG
    from src.strategies.multi_query import MultiQueryRAG
    from src.strategies.corrective import CorrectiveRAG
    from src.strategies.adaptive import AdaptiveRAG

    strategy_map = {
        "naive": NaiveRAG,
        "self_rag": SelfRAG,
        "multi_query": MultiQueryRAG,
        "corrective": CorrectiveRAG,
        "adaptive": AdaptiveRAG,
    }
    return strategy_map[name](llm=llm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Experiment 1.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 1: Strategy x Model Size — 5 strategies x 6 models on HotpotQA.",
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of HotpotQA examples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results/experiment_1",
                        help="Output directory (default: results/experiment_1)")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama server URL (default: localhost:11434)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in raw_scores.csv")
    parser.add_argument("--max-cost", type=float, default=10.0,
                        help="Maximum estimated API spend in USD (default: $10.00)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model subset (e.g., 'qwen3:4b,gemma3:1b')")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy subset (e.g., 'naive,self_rag')")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Re-score existing answers without re-generating")
    parser.add_argument("--scorer", type=str, default="google:gemini-2.5-flash",
                        help="Scorer as provider:model (default: google:gemini-2.5-flash)")
    parser.add_argument("--no-gallery", action="store_true",
                        help="Skip automatic gallery regeneration after experiment completes")
    return parser.parse_args()


def validate_models(model_str: str | None) -> list[str]:
    """Validate and filter the --models flag.

    Args:
        model_str: Comma-separated model names, or None for all models.

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


def validate_strategies(strategy_str: str | None) -> list[str]:
    """Validate and filter the --strategies flag.

    Args:
        strategy_str: Comma-separated strategy names, or None for all.

    Returns:
        List of valid strategy names.

    Raises:
        SystemExit: If any strategy name is invalid.
    """
    if strategy_str is None:
        return list(ALL_STRATEGIES.keys())
    requested = [s.strip() for s in strategy_str.split(",")]
    invalid = [s for s in requested if s not in ALL_STRATEGIES]
    if invalid:
        print(f"ERROR: Invalid strategy(ies): {', '.join(invalid)}")
        print(f"Valid strategies: {', '.join(ALL_STRATEGIES.keys())}")
        sys.exit(1)
    return requested


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame) -> str:
    """Generate a markdown summary report for Experiment 1.

    Includes strategy x model quality heatmap, per-strategy and per-model
    rankings, "strategy beats size" analysis, latency summary, and cost.

    Args:
        df: Results DataFrame with all scores and metadata.

    Returns:
        Markdown report string.
    """
    if df.empty:
        return "# Experiment 1: Strategy x Model Size\n\nNo data available.\n"

    lines = ["# Experiment 1: Strategy x Model Size Report\n"]

    # --- Strategy x Model quality heatmap ---
    lines.append("## Strategy x Model Quality Heatmap\n")
    if "quality" in df.columns and "strategy" in df.columns and "model" in df.columns:
        pivot = df.pivot_table(
            values="quality", index="strategy", columns="model", aggfunc="mean",
        )
        # Round for readability
        pivot = pivot.round(3)
        lines.append(pivot.to_markdown())
        lines.append("")
    else:
        lines.append("*Missing required columns for heatmap.*\n")

    # --- Per-strategy ranking ---
    lines.append("## Per-Strategy Ranking\n")
    if "quality" in df.columns:
        strat_stats = df.groupby("strategy")["quality"].agg(["mean", "std", "count"])
        strat_stats = strat_stats.sort_values("mean", ascending=False).round(3)
        lines.append(strat_stats.to_markdown())
        lines.append("")

    # --- Per-model ranking ---
    lines.append("## Per-Model Ranking\n")
    if "quality" in df.columns:
        model_stats = df.groupby("model")["quality"].agg(["mean", "std", "count"])
        model_stats = model_stats.sort_values("mean", ascending=False).round(3)
        lines.append(model_stats.to_markdown())
        lines.append("")

    # --- Strategy beats size analysis ---
    # Cases where small_model + smart_strategy > large_model + naive
    lines.append("## Strategy Beats Size Analysis\n")
    lines.append("Cases where a smaller model with a non-naive strategy outperforms "
                 "a larger model with NaiveRAG:\n")

    if "quality" in df.columns and "strategy" in df.columns and "model" in df.columns:
        config_means = df.groupby(["strategy", "model"])["quality"].mean()

        # Model sizes for ordering (approximate parameter counts)
        model_sizes = {
            "qwen3:0.6b": 0.6, "gemma3:1b": 1.0, "qwen3:1.7b": 1.7,
            "gemma3:4b": 4.0, "qwen3:4b": 4.0, "qwen3:8b": 8.0,
        }

        beats_count = 0
        beats_examples = []

        for strat in config_means.index.get_level_values(0).unique():
            if strat == "naive":
                continue
            for small_model in config_means.index.get_level_values(1).unique():
                small_size = model_sizes.get(small_model, 0)
                small_quality = config_means.get((strat, small_model), None)
                if small_quality is None or math.isnan(small_quality):
                    continue

                for large_model in config_means.index.get_level_values(1).unique():
                    large_size = model_sizes.get(large_model, 0)
                    if large_size <= small_size:
                        continue
                    naive_quality = config_means.get(("naive", large_model), None)
                    if naive_quality is None or math.isnan(naive_quality):
                        continue

                    if small_quality > naive_quality:
                        beats_count += 1
                        delta = small_quality - naive_quality
                        beats_examples.append(
                            f"- {strat} + {small_model} ({small_quality:.3f}) > "
                            f"naive + {large_model} ({naive_quality:.3f}) "
                            f"[+{delta:.3f}]"
                        )

        lines.append(f"**{beats_count} cases found.**\n")
        # Show top 10 by delta
        if beats_examples:
            for ex in beats_examples[:20]:
                lines.append(ex)
        lines.append("")

    # --- Latency summary ---
    lines.append("## Latency Summary\n")
    if "strategy_latency_ms" in df.columns:
        lat_stats = df.groupby(["strategy", "model"])["strategy_latency_ms"].agg(
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
    # Gemini Flash: ~$0.0001 per scoring call
    est_cost = n_scored * 0.0001
    lines.append(f"- Total scored answers: {n_scored}")
    lines.append(f"- Estimated scorer cost: ${est_cost:.2f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Experiment 1 pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores_path = output_dir / "raw_scores.csv"
    report_path = output_dir / "report.md"

    # Validate CLI filters
    models = validate_models(args.models)
    strategies = validate_strategies(args.strategies)

    print("=" * 60)
    print("Experiment 1: Strategy x Model Size")
    print("=" * 60)
    print(f"  Strategies:       {', '.join(strategies)}")
    print(f"  Models:           {', '.join(models)}")
    print(f"  Total configs:    {len(strategies) * len(models)}")
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

    # Check for resume
    completed_configs = set()
    if args.resume:
        completed_configs = load_checkpoint(raw_scores_path)
        if completed_configs:
            logger.info("Resuming — %d configs already completed: %s",
                        len(completed_configs),
                        ", ".join(f"{s}+{m}" for s, m in completed_configs))

    # Build scorer
    from src.cost_guard import CostLimitExceeded

    scorer = build_scorer(args.scorer, max_cost=args.max_cost)

    if args.skip_generation:
        # Re-score existing answers
        if not raw_scores_path.exists():
            print(f"\nERROR: {raw_scores_path} not found. Run without --skip-generation first.")
            sys.exit(1)
        logger.info("Loading existing answers for re-scoring...")
        existing_df = pd.read_csv(raw_scores_path)

        # Re-score each row using context_sent_to_llm (what the model saw)
        logger.info("Re-scoring %d answers...", len(existing_df))
        for idx, row in existing_df.iterrows():
            scores = score_answer(
                scorer, row["question"],
                row.get("context_sent_to_llm", ""),
                row["rag_answer"],
            )
            for k, v in scores.items():
                existing_df.at[idx, k] = v

        existing_df.to_csv(raw_scores_path, index=False)
        logger.info("Re-scored and saved to %s", raw_scores_path)
    else:
        # Full generation + scoring run
        # Load dataset
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
        from src.chunkers.recursive import RecursiveChunker
        from src.embedders import OllamaEmbedder
        from src.llms import OllamaLLM

        chunker = RecursiveChunker(500, 100)
        embedder = OllamaEmbedder(host=args.ollama_host)

        # Build the config matrix
        config_list = [
            (strat_name, model_name)
            for strat_name in strategies
            for model_name in models
        ]
        total_configs = len(config_list)
        configs_done = 0
        experiment_start = time.perf_counter()
        cost_limit_hit = False

        for config_idx, (strat_name, model_name) in enumerate(config_list, 1):
            if (strat_name, model_name) in completed_configs:
                logger.info("[config %d/%d] SKIPPING %s x %s (already completed)",
                            config_idx, total_configs, strat_name, model_name)
                configs_done += 1
                continue

            logger.info("[config %d/%d] %s x %s", config_idx, total_configs,
                        strat_name, model_name)

            # Ensure model is available
            try:
                ensure_model(client, model_name)
            except Exception as exc:
                logger.error("Failed to pull model %s: %s — skipping model", model_name, exc)
                continue

            # Build strategy with fresh LLM instance
            llm = OllamaLLM(host=args.ollama_host)
            strategy = _make_strategy(strat_name, llm)

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
                      f"{strat_name} x {model_name} — "
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

                # Score answer — use context_sent_to_llm so faithfulness is
                # judged against what the model actually saw (Exp 0 v2 fix)
                scorer_context = result.get("context_sent_to_llm", "")
                try:
                    scores = score_answer(
                        scorer, query.text, scorer_context, result["answer"]
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
                    "strategy": strat_name,
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
                    # Diagnostics
                    "context_sent_to_llm": result.get("context_sent_to_llm", ""),
                    "failure_stage": result.get("failure_stage"),
                    "failure_stage_confidence": result.get("failure_stage_confidence"),
                    "failure_stage_method": result.get("failure_stage_method"),
                    "gold_in_chunks": result.get("gold_in_chunks"),
                    "gold_in_retrieved": result.get("gold_in_retrieved"),
                    "gold_in_context": result.get("gold_in_context"),
                    # Pipeline metadata (held constant)
                    "chunk_type": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 100,
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
                        strat_name, model_name, len(config_rows),
                        format_duration(config_elapsed))

            append_rows(raw_scores_path, config_rows)
            configs_done += 1

            if cost_limit_hit:
                logger.error("Stopping experiment due to cost limit.")
                break

    # Compute BERTScore in batch at the end
    # Why batch: BERTScore loads a ~1.4GB model once, much faster than per-row
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
    print("Experiment 1 complete.")
    print(f"  Raw scores: {raw_scores_path}")
    print(f"  Report:     {report_path}")
    if cost_limit_hit:
        print("  WARNING: Cost limit was reached — results are partial.")
    print("=" * 60)

    # Auto-regenerate gallery unless --no-gallery is set
    if not args.no_gallery:
        try:
            # Lazy import to avoid breaking experiment if gallery deps are missing
            from scripts.generate_gallery import main as generate_gallery
            print("\nRegenerating gallery...")
            generate_gallery(experiments=[1])
            print("Gallery updated in site/")
        except Exception as exc:
            print(f"Gallery regeneration failed: {exc}")
            logger.warning("Gallery regeneration failed: %s", exc)


if __name__ == "__main__":
    main()
