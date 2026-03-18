"""Entry point for running RAGBench experiments.

Usage:
    python scripts/run_experiment.py                    # full experiment
    python scripts/run_experiment.py --quick            # 2 strategies, 2 models (testing)
    python scripts/run_experiment.py --models qwen3:4b  # specific model only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# All generation models (embedding model excluded)
ALL_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "gemma3:1b",
    "gemma3:4b",
]

ALL_STRATEGIES = ["naive", "self_rag", "multi_query", "corrective", "adaptive"]

# Quick mode: minimal configuration for testing
QUICK_MODELS = ["qwen3:0.6b", "qwen3:4b"]
QUICK_STRATEGIES = ["naive", "self_rag"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run RAGBench experiments across chunker x embedder x strategy x model configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py                         # full experiment matrix
  python scripts/run_experiment.py --quick                 # minimal test run
  python scripts/run_experiment.py --models qwen3:4b       # single model
  python scripts/run_experiment.py --strategies naive,self_rag --sample 50
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run minimal configuration for testing (2 strategies, 2 models, semantic chunker, ollama embedder)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of Ollama model names (default: all 6 generation models)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategy names (default: all 5)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus CSV file (default: first CSV in data/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of documents to sample from corpus (default: all)",
    )

    return parser.parse_args()


def find_corpus(corpus_arg: str | None) -> Path | None:
    """Find the corpus CSV file.

    Args:
        corpus_arg: User-specified path, or None for auto-detection.

    Returns:
        Path to the corpus CSV, or None if not found.
    """
    if corpus_arg:
        path = Path(corpus_arg)
        if path.exists():
            return path
        print(f"ERROR: Corpus file not found: {corpus_arg}")
        return None

    # Auto-detect: first CSV in data/
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        csvs = sorted(data_dir.glob("*.csv"))
        if csvs:
            return csvs[0]

    print("ERROR: No corpus CSV found. Provide --corpus or place a CSV in data/")
    return None


def build_components(
    args: argparse.Namespace,
) -> tuple[list, list, list, list]:
    """Build the experiment component lists based on CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (chunkers, embedders, strategies, models) lists.
    """
    from src.chunkers import SemanticChunker, FixedSizeChunker, RecursiveChunker
    from src.embedders import OllamaEmbedder
    from src.strategies import NaiveRAG, SelfRAG, MultiQueryRAG, CorrectiveRAG, AdaptiveRAG
    from src.llms import OllamaLLM

    # Single LLM backend shared by all strategies
    llm = OllamaLLM()

    strategy_map = {
        "naive": NaiveRAG,
        "self_rag": SelfRAG,
        "multi_query": MultiQueryRAG,
        "corrective": CorrectiveRAG,
        "adaptive": AdaptiveRAG,
    }

    # Select models
    if args.quick:
        models = QUICK_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = ALL_MODELS

    # Select strategies
    if args.quick:
        strategy_names = QUICK_STRATEGIES
    elif args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
    else:
        strategy_names = ALL_STRATEGIES

    strategies = []
    for name in strategy_names:
        if name in strategy_map:
            strategies.append(strategy_map[name](llm=llm))
        else:
            print(f"WARNING: Unknown strategy '{name}', skipping")

    # Chunkers and embedders
    if args.quick:
        chunkers = [SemanticChunker()]
        embedders = [OllamaEmbedder()]
    else:
        chunkers = [
            SemanticChunker(),
            FixedSizeChunker(),
            RecursiveChunker(),
        ]
        embedders = [OllamaEmbedder()]

    return chunkers, embedders, strategies, models


def main() -> None:
    """Run the experiment with the configured parameters."""
    args = parse_args()

    print("RAGBench Experiment Runner")
    print("=" * 40)

    # Find corpus
    corpus_path = find_corpus(args.corpus)
    if not corpus_path:
        sys.exit(1)

    print(f"Corpus: {corpus_path}")

    # Load corpus
    from src.document import load_corpus_from_csv, sample_corpus, documents_to_dicts

    documents = load_corpus_from_csv(corpus_path)
    print(f"Loaded {len(documents)} documents")

    if args.sample > 0:
        documents = sample_corpus(documents, n=args.sample)
        print(f"Sampled {len(documents)} documents")

    corpus_dicts = documents_to_dicts(documents)

    # Build components
    chunkers, embedders, strategies, models = build_components(args)

    print(f"Chunkers:   {[c.name for c in chunkers]}")
    print(f"Embedders:  {[e.name for e in embedders]}")
    print(f"Strategies: {[s.name for s in strategies]}")
    print(f"Models:     {models}")
    total_configs = len(chunkers) * len(embedders) * len(strategies) * len(models)
    print(f"Total configurations: {total_configs}")
    print()

    # Run experiment
    from src.experiment import Experiment

    experiment = Experiment(
        corpus=corpus_dicts,
        chunkers=chunkers,
        embedders=embedders,
        strategies=strategies,
        models=models,
    )

    print("Starting experiment...")
    result = experiment.run()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "experiment_results.parquet"
    csv_path = output_dir / "experiment_results.csv"

    result.to_parquet(parquet_path)
    result.to_csv(csv_path)

    print(f"\nResults saved to:")
    print(f"  {parquet_path}")
    print(f"  {csv_path}")

    # Print summary
    print("\nSummary:")
    print(result.summary())


if __name__ == "__main__":
    main()
