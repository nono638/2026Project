"""Entry point for running RAGBench experiments.

Usage:
    python scripts/run_experiment.py                    # full experiment
    python scripts/run_experiment.py --quick            # 2 strategies, 2 models (testing)
    python scripts/run_experiment.py --models qwen3:4b  # specific model only
    python scripts/run_experiment.py --dataset hotpotqa --sample 100  # built-in dataset
    python scripts/run_experiment.py --chunkers recursive,sentence --embedder google
    python scripts/run_experiment.py --llm-backend openai-compat --llm-base-url http://localhost:1234
"""

from __future__ import annotations

import argparse
import os
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

# Valid options for new CLI flags
VALID_CHUNKERS = {"semantic", "fixed", "recursive", "sentence"}
VALID_EMBEDDERS = {"ollama", "huggingface", "google"}
VALID_DATASETS = {"hotpotqa", "squad"}
VALID_RETRIEVAL_MODES = {"hybrid", "dense", "sparse"}
VALID_LLM_BACKENDS = {"ollama", "openai-compat"}
VALID_SCORER_PROVIDERS = {"anthropic", "google"}
DEFAULT_SCORER_MODELS = {
    "google": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}


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
  python scripts/run_experiment.py --dataset hotpotqa --sample 100
  python scripts/run_experiment.py --chunkers recursive,sentence --embedder google
  python scripts/run_experiment.py --llm-backend openai-compat --llm-base-url http://localhost:1234
  python scripts/run_experiment.py --retrieval-mode dense
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run minimal configuration for testing (2 strategies, 2 models, recursive chunker, ollama embedder)",
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
        help="Path to corpus CSV file (default: first CSV in data/). Mutually exclusive with --dataset.",
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
        help="Number of documents to sample (default: all for CSV, 50 for built-in datasets)",
    )
    parser.add_argument(
        "--chunkers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of chunker names. "
            f"Valid: {', '.join(sorted(VALID_CHUNKERS))}. "
            "Default: semantic,fixed,recursive (or recursive in --quick mode)."
        ),
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="ollama",
        help=(
            "Embedder to use (one per run). "
            f"Valid: {', '.join(sorted(VALID_EMBEDDERS))}. "
            "Default: ollama. 'google' requires GOOGLE_API_KEY env var."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Built-in dataset to use instead of --corpus. "
            f"Valid: {', '.join(sorted(VALID_DATASETS))}. "
            "Overrides --corpus when set."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        help=(
            "Retrieval mode for the experiment. "
            f"Valid: {', '.join(sorted(VALID_RETRIEVAL_MODES))}. "
            "Default: hybrid."
        ),
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="ollama",
        help=(
            "LLM backend for strategy generation. "
            f"Valid: {', '.join(sorted(VALID_LLM_BACKENDS))}. "
            "Default: ollama."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Base URL for openai-compat backend (default: http://localhost:1234/v1). Ignored unless --llm-backend is openai-compat.",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help=(
            "Ollama server URL (default: localhost:11434). "
            "Use RunPod proxy URL for remote GPU."
        ),
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="google:gemini-2.0-flash",
        help=(
            "Scorer as provider:model. "
            f"Providers: {', '.join(sorted(VALID_SCORER_PROVIDERS))}. "
            "Default: google:gemini-2.0-flash. "
            "Requires GOOGLE_API_KEY or ANTHROPIC_API_KEY env var."
        ),
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and exit with clear error on invalid input.

    Args:
        args: Parsed argument namespace.
    """
    # Mutual exclusivity: --dataset and --corpus
    if args.dataset and args.corpus:
        print("ERROR: --dataset and --corpus are mutually exclusive. Use one or the other.")
        sys.exit(1)

    # Validate --chunkers
    if args.chunkers:
        for name in args.chunkers.split(","):
            name = name.strip()
            if name not in VALID_CHUNKERS:
                print(f"ERROR: Unknown chunker '{name}'. Valid options: {', '.join(sorted(VALID_CHUNKERS))}")
                sys.exit(1)

    # Validate --embedder
    if args.embedder not in VALID_EMBEDDERS:
        print(f"ERROR: Unknown embedder '{args.embedder}'. Valid options: {', '.join(sorted(VALID_EMBEDDERS))}")
        sys.exit(1)

    # Validate --dataset
    if args.dataset and args.dataset not in VALID_DATASETS:
        print(f"ERROR: Unknown dataset '{args.dataset}'. Valid options: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)

    # Validate --retrieval-mode
    retrieval_mode = getattr(args, "retrieval_mode", "hybrid")
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        print(f"ERROR: Unknown retrieval mode '{retrieval_mode}'. Valid options: {', '.join(sorted(VALID_RETRIEVAL_MODES))}")
        sys.exit(1)

    # Validate --llm-backend
    llm_backend = getattr(args, "llm_backend", "ollama")
    if llm_backend not in VALID_LLM_BACKENDS:
        print(f"ERROR: Unknown LLM backend '{llm_backend}'. Valid options: {', '.join(sorted(VALID_LLM_BACKENDS))}")
        sys.exit(1)

    # Google embedder requires API key
    if args.embedder == "google" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: --embedder google requires GOOGLE_API_KEY environment variable.")
        print("  Set it in your .env file or export GOOGLE_API_KEY=<your-key>")
        sys.exit(1)

    # Warn if --llm-base-url provided without openai-compat
    if args.llm_base_url and llm_backend != "openai-compat":
        print("WARNING: --llm-base-url is ignored unless --llm-backend is openai-compat")

    # Validate --scorer
    if ":" not in args.scorer:
        print(f"ERROR: --scorer must be provider:model (e.g., google:gemini-2.0-flash). Got: '{args.scorer}'")
        sys.exit(1)
    scorer_provider = args.scorer.split(":", 1)[0]
    if scorer_provider not in VALID_SCORER_PROVIDERS:
        print(f"ERROR: Unknown scorer provider '{scorer_provider}'. Valid: {', '.join(sorted(VALID_SCORER_PROVIDERS))}")
        sys.exit(1)
    # Check for required API key
    key_env = {"google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if not os.environ.get(key_env[scorer_provider]):
        print(f"ERROR: --scorer {scorer_provider} requires {key_env[scorer_provider]} environment variable.")
        sys.exit(1)


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


def _build_llm(args: argparse.Namespace):
    """Build the LLM backend from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        An LLM instance (OllamaLLM or OpenAICompatibleLLM).
    """
    llm_backend = getattr(args, "llm_backend", "ollama")

    if llm_backend == "openai-compat":
        from src.llms import OpenAICompatibleLLM
        # Use custom base URL if provided, otherwise default
        kwargs = {}
        if args.llm_base_url:
            kwargs["base_url"] = args.llm_base_url
        return OpenAICompatibleLLM(**kwargs)
    else:
        # Default: Ollama — pass host for remote Ollama support
        from src.llms import OllamaLLM
        return OllamaLLM(host=getattr(args, "ollama_host", None))


def _build_embedder(args: argparse.Namespace):
    """Build the embedder from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        An Embedder instance.
    """
    if args.embedder == "huggingface":
        from src.embedders import HuggingFaceEmbedder
        return HuggingFaceEmbedder()
    elif args.embedder == "google":
        from src.embedders import GoogleTextEmbedder
        return GoogleTextEmbedder()
    else:
        # Default: Ollama — pass host for remote Ollama support
        from src.embedders import OllamaEmbedder
        return OllamaEmbedder(host=getattr(args, "ollama_host", None))


def _build_chunkers(args: argparse.Namespace) -> list:
    """Build the chunker list from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of Chunker instances.
    """
    from src.chunkers import SemanticChunker, FixedSizeChunker, RecursiveChunker, SentenceChunker

    chunker_map = {
        "semantic": SemanticChunker,
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
    }

    if args.chunkers:
        # User specified chunkers
        names = [n.strip() for n in args.chunkers.split(",")]
        return [chunker_map[name]() for name in names]
    elif args.quick:
        # Quick mode: recursive only (project default baseline)
        return [RecursiveChunker()]
    else:
        # Default: semantic, fixed, recursive (original behavior)
        return [SemanticChunker(), FixedSizeChunker(), RecursiveChunker()]


def _build_scorer(args: argparse.Namespace):
    """Build the scorer from --scorer flag.

    Args:
        args: Parsed command-line arguments.

    Returns:
        An LLMScorer instance.
    """
    from src.scorers import LLMScorer
    provider, model = args.scorer.split(":", 1)
    return LLMScorer(provider=provider, model=model)


def build_components(
    args: argparse.Namespace,
) -> tuple[list, list, list, list]:
    """Build the experiment component lists based on CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (chunkers, embedders, strategies, models) lists.
    """
    from src.strategies import NaiveRAG, SelfRAG, MultiQueryRAG, CorrectiveRAG, AdaptiveRAG

    # Build LLM backend — shared by all strategies
    llm = _build_llm(args)

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

    # Build chunkers and embedder
    chunkers = _build_chunkers(args)
    embedders = [_build_embedder(args)]

    return chunkers, embedders, strategies, models


def _load_builtin_dataset(
    dataset_name: str,
    sample_n: int,
) -> tuple[list, list, list]:
    """Load a built-in dataset and return (corpus_dicts, query_dicts, queries).

    Built-in datasets return (Document, Query) objects. These are converted
    to the dict format that Experiment.load_corpus expects.

    Args:
        dataset_name: Name of the built-in dataset ('hotpotqa' or 'squad').
        sample_n: Number of examples to sample. 0 means use default (50).

    Returns:
        Tuple of (corpus_dicts, query_dicts, original Query objects).
    """
    from src.document import documents_to_dicts

    # Default sample size for built-in datasets is 50
    n = sample_n if sample_n > 0 else 50

    if dataset_name == "hotpotqa":
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
        print(f"Loading HotpotQA dataset...")
        documents, queries = load_hotpotqa()
        print(f"  Full dataset: {len(documents)} examples")
        documents, queries = sample_hotpotqa(documents, queries, n=n, seed=42)
        print(f"  Sampled: {len(documents)} examples")
    elif dataset_name == "squad":
        from src.datasets.squad import load_squad, sample_squad
        print(f"Loading SQuAD 2.0 dataset...")
        documents, queries = load_squad()
        print(f"  Full dataset: {len(documents)} examples")
        documents, queries = sample_squad(documents, queries, n=n, seed=42)
        print(f"  Sampled: {len(documents)} examples")
    else:
        print(f"ERROR: Unknown dataset '{dataset_name}'")
        sys.exit(1)

    # Convert Document objects to dicts for Experiment.load_corpus
    corpus_dicts = documents_to_dicts(documents)

    # Convert Query objects to dicts for Experiment.load_corpus
    query_dicts = [
        {"text": q.text, "type": q.query_type}
        for q in queries
    ]

    return corpus_dicts, query_dicts, queries


def main() -> None:
    """Run the experiment with the configured parameters."""
    args = parse_args()

    print("RAGBench Experiment Runner")
    print("=" * 40)

    # Load corpus — either from built-in dataset or CSV
    queries_to_save = None

    if args.dataset:
        # Built-in dataset mode
        corpus_dicts, query_dicts, queries_to_save = _load_builtin_dataset(
            args.dataset, args.sample,
        )
        print(f"Dataset: {args.dataset} ({len(corpus_dicts)} examples)")
    else:
        # CSV corpus mode (original behavior)
        corpus_path = find_corpus(args.corpus)
        if not corpus_path:
            sys.exit(1)

        print(f"Corpus: {corpus_path}")

        from src.document import load_corpus_from_csv, sample_corpus, documents_to_dicts

        documents = load_corpus_from_csv(corpus_path)
        print(f"Loaded {len(documents)} documents")

        if args.sample > 0:
            documents = sample_corpus(documents, n=args.sample)
            print(f"Sampled {len(documents)} documents")

        corpus_dicts = documents_to_dicts(documents)
        query_dicts = None  # CSV mode doesn't have pre-built queries

    # Build components
    chunkers, embedders, strategies, models = build_components(args)
    scorer = _build_scorer(args)

    # Get retrieval mode
    retrieval_mode = getattr(args, "retrieval_mode", "hybrid")

    print(f"Chunkers:       {[c.name for c in chunkers]}")
    print(f"Embedders:      {[e.name for e in embedders]}")
    print(f"Strategies:     {[s.name for s in strategies]}")
    print(f"Models:         {models}")
    print(f"Scorer:         {scorer.name}")
    print(f"Retrieval mode: {retrieval_mode}")
    total_configs = len(chunkers) * len(embedders) * len(strategies) * len(models)
    print(f"Total configurations: {total_configs}")
    print()

    # Save queries for reproducibility when using built-in datasets
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if queries_to_save:
        from src.query import save_queries
        queries_path = output_dir / "queries.json"
        save_queries(queries_to_save, queries_path)
        print(f"Queries saved to: {queries_path}")

    # Run experiment
    from src.experiment import Experiment

    # Determine LLM provider/host for metadata
    llm_backend = getattr(args, "llm_backend", "ollama")
    llm_host = getattr(args, "llm_base_url", None) or getattr(args, "ollama_host", None) or "local"

    experiment = Experiment(
        chunkers=chunkers,
        embedders=embedders,
        strategies=strategies,
        models=models,
        scorer=scorer,
        retrieval_mode=retrieval_mode,
        dataset_name=getattr(args, "dataset", None),
        dataset_sample_seed=42 if getattr(args, "dataset", None) else None,
        llm_provider=llm_backend,
        llm_host=llm_host,
    )

    # Load corpus into experiment
    if query_dicts:
        experiment.load_corpus(corpus_dicts, query_dicts)
    else:
        experiment.load_corpus(corpus_dicts, [])

    print("Starting experiment...")
    result = experiment.run()

    # Save results
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
