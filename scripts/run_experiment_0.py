"""Experiment 0: Scorer Validation — compare up to 7 LLM judges (4 Gemini + 3 Claude).

v2 improvements over v1:
- Uses experiment_utils.generate_answer() for diagnostics, failure attribution, timing
- BGE reranker (retrieve 10, keep 3) for better context quality
- Scorer receives context_sent_to_llm (not full doc_text) — faithfulness is judged
  against what the LLM actually saw
- Medium+hard questions only (150 default) to avoid ceiling effect from easy questions
- answer_quality column: triangulates BERTScore + F1 + Sonnet agreement

Judges (in order):
  - gemini-2.5-flash-lite  (cheapest baseline)
  - gemini-2.5-flash
  - gemini-2.5-pro
  - gemini-3.1-pro-preview
  - claude-haiku-4-5        (optional)
  - claude-sonnet-4         (optional)
  - claude-opus-4           (optional)

Results go to results/experiment_0_v2/ by default. v1 data in results/experiment_0/
is untouched.

Usage:
    python scripts/run_experiment_0.py                          # full v2 run
    python scripts/run_experiment_0.py --n 10 --model qwen3:0.6b  # quick test
    python scripts/run_experiment_0.py --skip-generation        # re-score only
    python scripts/run_experiment_0.py --output-dir results/experiment_0  # v1 compat
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file for API keys (ANTHROPIC_API_KEY, GOOGLE_API_KEY)
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

# Import shared gold metrics from experiment_utils to avoid duplication
from scripts.experiment_utils import compute_f1, exact_match


# ---------------------------------------------------------------------------
# Scorer configurations — 7 LLM judges (4 Gemini + 3 Claude)
# ---------------------------------------------------------------------------
JUDGE_CONFIGS = [
    # Gemini judges (free via Google AI Studio)
    {"provider": "google", "model": "gemini-2.5-flash-lite"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "google", "model": "gemini-2.5-pro"},
    {"provider": "google", "model": "gemini-3.1-pro-preview"},
    # Anthropic judges (optional — skipped if ANTHROPIC_API_KEY not set)
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"provider": "anthropic", "model": "claude-opus-4-20250514"},
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_bertscores(predictions: list[str], golds: list[str]) -> list[float]:
    """Compute BERTScore F1 between each prediction-gold pair.

    Uses RoBERTa-large, the standard BERTScore model for English. Runs
    locally — no API calls. The model (~1.4GB) downloads on first run.

    Empty strings are replaced with a placeholder token ("[EMPTY]") before
    scoring because the bert_score library crashes on empty input (the
    tokenizer produces zero tokens, triggering an AttributeError). Pairs
    where either side is empty get a score of 0.0 instead of crashing.

    Reference: Zhang et al., "BERTScore: Evaluating Text Generation with
    BERT", ICLR 2020. https://arxiv.org/abs/1904.09675

    Args:
        predictions: List of RAG-generated answers.
        golds: List of gold reference answers.

    Returns:
        List of BERTScore F1 values (0.0 to 1.0), one per pair.
    """
    from bert_score import score

    # Track which pairs have empty strings — these get score 0.0
    empty_mask = [
        not p or not p.strip() or not g or not g.strip()
        for p, g in zip(predictions, golds)
    ]

    # Replace empty strings with placeholder to avoid bert_score crash
    PLACEHOLDER = "[EMPTY]"
    safe_preds = [p if (p and p.strip()) else PLACEHOLDER for p in predictions]
    safe_golds = [g if (g and g.strip()) else PLACEHOLDER for g in golds]

    _, _, f1 = score(
        cands=safe_preds,
        refs=safe_golds,
        lang="en",
        verbose=True,
    )
    results = f1.tolist()

    # Zero out scores for pairs that had empty input
    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            results[i] = 0.0

    return results


def _safe_scorer_name(name: str) -> str:
    """Convert scorer name to a safe column prefix.

    Replaces colons, dashes, and dots with underscores for CSV column names.

    Args:
        name: The scorer's name property (e.g., "google:gemini-2.5-flash").

    Returns:
        Safe column prefix (e.g., "google_gemini_2_5_flash").
    """
    return name.replace(":", "_").replace("-", "_").replace(".", "_")


def _build_reranker(name: str) -> object | None:
    """Build a reranker by name.

    Args:
        name: Reranker name ("bge", "minilm", or "none").

    Returns:
        A reranker instance, or None if name is "none".
    """
    if name == "none":
        return None
    elif name == "bge":
        from src.rerankers.bge import BGEReranker
        return BGEReranker()
    elif name == "minilm":
        from src.rerankers.minilm import MiniLMReranker
        return MiniLMReranker()
    else:
        raise ValueError(f"Unknown reranker: {name}. Choose from: bge, minilm, none")


def compute_answer_quality(
    bertscore: float, f1: float, sonnet_quality: float
) -> str:
    """Compute answer quality label from three metrics.

    Triangulates BERTScore (semantic), F1 (lexical), and Sonnet (LLM judgment)
    to classify answer quality. All three must agree for "good"; any single
    metric can flag "poor". Everything else is "questionable" (metrics disagree).

    Args:
        bertscore: BERTScore F1 (0.0 to 1.0).
        f1: Word-overlap F1 (0.0 to 1.0).
        sonnet_quality: Sonnet judge's quality score (1.0 to 5.0).

    Returns:
        "good", "poor", or "questionable".
    """
    # Poor thresholds — any single metric below these flags the answer
    is_poor = bertscore < 0.85 or f1 < 0.30 or sonnet_quality < 3.0
    # Good thresholds — all metrics must be above these
    is_good = bertscore >= 0.90 and f1 >= 0.50 and sonnet_quality >= 4.0

    if is_poor:
        return "poor"
    elif is_good:
        return "good"
    else:
        return "questionable"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 0: Scorer Validation — compare up to 7 LLM judges on HotpotQA.",
    )
    parser.add_argument("--n", type=int, default=150,
                        help="Number of HotpotQA examples (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--model", type=str, default="qwen3:4b",
                        help="Ollama model for answer generation (default: qwen3:4b)")
    parser.add_argument("--output-dir", type=str, default="results/experiment_0_v2",
                        help="Output directory (default: results/experiment_0_v2)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Load previously generated answers instead of re-running NaiveRAG")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama server URL (default: localhost:11434). "
                             "Use RunPod proxy URL for remote GPU.")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="Maximum estimated API spend in USD before aborting (default: $5.00)")
    parser.add_argument("--judges", type=str, nargs="+", default=None,
                        help="Run only these judges (substring match on model name). "
                             "E.g. --judges pro flash-lite   or   --judges gemini-2.5-pro")
    parser.add_argument("--generation-only", action="store_true",
                        help="Generate answers and save raw_answers.csv, then exit "
                             "(skip scoring). Use with --skip-generation later to score.")
    parser.add_argument("--no-gallery", action="store_true",
                        help="Skip automatic gallery regeneration after experiment completes")
    # v2 flags
    parser.add_argument("--reranker", type=str, default="bge",
                        choices=["bge", "minilm", "none"],
                        help="Reranker to use (default: bge). 'none' disables reranking.")
    parser.add_argument("--reranker-top-k", type=int, default=3,
                        help="Number of chunks to keep after reranking (default: 3)")
    parser.add_argument("--retrieval-top-k", type=int, default=10,
                        help="Number of chunks to retrieve before reranking (default: 10)")
    parser.add_argument("--difficulty", type=str, default="medium,hard",
                        help="Comma-separated HotpotQA difficulties to include (default: medium,hard)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all_answers(
    answers: list[dict],
    output_dir: Path,
    cost_guard: object | None = None,
    judge_filters: list[str] | None = None,
) -> pd.DataFrame:
    """Score each answer with all available LLM judges.

    v2 change: scorer receives context_sent_to_llm (from diagnostics) instead
    of the full doc_text. This means faithfulness is evaluated against what the
    LLM actually saw during generation, not information it never received.

    Args:
        answers: List of dicts from generate_answer() pipeline.
        output_dir: Directory for output files.
        cost_guard: Optional CostGuard instance for tracking API spend.
        judge_filters: If provided, only run judges whose model name contains
            one of these substrings (case-insensitive).

    Returns:
        DataFrame with all base columns + scorer columns.
    """
    from src.scorers.llm import LLMScorer, ScorerError

    # Filter judge configs if --judges was specified
    configs_to_run = JUDGE_CONFIGS
    if judge_filters:
        configs_to_run = [
            c for c in JUDGE_CONFIGS
            if any(f.lower() in c["model"].lower() for f in judge_filters)
        ]
        if not configs_to_run:
            logger.error("No judges matched filters: %s", judge_filters)
            logger.info("Available judges: %s",
                        ", ".join(c["model"] for c in JUDGE_CONFIGS))
            sys.exit(1)

    # Initialize scorers — skip any that fail (missing API key)
    scorers = []
    for config in configs_to_run:
        try:
            scorer = LLMScorer(**config, cost_guard=cost_guard)
            scorers.append(scorer)
            logger.info("Initialized scorer: %s", scorer.name)
        except (ScorerError, Exception) as exc:
            logger.warning("Skipping scorer %s:%s — %s",
                           config["provider"], config["model"], exc)

    if not scorers:
        logger.error("No scorers available. Check API keys.")
        sys.exit(1)

    # Report which scorers were initialized vs skipped
    initialized_names = [s.name for s in scorers]
    all_names = [f"{c['provider']}:{c['model']}" for c in JUDGE_CONFIGS]
    skipped_names = [n for n in all_names if n not in initialized_names]
    logger.info("Initialized %d/%d judges: %s",
                len(scorers), len(JUDGE_CONFIGS), ", ".join(initialized_names))
    if skipped_names:
        logger.info("Skipped %d judges (missing API keys): %s",
                     len(skipped_names), ", ".join(skipped_names))

    # Incremental checkpoint: write each scored row to disk as it completes
    # so that crashes only lose the current row, not all prior work.
    checkpoint_path = output_dir / "raw_scores_checkpoint.csv"

    # Load already-scored example_ids from checkpoint (if resuming after crash)
    scored_ids: set[int] = set()
    if checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_csv(checkpoint_path)
            scored_ids = set(checkpoint_df["example_id"].tolist())
            logger.info("Resuming from checkpoint: %d/%d already scored.",
                        len(scored_ids), len(answers))
        except Exception as exc:
            logger.warning("Could not read checkpoint, starting fresh: %s", exc)

    rows = []
    total = len(answers)

    for i, ans in enumerate(answers):
        # Skip rows already in checkpoint
        if ans["example_id"] in scored_ids:
            logger.info("[%d/%d] Already scored (checkpoint), skipping.", i + 1, total)
            continue

        logger.info("[%d/%d] Scoring: %s", i + 1, total, ans["question"][:60])

        # Base columns
        row = {
            "example_id": ans["example_id"],
            "question": ans["question"],
            "gold_answer": ans["gold_answer"],
            "rag_answer": ans["rag_answer"],
            "gold_exact_match": exact_match(ans["rag_answer"], ans["gold_answer"]),
            "gold_f1": compute_f1(ans["rag_answer"], ans["gold_answer"]),
            # Pipeline metadata
            "chunk_type": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "num_chunks": ans.get("num_chunks"),
            "embed_provider": "ollama",
            "embed_model": "mxbai-embed-large",
            "embed_dimension": 1024,
            "retrieval_mode": "hybrid",
            "retrieval_top_k": ans.get("retrieval_top_k", 10),
            "num_chunks_retrieved": ans.get("num_chunks_retrieved"),
            "context_char_length": ans.get("context_char_length"),
            "reranker_model": ans.get("reranker_model"),
            "reranker_top_k": ans.get("reranker_top_k"),
            "llm_provider": "ollama",
            "llm_host": "local",
            "dataset_name": "hotpotqa",
            "dataset_sample_seed": 42,
            # v2 metadata columns
            "difficulty": ans.get("difficulty"),
            "question_type": ans.get("question_type"),
            # v2 diagnostics columns
            "strategy_latency_ms": ans.get("strategy_latency_ms"),
            "failure_stage": ans.get("failure_stage"),
            "failure_stage_confidence": ans.get("failure_stage_confidence"),
            "failure_stage_method": ans.get("failure_stage_method"),
            "context_sent_to_llm": ans.get("context_sent_to_llm", ""),
            "gold_in_chunks": ans.get("gold_in_chunks"),
            "gold_in_retrieved": ans.get("gold_in_retrieved"),
            "gold_in_context": ans.get("gold_in_context"),
        }

        # Score with each judge — v2 fix: use context_sent_to_llm, not doc_text
        scorer_context = ans.get("context_sent_to_llm", ans.get("doc_text", ""))
        for scorer in scorers:
            safe_name = _safe_scorer_name(scorer.name)
            try:
                scores = scorer.score(
                    query=ans["question"],
                    context=scorer_context,
                    answer=ans["rag_answer"],
                )
                for metric, value in scores.items():
                    row[f"{safe_name}_{metric}"] = value
                # Compute quality as mean of the three metrics
                row[f"{safe_name}_quality"] = sum(scores.values()) / len(scores)
            except (ScorerError, Exception) as exc:
                logger.error("Scorer %s failed on example %d: %s",
                             scorer.name, i, exc)
                # Record NaN for all metrics on failure
                for metric in ["faithfulness", "relevance", "conciseness", "quality"]:
                    row[f"{safe_name}_{metric}"] = float("nan")

        rows.append(row)

        # Write row to checkpoint immediately — survives crashes
        row_df = pd.DataFrame([row])
        write_header = not checkpoint_path.exists()
        row_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)

    # Build full result from checkpoint (includes rows from prior runs + this run)
    if checkpoint_path.exists():
        result_df = pd.read_csv(checkpoint_path)
        logger.info("Loaded %d total scored rows from checkpoint.", len(result_df))
    else:
        result_df = pd.DataFrame(rows)

    # Compute BERTScore in batch (loads model once, much faster than per-row)
    logger.info("Computing BERTScore (local model, no API cost)...")
    try:
        preds = result_df["rag_answer"].fillna("").tolist()
        golds = result_df["gold_answer"].fillna("").tolist()
        result_df["gold_bertscore"] = compute_bertscores(preds, golds)
        logger.info("BERTScore computed for %d examples.", len(result_df))
    except Exception as exc:
        logger.error("BERTScore computation failed: %s", exc)

    return result_df


def add_answer_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add answer_quality column to results DataFrame.

    Requires gold_bertscore, gold_f1, and Sonnet quality columns. If Sonnet
    column is missing, logs a warning and returns df unchanged.

    Args:
        df: Results DataFrame with scorer columns.

    Returns:
        DataFrame with answer_quality column added (or unchanged if Sonnet missing).
    """
    sonnet_col = "anthropic_claude_sonnet_4_20250514_quality"
    if sonnet_col not in df.columns:
        logger.warning(
            "answer_quality requires Sonnet scores — column omitted. "
            "Missing column: %s", sonnet_col
        )
        return df

    if "gold_bertscore" not in df.columns or "gold_f1" not in df.columns:
        logger.warning(
            "answer_quality requires gold_bertscore and gold_f1 — column omitted."
        )
        return df

    df["answer_quality"] = df.apply(
        lambda r: compute_answer_quality(
            r["gold_bertscore"], r["gold_f1"], r[sonnet_col]
        ),
        axis=1,
    )
    quality_counts = df["answer_quality"].value_counts().to_dict()
    logger.info("answer_quality distribution: %s", quality_counts)
    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, scorers_used: list[str]) -> str:
    """Generate a markdown summary report.

    Args:
        df: Results DataFrame with all scores.
        scorers_used: List of scorer name strings that were used.

    Returns:
        Markdown report string.
    """
    lines = ["# Experiment 0: Scorer Validation Report\n"]

    # Per-judge mean scores
    lines.append("## Per-Judge Mean Scores\n")
    lines.append("| Judge | Faithfulness | Relevance | Conciseness | Quality |")
    lines.append("|-------|-------------|-----------|-------------|---------|")

    quality_cols = {}
    for name in scorers_used:
        safe = _safe_scorer_name(name)
        f_col = f"{safe}_faithfulness"
        r_col = f"{safe}_relevance"
        c_col = f"{safe}_conciseness"
        q_col = f"{safe}_quality"
        quality_cols[name] = q_col

        if q_col in df.columns:
            lines.append(
                f"| {name} "
                f"| {df[f_col].mean():.3f} "
                f"| {df[r_col].mean():.3f} "
                f"| {df[c_col].mean():.3f} "
                f"| {df[q_col].mean():.3f} |"
            )

    # Inter-scorer correlation matrix (Pearson on quality)
    lines.append("\n## Inter-Scorer Correlation (Pearson, Quality)\n")

    q_df = pd.DataFrame()
    for name in scorers_used:
        col = quality_cols.get(name)
        if col and col in df.columns:
            q_df[name] = df[col]

    if not q_df.empty and len(q_df.columns) > 1:
        corr = q_df.corr()
        lines.append(corr.round(3).to_markdown())
    else:
        lines.append("*Insufficient scorers for correlation matrix.*\n")

    # Correlation with gold metrics
    gold_metrics = []
    if "gold_bertscore" in df.columns:
        gold_metrics.append(("gold_bertscore", "BERTScore"))
    if "gold_f1" in df.columns:
        gold_metrics.append(("gold_f1", "F1 (word overlap)"))

    if gold_metrics:
        header_cols = " | ".join(label for _, label in gold_metrics)
        lines.append(f"\n## Correlation with Gold Metrics\n")
        lines.append(f"| Judge | {header_cols} |")
        lines.append("|-------" + "|----------" * len(gold_metrics) + "|")

        for name in scorers_used:
            col = quality_cols.get(name)
            if col and col in df.columns:
                row_parts = [f"| {name} "]
                for gold_col, _ in gold_metrics:
                    valid = df[[col, gold_col]].dropna()
                    if len(valid) > 2:
                        r = valid[col].corr(valid[gold_col])
                        row_parts.append(f"| {r:.3f} ")
                    else:
                        row_parts.append("| N/A ")
                lines.append("".join(row_parts) + "|")

    # Estimated cost breakdown
    lines.append("\n## Estimated Cost Breakdown\n")
    lines.append("| Judge | Calls | Est. Cost/Call | Est. Total |")
    lines.append("|-------|-------|----------------|------------|")

    cost_estimates = {
        "google:gemini-2.5-flash-lite": 0.00005,
        "google:gemini-2.5-flash": 0.0001,
        "google:gemini-2.5-pro": 0.001,
        "anthropic:claude-haiku-4-5-20251001": 0.001,
        "anthropic:claude-sonnet-4-20250514": 0.005,
    }

    n_examples = len(df)
    for name in scorers_used:
        cost = cost_estimates.get(name, 0.01)
        total = n_examples * cost
        lines.append(f"| {name} | {n_examples} | ${cost:.4f} | ${total:.2f} |")

    # Gold correctness summary
    lines.append("\n## Gold Correctness Summary\n")
    if "gold_exact_match" in df.columns:
        em_rate = df["gold_exact_match"].mean()
        f1_mean = df["gold_f1"].mean()
        lines.append(f"- Exact match rate: {em_rate:.1%}")
        lines.append(f"- Mean word-overlap F1: {f1_mean:.3f}")
    if "gold_bertscore" in df.columns:
        bs_mean = df["gold_bertscore"].mean()
        lines.append(f"- Mean BERTScore F1: {bs_mean:.3f}")

    # v2: answer_quality distribution
    if "answer_quality" in df.columns:
        lines.append("\n## Answer Quality Distribution\n")
        counts = df["answer_quality"].value_counts()
        for label in ["good", "questionable", "poor"]:
            c = counts.get(label, 0)
            pct = c / len(df) * 100
            lines.append(f"- **{label}**: {c} ({pct:.1f}%)")

    # v2: failure stage breakdown
    if "failure_stage" in df.columns:
        lines.append("\n## Failure Stage Breakdown\n")
        stage_counts = df["failure_stage"].value_counts()
        for stage, count in stage_counts.items():
            pct = count / len(df) * 100
            lines.append(f"- **{stage}**: {count} ({pct:.1f}%)")

    # Recommendation
    lines.append("\n## Recommendation\n")
    lines.append("*Review the correlation matrix and gold metric correlations above.*")
    lines.append("BERTScore (semantic) is more reliable than word-overlap F1 for generated text.")
    lines.append("Pick the cheapest judge with high BERTScore correlation for Experiments 1 & 2.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Experiment 0 pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_answers_path = output_dir / "raw_answers.csv"
    raw_scores_path = output_dir / "raw_scores.csv"
    report_path = output_dir / "report.md"

    # Parse difficulty filter
    allowed_difficulties = set(d.strip() for d in args.difficulty.split(",") if d.strip())
    if not allowed_difficulties:
        print("ERROR: --difficulty must specify at least one difficulty level.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment 0: Scorer Validation (v2)")
    print("=" * 60)
    print(f"  HotpotQA examples: {args.n}")
    print(f"  Difficulty filter: {', '.join(sorted(allowed_difficulties))}")
    print(f"  Generation model:  {args.model}")
    print(f"  Reranker:          {args.reranker} (top_k={args.reranker_top_k})")
    print(f"  Retrieval top_k:   {args.retrieval_top_k}")
    print(f"  Seed:              {args.seed}")
    print(f"  Output:            {output_dir}")
    print(f"  Skip generation:   {args.skip_generation}")
    print(f"  Max API cost:      ${args.max_cost:.2f}")
    if args.judges:
        print(f"  Judge filter:      {', '.join(args.judges)}")
    print()

    # Step 1: Load and filter HotpotQA
    if not args.skip_generation:
        logger.info("Loading HotpotQA (seed=%d)...", args.seed)
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa

        docs, queries = load_hotpotqa(split="train")

        # Filter by difficulty before sampling
        filtered_docs, filtered_queries = [], []
        for d, q in zip(docs, queries):
            if q.metadata.get("difficulty") in allowed_difficulties:
                filtered_docs.append(d)
                filtered_queries.append(q)

        if not filtered_docs:
            print(f"ERROR: No questions match difficulty filter {allowed_difficulties}.")
            print(f"Available difficulties: {set(q.metadata.get('difficulty') for q in queries)}")
            sys.exit(1)

        logger.info(
            "Filtered to %d %s questions (from %d total).",
            len(filtered_docs), "+".join(sorted(allowed_difficulties)), len(docs),
        )

        # Sample from filtered set
        docs, queries = sample_hotpotqa(
            filtered_docs, filtered_queries, n=args.n, seed=args.seed,
        )
        logger.info("Sampled %d examples.", len(docs))

        # Step 2: Check Ollama is running
        try:
            from ollama import Client
            client = Client(host=args.ollama_host) if args.ollama_host else Client()
            client.list()
        except Exception as exc:
            print(f"\nERROR: Cannot connect to Ollama: {exc}")
            print("Please start Ollama and try again.")
            sys.exit(1)

        # Step 3: Set up pipeline components
        from scripts.experiment_utils import generate_answer
        from src.strategies.naive import NaiveRAG
        from src.llms import OllamaLLM
        from src.chunkers.recursive import RecursiveChunker
        from src.embedders import OllamaEmbedder

        llm = OllamaLLM(host=args.ollama_host)
        strategy = NaiveRAG(llm=llm)
        chunker = RecursiveChunker(500, 100)
        embedder = OllamaEmbedder(host=args.ollama_host)
        reranker = _build_reranker(args.reranker)

        reranker_model_name = args.reranker if args.reranker != "none" else None

        logger.info("Generating answers with NaiveRAG + %s (reranker=%s)...",
                     args.model, args.reranker)

        answers = []
        total = len(docs)
        for i, (doc, query) in enumerate(zip(docs, queries)):
            logger.info("[%d/%d] Generating answer for: %s",
                        i + 1, total, query.text[:60])

            result = generate_answer(
                strategy=strategy,
                chunker=chunker,
                embedder=embedder,
                retrieval_mode="hybrid",
                query=query,
                doc=doc,
                model=args.model,
                ollama_host=args.ollama_host,
                reranker=reranker,
                reranker_top_k=args.reranker_top_k,
            )

            answers.append({
                "example_id": i,
                "question": query.text,
                "gold_answer": query.reference_answer or "",
                "rag_answer": result["answer"],
                "doc_text": doc.text,
                # v2 metadata
                "difficulty": query.metadata.get("difficulty", ""),
                "question_type": query.metadata.get("question_type", ""),
                # v2 diagnostics
                "strategy_latency_ms": result.get("strategy_latency_ms"),
                "num_chunks": result.get("num_chunks"),
                "num_chunks_retrieved": result.get("num_chunks_retrieved"),
                "context_char_length": result.get("context_char_length"),
                "context_sent_to_llm": result.get("context_sent_to_llm", ""),
                "failure_stage": result.get("failure_stage"),
                "failure_stage_confidence": result.get("failure_stage_confidence"),
                "failure_stage_method": result.get("failure_stage_method"),
                "gold_in_chunks": result.get("gold_in_chunks"),
                "gold_in_retrieved": result.get("gold_in_retrieved"),
                "gold_in_context": result.get("gold_in_context"),
                # v2 reranker info
                "reranker_model": reranker_model_name,
                "reranker_top_k": args.reranker_top_k if reranker_model_name else None,
                "retrieval_top_k": args.retrieval_top_k,
            })

        # Save raw answers for --skip-generation reruns
        answers_df = pd.DataFrame(answers)
        answers_df.to_csv(raw_answers_path, index=False)
        logger.info("Saved raw answers to %s", raw_answers_path)

        # Early exit for --generation-only: answers saved, skip scoring phase.
        # This allows terminating a GPU pod before the scoring phase that only
        # needs cloud APIs (no local GPU). Re-run with --skip-generation to score.
        if args.generation_only:
            print(f"\n--generation-only: answers saved to {raw_answers_path}")
            print(f"Run again with --skip-generation to score.")
            return
    else:
        # Load previously generated answers
        if not raw_answers_path.exists():
            print(f"\nERROR: {raw_answers_path} not found. Run without --skip-generation first.")
            sys.exit(1)
        logger.info("Loading previously generated answers from %s", raw_answers_path)
        answers_df = pd.read_csv(raw_answers_path)
        answers = answers_df.to_dict("records")
        logger.info("Loaded %d answers.", len(answers))

    # Step 4: Score all answers with judges (with cost guard)
    from src.cost_guard import CostGuard, CostLimitExceeded

    cost_guard = CostGuard(max_cost_usd=args.max_cost)
    logger.info("Scoring with judges (cost limit: $%.2f)...", args.max_cost)

    cost_limit_hit = False
    try:
        results_df = score_all_answers(
            answers, output_dir, cost_guard=cost_guard,
            judge_filters=args.judges,
        )
    except CostLimitExceeded as exc:
        logger.error("COST LIMIT REACHED: %s", exc)
        logger.error("Saving partial results...")
        cost_limit_hit = True
        results_df = pd.DataFrame(answers)

    # Merge new scores into existing CSV if it exists (preserves prior judge columns)
    if raw_scores_path.exists():
        existing_df = pd.read_csv(raw_scores_path)
        base_cols = {
            "example_id", "question", "gold_answer", "rag_answer",
            "gold_exact_match", "gold_f1", "gold_bertscore",
            # Pipeline metadata columns
            "chunk_type", "chunk_size", "chunk_overlap", "num_chunks",
            "embed_provider", "embed_model", "embed_dimension",
            "retrieval_mode", "retrieval_top_k", "num_chunks_retrieved",
            "context_char_length", "reranker_model", "reranker_top_k",
            "llm_provider", "llm_host", "dataset_name", "dataset_sample_seed",
            # v2 columns
            "difficulty", "question_type",
            "strategy_latency_ms", "failure_stage", "failure_stage_confidence",
            "failure_stage_method", "context_sent_to_llm",
            "gold_in_chunks", "gold_in_retrieved", "gold_in_context",
            "answer_quality",
        }
        new_scorer_cols = [c for c in results_df.columns
                          if c not in base_cols and c not in existing_df.columns]
        if new_scorer_cols:
            logger.info("Merging new judge columns into existing CSV: %s",
                        ", ".join(new_scorer_cols))
            merge_cols = ["example_id"] + new_scorer_cols
            existing_df = existing_df.merge(
                results_df[merge_cols], on="example_id", how="left",
            )
            results_df = existing_df

    # Step 5: Add answer_quality column (requires Sonnet + gold metrics)
    results_df = add_answer_quality(results_df)

    # Save raw scores (full or partial)
    results_df.to_csv(raw_scores_path, index=False)
    logger.info("Saved raw scores to %s", raw_scores_path)

    # Clean up checkpoint now that final CSV is written
    checkpoint_path = output_dir / "raw_scores_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file (final CSV written successfully).")

    # Step 6: Generate and save report
    scorers_used = []
    for config in JUDGE_CONFIGS:
        name = f"{config['provider']}:{config['model']}"
        safe = _safe_scorer_name(name)
        if f"{safe}_quality" in results_df.columns:
            scorers_used.append(name)

    report = generate_report(results_df, scorers_used)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    # Print report to stdout
    print("\n" + report)

    # Print cost summary
    print(f"\nAPI cost summary: {cost_guard.summary()} (limit: ${args.max_cost:.2f})")
    if cost_limit_hit:
        print("WARNING: Cost limit was reached — results are partial.")

    print("\n" + "=" * 60)
    print("Experiment 0 (v2) complete.")
    print(f"  Raw scores: {raw_scores_path}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    # Auto-regenerate gallery unless --no-gallery is set
    if not args.no_gallery:
        try:
            from scripts.generate_gallery import main as generate_gallery
            print("\nRegenerating gallery...")
            generate_gallery(experiments=[0])
            print("Gallery updated in site/")
        except Exception as exc:
            print(f"Gallery regeneration failed: {exc}")
            logger.warning("Gallery regeneration failed: %s", exc)


if __name__ == "__main__":
    main()
