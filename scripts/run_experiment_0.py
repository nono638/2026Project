"""Experiment 0: Scorer Validation — compare up to 6 LLM judges (4 Gemini + 2 Claude).

Generates 50 RAG answers using NaiveRAG + Qwen3 4B on HotpotQA, then scores
each answer with up to 6 LLM judges. Gemini judges run via free Google AI Studio;
Anthropic judges are optional and skipped if ANTHROPIC_API_KEY is not set.

Judges (in order):
  - gemini-2.5-flash-lite  (cheapest baseline)
  - gemini-2.5-flash
  - gemini-2.5-pro
  - claude-haiku-4-5        (optional)
  - claude-sonnet-4         (optional)

Produces a comparison report showing:
- Per-judge mean scores
- Inter-scorer correlation matrix (Pearson on quality)
- Each judge's correlation with gold F1
- Cost breakdown

This is the methodological safety net — it tells us whether cheap scorers
(Gemini Flash-Lite, Flash) give meaningfully different results than expensive
ones (Gemini Pro, Claude).

Usage:
    python scripts/run_experiment_0.py                          # full run
    python scripts/run_experiment_0.py --n 10 --model qwen3:0.6b  # quick test
    python scripts/run_experiment_0.py --skip-generation        # re-score only
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
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


# ---------------------------------------------------------------------------
# Scorer configurations — 5 LLM judges
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
# Utility functions — pure, no side effects
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, gold: str) -> float:
    """Word-level F1 between prediction and gold answer.

    Uses set-based token overlap — simple but effective for short answers.

    Args:
        prediction: The RAG-generated answer.
        gold: The gold reference answer.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    """Case-insensitive check: does prediction contain the gold answer?

    Uses string containment (not equality) because RAG answers are usually
    longer than the gold answer. "The capital is Paris" should match gold
    "Paris". Strict equality would penalize verbose-but-correct answers.

    Args:
        prediction: The RAG-generated answer.
        gold: The gold reference answer.

    Returns:
        True if gold appears in prediction (case-insensitive).
    """
    return gold.lower() in prediction.lower()


def compute_bertscores(predictions: list[str], golds: list[str]) -> list[float]:
    """Compute BERTScore F1 between each prediction-gold pair.

    Uses RoBERTa-large, the standard BERTScore model for English. Runs
    locally — no API calls. The model (~1.4GB) downloads on first run.

    Token-level semantic matching is more appropriate than word-overlap F1
    for evaluating generated text against short gold answers, because it
    captures meaning rather than exact word choice.

    Reference: Zhang et al., "BERTScore: Evaluating Text Generation with
    BERT", ICLR 2020. https://arxiv.org/abs/1904.09675

    Args:
        predictions: List of RAG-generated answers.
        golds: List of gold reference answers.

    Returns:
        List of BERTScore F1 values (0.0 to 1.0), one per pair.
    """
    from bert_score import score

    _, _, f1 = score(
        cands=predictions,
        refs=golds,
        lang="en",
        verbose=True,
    )
    return f1.tolist()


def _safe_scorer_name(name: str) -> str:
    """Convert scorer name to a safe column prefix.

    Replaces colons, dashes, and dots with underscores for CSV column names.

    Args:
        name: The scorer's name property (e.g., "google:gemini-2.5-flash").

    Returns:
        Safe column prefix (e.g., "google_gemini_2_5_flash").
    """
    return name.replace(":", "_").replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 0: Scorer Validation — compare up to 6 LLM judges on HotpotQA.",
    )
    parser.add_argument("--n", type=int, default=50,
                        help="Number of HotpotQA examples (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--model", type=str, default="qwen3:4b",
                        help="Ollama model for answer generation (default: qwen3:4b)")
    parser.add_argument("--output-dir", type=str, default="results/experiment_0",
                        help="Output directory (default: results/experiment_0)")
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
    parser.add_argument("--no-gallery", action="store_true",
                        help="Skip automatic gallery regeneration after experiment completes")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answers(
    documents: list,
    queries: list,
    model: str,
    ollama_host: str | None = None,
) -> list[dict]:
    """Generate RAG answers for each (document, query) pair using NaiveRAG.

    Args:
        documents: List of Document objects from HotpotQA.
        queries: List of Query objects (parallel to documents).
        model: Ollama model name for generation.
        ollama_host: Ollama server URL, or None for localhost.

    Returns:
        List of dicts with example_id, question, gold_answer, rag_answer, doc_text.
    """
    from src.strategies.naive import NaiveRAG
    from src.llms import OllamaLLM
    from src.chunkers.recursive import RecursiveChunker
    from src.embedders import OllamaEmbedder
    from src.retriever import Retriever

    # Set up pipeline components — held constant across all examples
    # Pass host to both LLM and embedder for remote Ollama support
    llm = OllamaLLM(host=ollama_host)
    strategy = NaiveRAG(llm=llm)
    chunker = RecursiveChunker(500, 100)
    embedder = OllamaEmbedder(host=ollama_host)

    results = []
    total = len(documents)

    for i, (doc, query) in enumerate(zip(documents, queries)):
        logger.info("[%d/%d] Generating answer for: %s", i + 1, total, query.text[:60])

        try:
            # Build retriever for this document
            chunks = chunker.chunk(doc.text)
            retriever = Retriever(chunks, embedder)

            # Retrieve once for metadata
            retrieved = retriever.retrieve(query.text)

            # Generate answer
            answer = strategy.run(query.text, retriever, model)
        except Exception as exc:
            logger.error("Generation failed for example %d: %s", i, exc)
            answer = ""
            chunks = []
            retrieved = []

        results.append({
            "example_id": i,
            "question": query.text,
            "gold_answer": query.reference_answer or "",
            "rag_answer": answer,
            "doc_text": doc.text,
            "num_chunks": len(chunks),
            "num_chunks_retrieved": len(retrieved),
            "context_char_length": sum(len(r.get("text", "")) for r in retrieved),
        })

    return results


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

    Args:
        answers: List of dicts from generate_answers().
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

    # Build result rows
    rows = []
    total = len(answers)

    for i, ans in enumerate(answers):
        logger.info("[%d/%d] Scoring: %s", i + 1, total, ans["question"][:60])

        # Base columns
        row = {
            "example_id": ans["example_id"],
            "question": ans["question"],
            "gold_answer": ans["gold_answer"],
            "rag_answer": ans["rag_answer"],
            "gold_exact_match": exact_match(ans["rag_answer"], ans["gold_answer"]),
            "gold_f1": compute_f1(ans["rag_answer"], ans["gold_answer"]),
            # Pipeline metadata (constant for Exp 0)
            "chunk_type": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "num_chunks": ans.get("num_chunks"),
            "embed_provider": "ollama",
            "embed_model": "mxbai-embed-large",
            "embed_dimension": 1024,
            "retrieval_mode": "hybrid",
            "retrieval_top_k": 5,
            "num_chunks_retrieved": ans.get("num_chunks_retrieved"),
            "context_char_length": ans.get("context_char_length"),
            "reranker_model": None,
            "reranker_top_k": None,
            "llm_provider": "ollama",
            "llm_host": "local",
            "dataset_name": "hotpotqa",
            "dataset_sample_seed": 42,
        }

        # Score with each judge
        for scorer in scorers:
            safe_name = _safe_scorer_name(scorer.name)
            try:
                scores = scorer.score(
                    query=ans["question"],
                    context=ans["doc_text"],
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

    # Rough per-call cost estimates (input + output for ~500 token prompt)
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

    print("=" * 60)
    print("Experiment 0: Scorer Validation")
    print("=" * 60)
    print(f"  HotpotQA examples: {args.n}")
    print(f"  Generation model:  {args.model}")
    print(f"  Seed:              {args.seed}")
    print(f"  Output:            {output_dir}")
    print(f"  Skip generation:   {args.skip_generation}")
    print(f"  Max API cost:      ${args.max_cost:.2f}")
    if args.judges:
        print(f"  Judge filter:      {', '.join(args.judges)}")
    print()

    # Step 1: Load HotpotQA
    if not args.skip_generation:
        logger.info("Loading HotpotQA (n=%d, seed=%d)...", args.n, args.seed)
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa

        docs, queries = load_hotpotqa(split="train")
        docs, queries = sample_hotpotqa(docs, queries, n=args.n, seed=args.seed)
        logger.info("Loaded %d examples.", len(docs))

        # Step 2: Check Ollama is running (use remote host if specified)
        try:
            from ollama import Client
            client = Client(host=args.ollama_host) if args.ollama_host else Client()
            client.list()
        except Exception as exc:
            print(f"\nERROR: Cannot connect to Ollama: {exc}")
            print("Please start Ollama and try again.")
            sys.exit(1)

        # Step 3: Generate answers
        logger.info("Generating answers with NaiveRAG + %s...", args.model)
        answers = generate_answers(docs, queries, args.model, ollama_host=args.ollama_host)

        # Save raw answers for --skip-generation reruns
        answers_df = pd.DataFrame(answers)
        answers_df.to_csv(raw_answers_path, index=False)
        logger.info("Saved raw answers to %s", raw_answers_path)
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
        # Partial results — score_all_answers doesn't return on exception,
        # so we build a minimal DataFrame from the raw answers
        results_df = pd.DataFrame(answers)

    # Merge new scores into existing CSV if it exists (preserves prior judge columns)
    if raw_scores_path.exists():
        existing_df = pd.read_csv(raw_scores_path)
        # Find new scorer columns (not in existing)
        base_cols = {
            "example_id", "question", "gold_answer", "rag_answer",
            "gold_exact_match", "gold_f1", "gold_bertscore",
            # Pipeline metadata columns
            "chunk_type", "chunk_size", "chunk_overlap", "num_chunks",
            "embed_provider", "embed_model", "embed_dimension",
            "retrieval_mode", "retrieval_top_k", "num_chunks_retrieved",
            "context_char_length", "reranker_model", "reranker_top_k",
            "llm_provider", "llm_host", "dataset_name", "dataset_sample_seed",
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

    # Save raw scores (full or partial)
    results_df.to_csv(raw_scores_path, index=False)
    logger.info("Saved raw scores to %s", raw_scores_path)

    # Step 5: Generate and save report
    # Determine which scorers were actually used (have columns in the df)
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
    print("Experiment 0 complete.")
    print(f"  Raw scores: {raw_scores_path}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    # Auto-regenerate gallery unless --no-gallery is set
    if not args.no_gallery:
        try:
            # Lazy import to avoid breaking experiment if gallery deps are missing
            from scripts.generate_gallery import main as generate_gallery
            print("\nRegenerating gallery...")
            generate_gallery(experiments=[0])
            print("Gallery updated in site/")
        except Exception as exc:
            print(f"Gallery regeneration failed: {exc}")
            logger.warning("Gallery regeneration failed: %s", exc)


if __name__ == "__main__":
    main()
