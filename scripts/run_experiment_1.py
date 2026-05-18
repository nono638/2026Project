"""Experiment 1: Strategy x Model Size — 5 strategies x 6 models on 200 HotpotQA.

This is the project's core research question: does a smart RAG strategy on a
small model beat a naive strategy on a large model?

Matrix: 5 strategies (NaiveRAG, SelfRAG, MultiQueryRAG, CorrectiveRAG,
AdaptiveRAG) x 6 models from current open-weight families
(Qwen 3.5 small: 0.8b/2b/4b/9b; Gemma 4 e-tier: e2b/e4b) = 30 configurations.

Why this model set (2026-05-13 redesign): the original 10-model matrix
(adding qwen3.6:27b/35b-a3b and gemma4:26b/31b) was infeasible inside a
24 GB VRAM single-GPU budget and 48 h deadline. Drop the four largest
models, keep the full small-Qwen size curve (0.8b/2b/4b/9b) and two
size-matched Gemma 4 cross-family points (e2b ↔ qwen3.5:2b, e4b ↔ qwen3.5:4b).
The dropped models remain available via --models for future runs.

Held constant: RecursiveChunker(500, 100), OllamaEmbedder(embeddinggemma:300m),
hybrid retrieval, retrieval_top_k=5, no reranker. Embedder switched from
qwen3-embedding:4b → embeddinggemma:300m on 2026-05-13 (project-wide; see
docs/methodology.html).

Scorer: two-judge panel (Claude Haiku + GPT-5.4 mini), validated in Exp 0.

Checkpoint/resume: after each (strategy, model) config completes, rows are
flushed to raw_scores.csv. On restart with --resume, completed configs are
skipped. This is critical — 6,000 runs will take hours.

Usage:
    python scripts/run_experiment_1.py                              # full run
    python scripts/run_experiment_1.py --resume                     # resume interrupted
    python scripts/run_experiment_1.py --models qwen3.5:4b --strategies naive  # subset
    python scripts/run_experiment_1.py --skip-generation            # re-score only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
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
    score_answer_multi,
    _safe_scorer_name,
    load_checkpoint,
    append_rows,
    format_duration,
    build_scorer,
    get_ollama_model_details,
    write_experiment_metadata,
)

from src.monitoring import EventLog, call_tracker, snapshot as gpu_snapshot

# How long the GPU/driver is given to settle between configs. The 5090
# Laptop has been observed to BSOD with nvlddmkm DPC_WATCHDOG_VIOLATION
# (0x133) under sustained load — a known driver bug across multiple
# versions, see https://forums.developer.nvidia.com/t/bug-rtx-5090-hibernate-resume-causes-nvlddmkm-sys-0x133-dpc-watchdog/364994
# 2026-05-18: bumped 5s → 30s after a second BSOD (0x3b SYSTEM_SERVICE_EXCEPTION)
# during a resumed run. The clean driver re-install didn't prevent it, so we
# lean harder on workload pacing.
CONFIG_COOLDOWN_S = 30

# Per-row pacing — inserted between every completed (generate + score) row
# inside a config. The pre-fortification observed rate was ~1 query / 5-7 s
# with embed/chat calls firing every ~1 s; that's the sustained-DPC regime
# that correlates with the 5090 driver bug. A small fixed sleep stretches
# the inter-call cadence enough to let the kernel drain its DPC queue
# between calls without meaningfully slowing the overall run.
ROW_PACE_S_DEFAULT = 0.5

# Periodic longer rest mid-config to let GPU temperature drop. The
# pre-fortification trail showed temp climbing 47 °C → 66 °C over ~30 min
# right before BSOD #2. Every N rows we pause for a longer beat.
ROW_REST_EVERY_N = 50
ROW_REST_S = 10

# Sample GPU stats every N completed rows during a config. After the
# 2026-05-18 BSOD cluster (4 distinct bugchecks in 12h), 50-row spacing
# left the post-mortem with a ~25-minute uncertainty window. Tightening
# to 10 rows shrinks that to ~5 min at ~30s/query. Cost: roughly +1s/config
# of nvidia-smi overhead — negligible against ~25 min config wallclock.
GPU_SNAPSHOT_EVERY_N_ROWS = 10

# Default judge panel for Experiments 1 and 2: cross-provider two-judge
# panel chosen from Exp 0 v3 cost vs accuracy results. Documented in
# docs/methodology.html "Scorer Selection".
DEFAULT_SCORER_PANEL = [
    "anthropic:claude-haiku-4-5-20251001",
    "openai:gpt-5.4-mini",
]


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

# 2026-05-13 redesign: trimmed from 10 models to 6 to fit a 24 GB VRAM single-GPU
# budget and 48 h deadline. The dropped 4 (qwen3.6:27b, qwen3.6:35b-a3b,
# gemma4:26b, gemma4:31b) remain pull-able via `ollama pull` and runnable via
# `--models <name>`; they're just not in the held-constant matrix anymore.
#
# task-053: bare tags retained — explicit -q4_K_M pinning per the spec's
# verification table couldn't be confirmed against ollama.com from the
# nighttime sandbox. The runtime helper get_ollama_model_details() stamps
# the resolved quantization on every row in the metadata + CSV, so analysis
# can audit which artifact actually loaded regardless of tag form.
ALL_MODELS = [
    # Qwen 3.5 small (Apache 2.0, released 2026-03-02) — full small-tier curve.
    # Announcement: https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    # Gemma 4 (Apache 2.0, released 2026-04-02) — cross-family small tier.
    # Announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
    "gemma4:e2b",           # 2.3B effective — pairs with qwen3.5:2b
    "gemma4:e4b",           # 4.5B effective — pairs with qwen3.5:4b
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
    # Pre-flight: reject the removed singular `--scorer` flag explicitly
    # before argparse runs, so `--scorer X --help` exits non-zero (argparse
    # would otherwise let `--help` win and exit 0). See task-052 spec.
    if "--scorer" in sys.argv:
        print(
            "error: unrecognized argument '--scorer' (renamed to '--scorers' in task-052)",
            file=sys.stderr,
        )
        sys.exit(2)
    parser = argparse.ArgumentParser(
        description="Experiment 1: Strategy x Model Size — 5 strategies x 6 models on HotpotQA.",
        # Disable prefix matching so `--scorer` cannot abbreviate to `--scorers`.
        allow_abbrev=False,
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of HotpotQA examples (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results/experiment_1",
                        help="Output directory (default: results/experiment_1)")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama server URL (default: localhost:11434)")
    parser.add_argument("--embedder", type=str, default="embeddinggemma:300m",
                        help="Ollama embedding model tag for retrieval (default: "
                             "embeddinggemma:300m, the project-wide held-constant "
                             "embedder since 2026-05-13). Override only for embedder "
                             "A/B experiments; held-constant runs should leave this.")
    parser.add_argument("--embedder-max-chars", type=int, default=None,
                        help="Override OllamaEmbedder client-side input cap. "
                             "Default uses OllamaEmbedder.DEFAULT_MAX_CHARS (7000). "
                             "Lower for embedders with small context windows.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in raw_scores.csv")
    parser.add_argument("--max-cost", type=float, default=10.0,
                        help="Maximum estimated API spend in USD (default: $10.00)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model subset (e.g., 'qwen3.5:4b,gemma4:e2b')")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy subset (e.g., 'naive,self_rag')")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Re-score existing answers without re-generating")
    parser.add_argument(
        "--scorers",
        type=str,
        nargs="+",
        default=list(DEFAULT_SCORER_PANEL),
        help=(
            "Judge panel as one or more provider:model strings "
            "(default: anthropic:claude-haiku-4-5-20251001 openai:gpt-5.4-mini)."
        ),
    )
    parser.add_argument("--no-gallery", action="store_true",
                        help="Skip automatic gallery regeneration after experiment completes")
    parser.add_argument(
        "--row-pace-s",
        type=float,
        default=ROW_PACE_S_DEFAULT,
        help=(
            "Seconds to sleep between completed rows inside a config. "
            "5090-stability knob — increase if BSODs recur, set to 0 to "
            f"disable. Default {ROW_PACE_S_DEFAULT}s."
        ),
    )
    args = parser.parse_args()
    # Reject duplicate scorers — a doubled judge would silently double-cost
    # and produce one column representing two runs averaged. See spec.
    if len(set(args.scorers)) != len(args.scorers):
        parser.error(
            "Duplicate scorers in --scorers are not allowed; pass unique provider:model entries."
        )
    return args


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
    if "consensus_quality" in df.columns and "strategy" in df.columns and "model" in df.columns:
        pivot = df.pivot_table(
            values="consensus_quality", index="strategy", columns="model", aggfunc="mean",
        )
        # Round for readability
        pivot = pivot.round(3)
        lines.append(pivot.to_markdown())
        lines.append("")
    else:
        lines.append("*Missing required columns for heatmap.*\n")

    # --- Per-strategy ranking ---
    lines.append("## Per-Strategy Ranking\n")
    if "consensus_quality" in df.columns:
        strat_stats = df.groupby("strategy")["consensus_quality"].agg(["mean", "std", "count"])
        strat_stats = strat_stats.sort_values("mean", ascending=False).round(3)
        lines.append(strat_stats.to_markdown())
        lines.append("")

    # --- Per-model ranking ---
    lines.append("## Per-Model Ranking\n")
    if "consensus_quality" in df.columns:
        model_stats = df.groupby("model")["consensus_quality"].agg(["mean", "std", "count"])
        model_stats = model_stats.sort_values("mean", ascending=False).round(3)
        lines.append(model_stats.to_markdown())
        lines.append("")

    # --- Per-judge agreement (Pearson r between each judge pair) ---
    judge_quality_cols = [
        c for c in df.columns
        if c.endswith("_quality") and c not in ("consensus_quality", "answer_quality")
    ]
    if len(judge_quality_cols) >= 2:
        lines.append("## Per-Judge Agreement\n")
        lines.append("Pearson correlation of per-row quality between each judge pair:\n")
        lines.append("| Judge A | Judge B | Pearson r | n |")
        lines.append("|---------|---------|-----------|---|")
        for i in range(len(judge_quality_cols)):
            for j in range(i + 1, len(judge_quality_cols)):
                a_col, b_col = judge_quality_cols[i], judge_quality_cols[j]
                pair = df[[a_col, b_col]].dropna()
                if len(pair) >= 2:
                    r = pair[a_col].corr(pair[b_col])
                    a_short = a_col.replace("_quality", "")
                    b_short = b_col.replace("_quality", "")
                    lines.append(f"| {a_short} | {b_short} | {r:.3f} | {len(pair)} |")
        lines.append("")

    # --- Strategy beats size analysis ---
    # Cases where small_model + smart_strategy > large_model + naive
    lines.append("## Strategy Beats Size Analysis\n")
    lines.append("Cases where a smaller model with a non-naive strategy outperforms "
                 "a larger model with NaiveRAG:\n")

    if "consensus_quality" in df.columns and "strategy" in df.columns and "model" in df.columns:
        config_means = df.groupby(["strategy", "model"])["consensus_quality"].mean()

        # Model sizes for ordering (approximate parameter counts)
        # Approximate parameter counts (effective for MoE) for size ordering.
        # See ALL_MODELS for source links.
        model_sizes = {
            "qwen3.5:0.8b": 0.8,
            "qwen3.5:2b": 2.0,
            "qwen3.5:4b": 4.0,
            "qwen3.5:9b": 9.0,
            "qwen3.6:27b": 27.0,
            "qwen3.6:35b-a3b": 3.0,    # ~3B active params at inference
            "gemma4:e2b": 2.3,
            "gemma4:e4b": 4.5,
            "gemma4:26b": 3.8,         # ~3.8B active params at inference
            "gemma4:31b": 31.0,
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
    n_scored = (
        len(df.dropna(subset=["consensus_quality"]))
        if "consensus_quality" in df.columns else 0
    )
    lines.append(f"- Total scored answers (any judge non-NaN): {n_scored}")
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

    # Structured event log — written alongside raw_scores.csv. The plaintext
    # run log is dominated by per-HTTP-request lines (useful for live tail,
    # near-useless for post-mortem); events.jsonl carries the run skeleton:
    # run_start, config_start, gpu_snapshot, config_end, error, run_end.
    events = EventLog(output_dir / "events.jsonl")

    # Signal handlers — give the run a chance to mark a clean shutdown line
    # in events.jsonl before the process exits. This won't help against a
    # full BSOD (the kernel never lets us run), but it does turn Ctrl-C
    # and any orderly SIGTERM into a recoverable, labeled stop.
    def _on_signal(signum, _frame):
        events.write("signal_received",
                     signal=signal.Signals(signum).name)
        # Restore default and re-raise so the standard exit path runs.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError):
            # SIGTERM isn't always available on Windows in some contexts;
            # missing a signal handler is non-fatal — telemetry must not
            # block startup.
            pass

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
    print(f"  Scorer panel:     {', '.join(args.scorers)}")
    print(f"  Max API cost:     ${args.max_cost:.2f}")
    print(f"  Output:           {output_dir}")
    print(f"  Resume:           {args.resume}")
    print(f"  Skip generation:  {args.skip_generation}")
    if args.ollama_host:
        print(f"  Ollama host:      {args.ollama_host}")
    print()

    # Check for resume — row-level so a partial config (interrupted mid-run
    # by power loss / restart) picks up exactly where it left off instead of
    # being re-run from question 1.
    completed_rows: set[tuple] = set()
    if args.resume:
        completed_rows = load_checkpoint(
            raw_scores_path, key_cols=("strategy", "model", "question"),
        )
        if completed_rows:
            by_config: dict[tuple[str, str], int] = {}
            for s, m, _q in completed_rows:
                by_config[(s, m)] = by_config.get((s, m), 0) + 1
            logger.info("Resuming — %d rows across %d configs already done.",
                        len(completed_rows), len(by_config))
            for (s, m), n in sorted(by_config.items()):
                logger.info("  %s x %s: %d rows", s, m, n)

    # Emit the run_start event with the initial GPU snapshot. The first
    # snapshot anchors the post-mortem timeline: if a later crash dump
    # shows GPU memory climbing or temps rising past this baseline,
    # that's diagnostic signal.
    events.write(
        "run_start",
        experiment="experiment_1",
        strategies=strategies,
        models=models,
        n_questions=args.n,
        seed=args.seed,
        scorers=list(args.scorers),
        max_cost_usd=args.max_cost,
        resume=args.resume,
        gpu=gpu_snapshot(),
    )

    # Build judge panel — one shared CostGuard so --max-cost is the global
    # ceiling across the panel, not a per-judge ceiling.
    from src.cost_guard import CostGuard, CostLimitExceeded

    shared_guard = CostGuard(max_cost_usd=args.max_cost)
    scorers = [build_scorer(s, cost_guard=shared_guard) for s in args.scorers]
    logger.info("Built %d-judge panel (shared cost limit: $%.2f)",
                len(scorers), args.max_cost)

    cost_limit_hit = False  # May be set True during generation; stays False for --skip-generation

    # task-053: populated below when not --skip-generation; stays empty so
    # the metadata write at the bottom can reference it unconditionally.
    model_details_by_tag: dict[str, dict] = {}

    if args.skip_generation:
        # Re-score existing answers
        if not raw_scores_path.exists():
            print(f"\nERROR: {raw_scores_path} not found. Run without --skip-generation first.")
            sys.exit(1)
        logger.info("Loading existing answers for re-scoring...")
        existing_df = pd.read_csv(raw_scores_path)

        # Re-score each row using context_sent_to_llm (what the model saw).
        # Per-judge resume: rows that already have non-NaN values for a
        # judge keep them; new judges are scored from scratch.
        logger.info("Re-scoring %d answers across %d judges...",
                    len(existing_df), len(scorers))
        for idx, row in existing_df.iterrows():
            try:
                scores = score_answer_multi(
                    scorers,
                    row["question"],
                    row.get("context_sent_to_llm", ""),
                    row["rag_answer"],
                    existing_row=row.to_dict(),
                )
            except CostLimitExceeded as exc:
                logger.error("\nCOST LIMIT REACHED during re-score: %s", exc)
                cost_limit_hit = True
                break
            for k, v in scores.items():
                existing_df.at[idx, k] = v

        # Drop legacy bare metric columns from prior single-judge runs so
        # downstream code can't silently treat them as a primary judge.
        for legacy in ("faithfulness", "relevance", "conciseness", "quality",
                       "scorer_latency_ms"):
            if legacy in existing_df.columns:
                existing_df = existing_df.drop(columns=[legacy])

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
        embedder = OllamaEmbedder(
            model=args.embedder,
            host=args.ollama_host,
            max_chars=args.embedder_max_chars,
        )
        logger.info(
            "Embedder: %s (max_chars=%d, dim=%d)",
            embedder.name, embedder._max_chars, embedder.dimension,
        )

        # task-053: capture per-tag Ollama provenance once at startup so
        # every row gets a llm_quantization stamp without re-querying.
        model_details_by_tag = {
            m: get_ollama_model_details(m, host=args.ollama_host) for m in models
        }
        for tag, det in model_details_by_tag.items():
            logger.info(
                "Ollama details for %s: quant=%s digest=%s",
                tag, det.get("quantization_level"), det.get("digest"),
            )

        # Build the config matrix.
        # Model is the OUTER loop so each model loads into VRAM exactly once
        # and runs through every strategy before the next model takes over.
        # Reversing this (strategy outer, model inner) made models reload
        # len(strategies) times each on the 5090 — observed 2026-05-06.
        config_list = [
            (strat_name, model_name)
            for model_name in models
            for strat_name in strategies
        ]
        total_configs = len(config_list)
        configs_done = 0
        experiment_start = time.perf_counter()
        cost_limit_hit = False

        for config_idx, (strat_name, model_name) in enumerate(config_list, 1):
            done_questions_this_config = {
                q for (s, m, q) in completed_rows
                if s == strat_name and m == model_name
            }
            if len(done_questions_this_config) >= len(queries):
                logger.info("[config %d/%d] SKIPPING %s x %s (all %d rows already done)",
                            config_idx, total_configs, strat_name, model_name,
                            len(done_questions_this_config))
                configs_done += 1
                continue

            if done_questions_this_config:
                logger.info("[config %d/%d] %s x %s — resuming, %d/%d rows already done",
                            config_idx, total_configs, strat_name, model_name,
                            len(done_questions_this_config), len(queries))
            else:
                logger.info("[config %d/%d] %s x %s", config_idx, total_configs,
                            strat_name, model_name)

            events.write(
                "config_start",
                config_idx=config_idx,
                total_configs=total_configs,
                strategy=strat_name,
                model=model_name,
                rows_already_done=len(done_questions_this_config),
                gpu=gpu_snapshot(),
            )

            # Ensure model is available
            try:
                ensure_model(client, model_name)
            except Exception as exc:
                logger.error("Failed to pull model %s: %s — skipping model", model_name, exc)
                events.write("model_pull_failed", model=model_name, error=str(exc))
                continue

            # Build strategy with fresh LLM instance
            llm = OllamaLLM(host=args.ollama_host)
            strategy = _make_strategy(strat_name, llm)

            # Run all queries for this config
            config_rows_written = 0
            config_start = time.perf_counter()

            for q_idx, (doc, query) in enumerate(zip(docs, queries)):
                if query.text in done_questions_this_config:
                    continue

                row_start = time.perf_counter()

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
                    model_details=model_details_by_tag.get(model_name),
                )

                # Score answer — use context_sent_to_llm so faithfulness is
                # judged against what the model actually saw (Exp 0 v2 fix).
                # score_answer_multi catches per-judge exceptions (including
                # CostLimitExceeded) and records NaN; we still detect cost
                # limit via the shared guard's running total.
                scorer_context = result.get("context_sent_to_llm", "")
                scores = score_answer_multi(
                    scorers, query.text, scorer_context, result["answer"],
                )
                if shared_guard.total_estimated_cost > args.max_cost and not cost_limit_hit:
                    logger.error("\nCOST LIMIT REACHED: %s", shared_guard.summary())
                    logger.error("Saving partial results for this config...")
                    cost_limit_hit = True

                # Total scorer latency = sum across judges in panel
                total_scorer_latency = 0.0
                for s in scorers:
                    safe = _safe_scorer_name(s.name)
                    lat = scores.get(f"{safe}_scorer_latency_ms", 0.0)
                    if isinstance(lat, (int, float)) and not math.isnan(lat):
                        total_scorer_latency += float(lat)

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
                    # Per-judge scorer metrics + consensus_quality
                    **scores,
                    # Latency
                    "strategy_latency_ms": result.get("strategy_latency_ms", float("nan")),
                    "scorer_latency_ms_total": total_scorer_latency,
                    "total_latency_ms": (
                        result.get("strategy_latency_ms", 0) + total_scorer_latency
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
                    "embed_model": embedder.name.split(":", 1)[1],
                    "embed_dimension": embedder.dimension,
                    "retrieval_mode": "hybrid",
                    "retrieval_top_k": 5,
                    "num_chunks_retrieved": result.get("num_chunks_retrieved", 0),
                    "context_char_length": result.get("context_char_length", 0),
                    "reranker_model": None,
                    "reranker_top_k": None,
                    "llm_provider": "ollama",
                    "llm_host": args.ollama_host or "local",
                    "llm_model": model_name,
                    "llm_quantization": result.get("llm_quantization", "unknown"),
                    "dataset_name": "hotpotqa",
                    "dataset_sample_seed": args.seed,
                }
                # Per-row checkpoint: flush+fsync immediately so a power loss
                # mid-config loses at most this single row, not the whole
                # ~25-min config. Resume picks up by question text.
                append_rows(raw_scores_path, [row])
                completed_rows.add((strat_name, model_name, query.text))
                config_rows_written += 1

                # Per-row heartbeat — single fsynced line so the post-mortem
                # of any BSOD can identify which row had just completed and
                # what call (embed/chat) was most recently in flight. Cheap
                # (no nvidia-smi shell-out here — that stays on the every-N
                # snapshot below). Shrinks the BSOD blast radius from one
                # config-snapshot-interval to one row.
                events.write(
                    "row_done",
                    config_idx=config_idx,
                    strategy=strat_name,
                    model=model_name,
                    row_idx_config=config_rows_written,
                    q_idx=q_idx + 1,
                    row_duration_s=round(time.perf_counter() - row_start, 3),
                    call=call_tracker.snapshot(),
                )

                # Periodic GPU snapshot. Anchors the timeline inside long
                # configs so a crash dump can be matched to the workload
                # phase that preceded it.
                if config_rows_written % GPU_SNAPSHOT_EVERY_N_ROWS == 0:
                    events.write(
                        "gpu_snapshot",
                        config_idx=config_idx,
                        strategy=strat_name,
                        model=model_name,
                        rows_written_this_config=config_rows_written,
                        gpu=gpu_snapshot(),
                    )

                if cost_limit_hit:
                    break

                # 5090-stability pacing: short sleep between rows, longer
                # periodic rest. See ROW_PACE_S_DEFAULT / ROW_REST_EVERY_N
                # constants for rationale.
                if (
                    ROW_REST_EVERY_N > 0
                    and config_rows_written % ROW_REST_EVERY_N == 0
                ):
                    events.write(
                        "row_rest",
                        config_idx=config_idx,
                        rows_written_this_config=config_rows_written,
                        rest_s=ROW_REST_S,
                    )
                    time.sleep(ROW_REST_S)
                elif args.row_pace_s > 0:
                    time.sleep(args.row_pace_s)

            print()  # newline after progress display
            config_elapsed = time.perf_counter() - config_start
            logger.info("Config %s x %s done: %d new rows in %s",
                        strat_name, model_name, config_rows_written,
                        format_duration(config_elapsed))
            events.write(
                "config_end",
                config_idx=config_idx,
                strategy=strat_name,
                model=model_name,
                rows_written=config_rows_written,
                elapsed_s=round(config_elapsed, 2),
                gpu=gpu_snapshot(),
            )
            configs_done += 1

            if cost_limit_hit:
                logger.error("Stopping experiment due to cost limit.")
                break

            # Cooldown between configs — gives the NVIDIA driver a moment
            # to clean up before the next model swap. Cheap (a few seconds
            # over a multi-hour run) and labelled in events.jsonl for
            # post-mortem.
            if config_idx < total_configs:
                time.sleep(CONFIG_COOLDOWN_S)

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

        # Write metadata sidecar — judge model versions + run config.
        # Mirrors run_experiment_0.write_metadata: list every judge in the
        # active panel whose <safe>_quality column is present in the CSV.
        from scripts.generate_experiment0_dashboard import JUDGE_DISPLAY_NAMES
        judges_in_data = []
        for s in scorers:
            provider, model = s.name.split(":", 1)
            safe = _safe_scorer_name(s.name)
            q_col = f"{safe}_quality"
            if q_col not in results_df.columns:
                continue
            judges_in_data.append({
                "provider": provider,
                "model": model,
                "display_name": JUDGE_DISPLAY_NAMES.get(safe, model),
                "n_scored": int(results_df[q_col].notna().sum()),
            })
        write_experiment_metadata(
            output_dir=output_dir,
            n_examples=len(results_df),
            config={
                "n_questions": args.n,
                "seed": args.seed,
                "models": models,
                "strategies": strategies,
                "ollama_host": args.ollama_host,
                "skip_generation": args.skip_generation,
                "scorers": list(args.scorers),
            },
            judges=judges_in_data,
            model_details=model_details_by_tag or None,
            extra={"experiment_axis": "strategy_x_model_size"},
        )
    else:
        logger.warning("No results file found — skipping report.")

    print("\n" + "=" * 60)
    print("Experiment 1 complete.")
    print(f"  Raw scores: {raw_scores_path}")
    print(f"  Report:     {report_path}")
    if cost_limit_hit:
        print("  WARNING: Cost limit was reached — results are partial.")
    print("=" * 60)

    events.write(
        "run_end",
        cost_limit_hit=cost_limit_hit,
        gpu=gpu_snapshot(),
    )

    # Auto-regenerate gallery unless --no-gallery is set
    if not args.no_gallery:
        try:
            # Lazy import to avoid breaking experiment if gallery deps are missing
            from scripts.generate_gallery import main as generate_gallery
            print("\nRegenerating gallery...")
            generate_gallery(experiments=[1])
            print("Gallery updated in docs/")
        except Exception as exc:
            print(f"Gallery regeneration failed: {exc}")
            logger.warning("Gallery regeneration failed: %s", exc)


if __name__ == "__main__":
    main()
