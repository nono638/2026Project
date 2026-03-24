"""Shared experiment infrastructure for Experiments 1 and 2.

Provides checkpoint/resume, model pre-pulling, gold metrics, scoring, and
progress tracking. Extracted from run_experiment_0.py patterns so both
experiment scripts share identical logic for these concerns.

Why a shared module instead of duplicating across scripts: Both experiments
need identical checkpoint, model-pull, scoring, and gold-metric logic.
Duplicating ~200 lines across two scripts is worse than a small shared module.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.diagnostics import detect_failure_stage, _gold_in_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reranked retriever wrapper
# ---------------------------------------------------------------------------

class _RerankedRetriever:
    """Wraps a Retriever and a reranker to chain retrieval then reranking.

    Strategies call .retrieve() as normal and get reranked results. This avoids
    modifying the Retriever class itself, preventing double-reranking conflicts
    with the Experiment class which does its own reranking.

    Args:
        retriever: The underlying Retriever instance.
        reranker: A reranker with .rerank(query, chunks, top_k) method.
        top_k: Number of results to keep after reranking.
    """

    def __init__(self, retriever: object, reranker: object, top_k: int) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._top_k = top_k

    @property
    def chunks(self) -> list:
        """Proxy .chunks from the underlying retriever."""
        return self._retriever.chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve then rerank.

        Calls the underlying retriever with top_k=None (retrieve all it would
        normally return), then applies the reranker to select the best top_k.

        Args:
            query: The search query.
            top_k: Ignored — reranker_top_k from __init__ is used instead.

        Returns:
            Reranked list of chunk dicts.
        """
        # Retrieve full candidate set from underlying retriever
        candidates = self._retriever.retrieve(query, top_k=None)
        # Rerank down to top_k
        return self._reranker.rerank(query, candidates, self._top_k)


# ---------------------------------------------------------------------------
# Gold metrics — pure functions, no side effects
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


# ---------------------------------------------------------------------------
# Ollama model management
# ---------------------------------------------------------------------------

def ensure_model(client: object, model_name: str) -> None:
    """Verify an Ollama model is available; pull it if not.

    Uses client.show() to check availability, then client.pull(stream=True)
    to download if missing. Logs pull progress.

    Args:
        client: An ollama.Client instance.
        model_name: The model tag (e.g., "qwen3:4b").

    Raises:
        Exception: If the pull itself fails (network error, invalid model).
    """
    try:
        client.show(model_name)
        logger.info("Model %s already available.", model_name)
    except Exception:
        logger.info("Pulling model %s...", model_name)
        for progress in client.pull(model_name, stream=True):
            status = progress.get("status", "")
            if "pulling" in status or "downloading" in status:
                total = progress.get("total", 0)
                completed = progress.get("completed", 0)
                if total > 0:
                    pct = completed / total * 100
                    logger.info("  %s: %.1f%%", model_name, pct)
            elif status == "success":
                logger.info("Model %s pulled successfully.", model_name)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hotpotqa_examples(
    n: int = 200,
    seed: int = 42,
) -> tuple[list, list]:
    """Load and sample HotpotQA examples.

    Wraps src.datasets.hotpotqa.load_hotpotqa and sample_hotpotqa into a
    single call returning parallel (documents, queries) lists.

    Args:
        n: Number of examples to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (documents, queries) — parallel lists of Document and Query objects.
    """
    from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa

    docs, queries = load_hotpotqa(split="train")
    docs, queries = sample_hotpotqa(docs, queries, n=n, seed=seed)
    logger.info("Loaded %d HotpotQA examples (seed=%d).", len(docs), seed)
    return docs, queries


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(
    strategy: object,
    chunker: object,
    embedder: object,
    retrieval_mode: str,
    query: object,
    doc: object,
    model: str,
    ollama_host: str | None = None,
    reranker: object | None = None,
    reranker_top_k: int | None = None,
) -> dict:
    """Generate a single RAG answer with timing and metadata.

    Builds a Retriever per document, runs the strategy, and captures
    latency and pipeline metadata. When a reranker is provided, wraps the
    Retriever in _RerankedRetriever so strategies get reranked results
    transparently.

    Args:
        strategy: A Strategy instance (e.g., NaiveRAG).
        chunker: A Chunker instance (e.g., RecursiveChunker).
        embedder: An Embedder instance (e.g., OllamaEmbedder).
        retrieval_mode: Retrieval mode ("hybrid", "dense", or "sparse").
        query: A Query object with .text and .reference_answer attributes.
        doc: A Document object with .text attribute.
        model: Ollama model name for generation.
        ollama_host: Ollama server URL, or None for localhost.
        reranker: Optional reranker with .rerank(query, chunks, top_k) method.
        reranker_top_k: Number of chunks to keep after reranking.

    Returns:
        Dict with answer text, timing, gold metrics, and pipeline metadata.
    """
    try:
        chunks = chunker.chunk(doc.text)
        retriever = Retriever(chunks, embedder, mode=retrieval_mode)

        # Wrap retriever with reranker if provided
        if reranker is not None:
            retriever = _RerankedRetriever(
                retriever, reranker, top_k=reranker_top_k or 3,
            )

        # Diagnostics dict captures pipeline internals from inside the strategy
        diagnostics: dict = {}

        # Time the strategy run
        start = time.perf_counter()
        answer = strategy.run(
            query.text, retriever, model, diagnostics=diagnostics,
        )
        strategy_latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return {
            "answer": "",
            "strategy_latency_ms": float("nan"),
            "num_chunks": 0,
            "num_chunks_retrieved": 0,
            "context_char_length": 0,
            "error": str(exc),
            "failure_stage": "unknown",
            "failure_stage_confidence": "n/a",
            "failure_stage_method": "substring",
            "context_sent_to_llm": "",
            "gold_in_chunks": False,
            "gold_in_retrieved": False,
            "gold_in_context": False,
        }

    gold_answer = query.reference_answer or ""

    # Extract diagnostics data with safe defaults
    context_sent = diagnostics.get("context_sent_to_llm", "")
    retrieved_chunks = diagnostics.get("retrieved_chunks", [])
    retrieved_texts = [r["text"] for r in retrieved_chunks] if retrieved_chunks else []
    skipped = diagnostics.get("skipped_retrieval", False)

    # Failure attribution — only meaningful when gold answer exists
    stage, confidence = detect_failure_stage(
        gold_answer=gold_answer or None,
        rag_answer=answer,
        all_chunks=chunks,
        retrieved_chunk_texts=retrieved_texts,
        context_sent_to_llm=context_sent,
        skipped_retrieval=skipped,
    )

    # Gold presence booleans for analysis
    gold_in_chunks = any(
        _gold_in_text(gold_answer, c) for c in chunks
    ) if gold_answer else False
    gold_in_retrieved = any(
        _gold_in_text(gold_answer, t) for t in retrieved_texts
    ) if gold_answer else False
    gold_in_context = (
        _gold_in_text(gold_answer, context_sent) if gold_answer else False
    )

    return {
        "answer": answer,
        "strategy_latency_ms": strategy_latency_ms,
        "num_chunks": len(chunks),
        "num_chunks_retrieved": len(retrieved_chunks),
        # Use actual context from diagnostics instead of pre-strategy estimate
        "context_char_length": len(context_sent),
        "gold_f1": compute_f1(answer, gold_answer) if gold_answer else float("nan"),
        "gold_exact_match": exact_match(answer, gold_answer) if gold_answer else False,
        "context_sent_to_llm": context_sent,
        "failure_stage": stage,
        "failure_stage_confidence": confidence,
        "failure_stage_method": "substring",
        "gold_in_chunks": gold_in_chunks,
        "gold_in_retrieved": gold_in_retrieved,
        "gold_in_context": gold_in_context,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_answer(
    scorer: object,
    query: str,
    context: str,
    answer: str,
) -> dict:
    """Score a single answer with an LLMScorer.

    Captures scorer latency and handles failures gracefully by returning
    NaN values.

    Args:
        scorer: An LLMScorer instance.
        query: The original question text.
        context: The source document text.
        answer: The RAG-generated answer.

    Returns:
        Dict with score metrics and scorer_latency_ms. On failure,
        metrics are NaN.
    """
    start = time.perf_counter()
    try:
        scores = scorer.score(query=query, context=context, answer=answer)
        scorer_latency_ms = (time.perf_counter() - start) * 1000
        quality = sum(scores.values()) / len(scores)
        return {
            **scores,
            "quality": quality,
            "scorer_latency_ms": scorer_latency_ms,
        }
    except Exception as exc:
        scorer_latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Scorer failed: %s", exc)
        return {
            "faithfulness": float("nan"),
            "relevance": float("nan"),
            "conciseness": float("nan"),
            "quality": float("nan"),
            "scorer_latency_ms": scorer_latency_ms,
        }


# ---------------------------------------------------------------------------
# Checkpoint / CSV management
# ---------------------------------------------------------------------------

def load_checkpoint(csv_path: Path) -> set[tuple[str, str]]:
    """Load completed (strategy, model) config pairs from a checkpoint CSV.

    Reads the CSV and extracts unique (strategy, model) pairs. If the file
    doesn't exist or is empty, returns an empty set.

    Args:
        csv_path: Path to the raw_scores.csv checkpoint file.

    Returns:
        Set of (strategy, model) tuples that have already been completed.
    """
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "strategy" not in df.columns or "model" not in df.columns:
            return set()
        return set(zip(df["strategy"], df["model"]))
    except Exception:
        return set()


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV file, creating header if the file is new.

    Writes atomically per-config: flushes after each call so interrupted
    runs don't lose completed configs.

    Args:
        csv_path: Path to the output CSV file.
        rows: List of row dicts to append. If empty, does nothing.
    """
    if not rows:
        return

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = list(rows[0].keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Scorer construction
# ---------------------------------------------------------------------------

def build_scorer(scorer_str: str, max_cost: float = 10.0) -> object:
    """Build an LLMScorer from a "provider:model" string with CostGuard.

    Parses the scorer string, validates the format, and constructs the
    LLMScorer with an attached CostGuard for spend tracking.

    Args:
        scorer_str: Scorer specification as "provider:model"
            (e.g., "google:gemini-2.5-flash").
        max_cost: Maximum estimated API spend in USD.

    Returns:
        An LLMScorer instance with cost guard attached.

    Raises:
        ValueError: If scorer_str doesn't contain exactly one colon.
        SystemExit: If the scorer can't be initialized (missing API key, etc.).
    """
    from src.scorers.llm import LLMScorer
    from src.cost_guard import CostGuard

    if ":" not in scorer_str:
        raise ValueError(
            f"Invalid scorer format '{scorer_str}'. Expected 'provider:model' "
            f"(e.g., 'google:gemini-2.5-flash')."
        )

    provider, model = scorer_str.split(":", 1)
    cost_guard = CostGuard(max_cost_usd=max_cost)
    scorer = LLMScorer(provider=provider, model=model, cost_guard=cost_guard)
    logger.info("Built scorer: %s (cost limit: $%.2f)", scorer.name, max_cost)
    return scorer


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string like "45s", "2m 5s", or "1h 3m".
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, _ = divmod(remainder, 60)
        return f"{h}h {m}m"
