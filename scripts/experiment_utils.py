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
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Ensure project root is on sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.diagnostics import detect_failure_stage, _gold_in_text

logger = logging.getLogger(__name__)

# Retry settings — applies to Ollama generation, API scoring, and model pulls
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry (2s, 4s, 8s)

# After an Ollama model runner crash, the process needs time to restart
# before a retry attempt will succeed. 2-8 s (normal backoff) is far too
# short — in practice the runner takes 30-90 s to come back up.
# The 8-17 minute hung chat() calls before BSOD #3 (2026-05-18) were caused
# by: no generation timeout + no runner-crash recovery wait = the retry
# immediately re-sent the request into a dead runner and hung again.
RUNNER_CRASH_RECOVERY_S = 60.0
# After waiting RUNNER_CRASH_RECOVERY_S we poll Ollama's /api/tags up to
# this many seconds before giving up and retrying anyway.
RUNNER_HEALTH_POLL_S = 60.0


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is likely transient and worth retrying.

    Catches connection drops, timeouts, rate limits, and server errors
    without importing every library's specific exception classes.

    Args:
        exc: The exception to check.

    Returns:
        True if the error looks transient.
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    msg = str(exc).lower()
    transient_patterns = [
        "timeout", "timed out", "connection", "reset by peer",
        "rate limit", "429", "500", "502", "503", "504",
        "overloaded", "temporarily", "unavailable", "broken pipe",
        "server error", "internal error", "resource exhausted",
    ]
    return any(p in msg for p in transient_patterns)


def _is_runner_crash(exc: Exception) -> bool:
    """True if the Ollama model runner itself crashed (not a transient blip).

    Runner crashes need a long recovery wait — the process has to restart
    before any retry will succeed. This is distinct from a TCP reset or
    HTTP 500 where Ollama itself is still up.

    Args:
        exc: The exception to classify.

    Returns:
        True when the error text matches known runner-crash patterns.
    """
    msg = str(exc).lower()
    return any(p in msg for p in [
        "model runner has unexpectedly stopped",
        "resource limitations",
    ])


def _wait_for_ollama(
    host: str | None = None,
    max_wait_s: float = RUNNER_HEALTH_POLL_S,
) -> bool:
    """Poll Ollama's /api/tags until it responds or max_wait_s elapses.

    Used after a model runner crash to confirm the server is accepting
    requests again before issuing a retry. Avoids re-hanging on a
    still-recovering runner.

    Args:
        host: Ollama server URL. None defaults to http://localhost:11434.
        max_wait_s: Maximum seconds to poll before giving up.

    Returns:
        True if Ollama responded OK within the window; False on timeout.
    """
    import time as _time

    base = (host or "http://localhost:11434").rstrip("/")
    url = f"{base}/api/tags"
    deadline = _time.monotonic() + max_wait_s
    poll_s = 5.0
    while _time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        _time.sleep(poll_s)
    return False


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


# ---------------------------------------------------------------------------
# Ollama model management
# ---------------------------------------------------------------------------

def get_ollama_model_details(
    model_tag: str,
    host: str | None = None,
) -> dict:
    """Query Ollama's /api/show for resolved model details.

    Captures provenance fields (quantization, digest, parameter size, family,
    format) so experiments can record exactly which artifact Ollama loaded.
    Reproducibility hole closer for task-053 — Ollama can re-point a default
    tag without notice; this stamps the resolved manifest at run time.

    See: https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information

    Never raises — on any HTTP error, missing model, or unreachable host the
    return dict still has tag and captured_at populated, with the rest None.
    A misconfigured Ollama host shouldn't kill an experiment that already
    generated answers; better to write 'unknown' and surface the issue in
    logs than lose hours of generation.

    Args:
        model_tag: The Ollama tag (e.g., "qwen3:4b" or "qwen3:4b-q4_K_M").
        host: Ollama server URL like "http://1.2.3.4:11434". When None,
            defaults to http://localhost:11434.

    Returns:
        Dict with keys: tag, digest, quantization_level, parameter_size,
        family, format, captured_at. Only tag and captured_at are guaranteed
        non-None.
    """
    captured_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    base_host = host or "http://localhost:11434"
    if not base_host.startswith(("http://", "https://")):
        # Allow callers to pass a bare host:port like the Ollama Python client
        base_host = f"http://{base_host}"
    url = f"{base_host.rstrip('/')}/api/show"

    blank = {
        "tag": model_tag,
        "digest": None,
        "quantization_level": None,
        "parameter_size": None,
        "family": None,
        "format": None,
        "captured_at": captured_at,
    }

    try:
        resp = requests.post(
            url,
            json={"model": model_tag, "verbose": False},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("Ollama /api/show unreachable for %s: %s", model_tag, exc)
        return blank

    if not resp.ok:
        logger.warning(
            "Ollama /api/show returned %s for %s: %s",
            resp.status_code, model_tag, resp.text[:200],
        )
        return blank

    try:
        body = resp.json()
    except ValueError as exc:
        logger.warning("Ollama /api/show returned non-JSON for %s: %s", model_tag, exc)
        return blank

    details = body.get("details") if isinstance(body, dict) else None
    if not isinstance(details, dict):
        details = {}

    return {
        "tag": model_tag,
        "digest": body.get("digest") if isinstance(body, dict) else None,
        "quantization_level": details.get("quantization_level"),
        "parameter_size": details.get("parameter_size"),
        "family": details.get("family"),
        "format": details.get("format"),
        "captured_at": captured_at,
    }


def ensure_model(client: object, model_name: str) -> None:
    """Verify an Ollama model is available; pull it if not.

    Uses client.show() to check availability, then client.pull(stream=True)
    to download if missing. Retries on transient network failures.

    Args:
        client: An ollama.Client instance.
        model_name: The model tag (e.g., "qwen3:4b").

    Raises:
        Exception: If the pull fails after all retries.
    """
    try:
        client.show(model_name)
        logger.info("Model %s already available.", model_name)
        return
    except Exception:
        pass  # model not present — pull it

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info("Pulling model %s (attempt %d/%d)...",
                        model_name, attempt + 1, MAX_RETRIES + 1)
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
            return  # success
        except Exception as exc:
            if attempt < MAX_RETRIES and _is_transient(exc):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Model pull retry %d/%d for %s: %s (waiting %.0fs)",
                    attempt + 1, MAX_RETRIES, model_name, exc, delay,
                )
                time.sleep(delay)
                continue
            raise


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
    model_details: dict | None = None,
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
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
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
            break  # success
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES and _is_transient(exc):
                if _is_runner_crash(exc):
                    # Model runner process died — needs significant recovery
                    # time before a retry will succeed. Normal 2-8 s backoff
                    # sends the request straight back into a dead runner.
                    logger.warning(
                        "Generation retry %d/%d: %s (waiting %.0fs for "
                        "runner recovery + polling until Ollama is ready)",
                        attempt + 1, MAX_RETRIES, exc, RUNNER_CRASH_RECOVERY_S,
                    )
                    time.sleep(RUNNER_CRASH_RECOVERY_S)
                    ready = _wait_for_ollama(ollama_host, max_wait_s=RUNNER_HEALTH_POLL_S)
                    if not ready:
                        logger.warning(
                            "Ollama did not respond within %.0fs after runner "
                            "crash — retrying anyway",
                            RUNNER_HEALTH_POLL_S,
                        )
                else:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Generation retry %d/%d: %s (waiting %.0fs)",
                        attempt + 1, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                continue
            logger.error("Generation failed after %d attempt(s): %s",
                         attempt + 1, exc)
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
                "llm_quantization": (
                    (model_details or {}).get("quantization_level") or "unknown"
                ),
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
        # task-053: per-row quantization provenance. "unknown" when the
        # /api/show lookup wasn't supplied or returned None — keeps the
        # column populated so downstream analysis never silently sees NaN.
        "llm_quantization": (
            (model_details or {}).get("quantization_level") or "unknown"
        ),
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
    for attempt in range(MAX_RETRIES + 1):
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
            if attempt < MAX_RETRIES and _is_transient(exc):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Scorer retry %d/%d: %s (waiting %.0fs)",
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            scorer_latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Scorer failed after %d attempt(s): %s",
                         attempt + 1, exc)
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

def load_checkpoint(
    csv_path: Path,
    key_cols: tuple[str, ...] = ("strategy", "model"),
) -> set[tuple]:
    """Load completed key tuples from a checkpoint CSV.

    Reads the CSV and extracts unique tuples across the specified key columns.
    If the file doesn't exist or is empty, returns an empty set.

    Args:
        csv_path: Path to the raw_scores.csv checkpoint file.
        key_cols: Tuple of column names to use as the checkpoint key.
            Default is ("strategy", "model") for Experiment 1 config-level
            resume. Pass ("strategy", "model", "question") for row-level
            resume so a partial config picks up where it left off.

    Returns:
        Set of value-tuples already present in the CSV.
    """
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if df.empty or any(c not in df.columns for c in key_cols):
            return set()
        return set(zip(*(df[c] for c in key_cols)))
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

    if file_exists:
        # When appending, fieldnames MUST match the existing header's order;
        # otherwise DictWriter writes columns in rows[0].keys() order and
        # silently misaligns with the header already on disk. Read the header
        # back from the file as the source of truth.
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader, None) or list(rows[0].keys())
        extrasaction = "ignore"  # tolerate new columns that weren't in the original header
    else:
        # New file: collect all keys across the batch so heterogeneous rows
        # don't drop columns from the very first header.
        seen: dict[str, None] = {}
        for row in rows:
            for k in row.keys():
                seen.setdefault(k, None)
        fieldnames = list(seen.keys())
        extrasaction = "raise"

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Scorer construction
# ---------------------------------------------------------------------------

def build_scorer(
    scorer_str: str,
    max_cost: float = 10.0,
    cost_guard: object | None = None,
) -> object:
    """Build an LLMScorer from a "provider:model" string with CostGuard.

    Parses the scorer string, validates the format, and constructs the
    LLMScorer with an attached CostGuard for spend tracking.

    Args:
        scorer_str: Scorer specification as "provider:model"
            (e.g., "google:gemini-2.5-flash").
        max_cost: Maximum estimated API spend in USD. Ignored if
            ``cost_guard`` is provided.
        cost_guard: Optional pre-built CostGuard to share across multiple
            scorers. When None, a fresh guard is built from ``max_cost``.
            Multi-judge panels use a single shared guard so ``--max-cost``
            is the global ceiling, not a per-judge ceiling.

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
    if cost_guard is None:
        cost_guard = CostGuard(max_cost_usd=max_cost)
    scorer = LLMScorer(provider=provider, model=model, cost_guard=cost_guard)
    logger.info("Built scorer: %s", scorer.name)
    return scorer


# ---------------------------------------------------------------------------
# Multi-judge scoring
# ---------------------------------------------------------------------------

def _safe_scorer_name(name: str) -> str:
    """Convert a scorer name to a CSV-safe column prefix.

    Mirrors the helper exported from run_experiment_0.py so this module
    doesn't import a sibling experiment script (avoiding the circular
    import that would create).
    """
    return name.replace(":", "_").replace("-", "_").replace(".", "_")


def score_answer_multi(
    scorers: list,
    query: str,
    context: str,
    answer: str,
    existing_row: dict | None = None,
) -> dict:
    """Score one answer with a panel of judges and return prefixed columns.

    For each judge in ``scorers``, produces five columns prefixed with the
    judge's safe name: ``<safe>_faithfulness``, ``<safe>_relevance``,
    ``<safe>_conciseness``, ``<safe>_quality``, ``<safe>_scorer_latency_ms``.
    Adds an unprefixed ``consensus_quality`` = NaN-safe mean of each judge's
    quality score.

    Resume support: if ``existing_row`` is provided AND already has a
    non-NaN ``<safe>_quality`` for a judge, that judge's call is skipped
    and the existing values are copied through. Why: re-scoring with a
    larger panel shouldn't re-pay for judges that already scored.

    Failure handling: any exception from a judge's ``.score(...)`` call
    (transient retries already happen inside ``score_answer``) yields NaN
    for that judge's columns. Cost-limit exceptions are also caught here
    so a tripped guard doesn't poison the rest of the panel. The caller
    can detect a cost-limit hit by inspecting the shared CostGuard.

    Args:
        scorers: List of LLMScorer instances (or mocks with .name and
            .score(query, context, answer)).
        query: Question text.
        context: Source/retrieved context the answer was generated against.
        answer: RAG-generated answer text.
        existing_row: Optional dict of pre-existing per-judge values for
            the same (query, answer). Keys must use the same safe-name
            prefix scheme.

    Returns:
        Flat dict with per-judge prefixed columns + consensus_quality.
    """
    import math as _math

    out: dict = {}
    quality_values: list[float] = []

    for scorer in scorers:
        safe = _safe_scorer_name(scorer.name)
        q_col = f"{safe}_quality"

        # Resume: skip judge that already has a non-NaN quality value.
        # Why: lets us add new judges to a panel without re-paying for
        # previously-scored rows. Same pattern as score_all_answers in
        # run_experiment_0.py.
        if existing_row is not None and q_col in existing_row:
            existing_q = existing_row.get(q_col)
            try:
                if existing_q is not None and not _math.isnan(float(existing_q)):
                    for metric in ("faithfulness", "relevance", "conciseness",
                                   "quality", "scorer_latency_ms"):
                        col = f"{safe}_{metric}"
                        if col in existing_row:
                            out[col] = existing_row[col]
                    quality_values.append(float(existing_q))
                    continue
            except (TypeError, ValueError):
                # Existing value isn't numeric — fall through to re-score
                pass

        start = time.perf_counter()
        try:
            scores = scorer.score(query=query, context=context, answer=answer)
            scorer_latency_ms = (time.perf_counter() - start) * 1000
            quality = sum(scores.values()) / len(scores)
            out[f"{safe}_faithfulness"] = scores.get("faithfulness", float("nan"))
            out[f"{safe}_relevance"] = scores.get("relevance", float("nan"))
            out[f"{safe}_conciseness"] = scores.get("conciseness", float("nan"))
            out[f"{safe}_quality"] = quality
            out[f"{safe}_scorer_latency_ms"] = scorer_latency_ms
            quality_values.append(quality)
        except Exception as exc:
            scorer_latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("Judge %s failed: %s", scorer.name, exc)
            out[f"{safe}_faithfulness"] = float("nan")
            out[f"{safe}_relevance"] = float("nan")
            out[f"{safe}_conciseness"] = float("nan")
            out[f"{safe}_quality"] = float("nan")
            out[f"{safe}_scorer_latency_ms"] = scorer_latency_ms

    # NaN-safe consensus: mean of judges that produced numeric quality.
    valid = [q for q in quality_values if not _math.isnan(q)]
    out["consensus_quality"] = (
        sum(valid) / len(valid) if valid else float("nan")
    )
    return out


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


def write_experiment_metadata(
    output_dir: Path,
    n_examples: int,
    config: dict,
    judges: list[dict],
    model_details: dict[str, dict] | None = None,
    extra: dict | None = None,
) -> None:
    """Write a metadata.json sidecar capturing run provenance.

    Each experiment writes one of these alongside its raw_scores.csv to
    record exactly which judge model versions were used, the run config,
    and any experiment-specific fields. Without it, someone reading the
    CSV later has no way to know whether a "claude-sonnet-4" column came
    from Sonnet 4 or 4.6 — the column name doesn't include the snapshot.

    Args:
        output_dir: Experiment output directory (e.g. results/experiment_1).
            ``output_dir.name`` becomes the ``experiment`` field.
        n_examples: Row count in raw_scores.csv.
        config: Run config dict — anything the caller wants preserved
            (model, seed, strategies, chunkers, etc). Pass a flat dict.
        judges: List of judge dicts. Each should have at least
            ``provider``, ``model``, and ``display_name``. Caller decides
            what counts as a "judge that contributed scores."
        extra: Optional dict merged at the top level for experiment-
            specific fields (e.g. a "test_models" list for Exp 1/2).
    """
    metadata = {
        "experiment": output_dir.name,
        "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_examples": n_examples,
        "config": config,
        "judges": judges,
    }
    # task-053: optional Ollama model_details map (tag -> /api/show fields).
    # Only written when supplied; absent metadata predates this hook.
    if model_details:
        metadata["model_details"] = model_details
    if extra:
        metadata.update(extra)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", metadata_path)
