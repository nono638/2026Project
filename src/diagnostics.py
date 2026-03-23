"""Pipeline failure attribution diagnostics.

Provides detect_failure_stage() to identify which RAG pipeline stage
caused an incorrect answer by tracing gold answer presence through
each stage: chunker -> retrieval -> filtering -> generation.

Motivated by Experiment 0 analysis — the Church of St. Anne question
showed a retrieval failure but we couldn't diagnose it from the CSV
alone. This module closes that observability gap.
"""

from __future__ import annotations


def _gold_in_text(gold: str, text: str) -> bool:
    """Check if gold answer appears in text (case-insensitive substring).

    Uses simple ``in`` operator, not regex, so gold answers with special
    characters (parentheses, brackets, etc.) are safe.

    Args:
        gold: The gold reference answer.
        text: The text to search.

    Returns:
        True if gold appears in text (case-insensitive).
    """
    return gold.lower() in text.lower()


def detect_failure_stage(
    gold_answer: str | None,
    rag_answer: str,
    all_chunks: list[str],
    retrieved_chunk_texts: list[str],
    context_sent_to_llm: str,
    skipped_retrieval: bool = False,
) -> tuple[str, str]:
    """Identify which pipeline stage caused an incorrect RAG answer.

    Traces gold answer presence through each pipeline stage using
    case-insensitive substring matching (consistent with exact_match()
    in experiment_utils.py). Returns a (stage, confidence) tuple.

    Confidence is "high" when stage-to-stage transitions are definitional
    (gold present at stage N but absent at N+1). The only "low" case is
    "chunker" — the gold answer might be paraphrased differently in the
    source document (e.g., "887" vs "eight hundred eighty-seven").

    Args:
        gold_answer: The gold reference answer. None or empty = unknown.
        rag_answer: The RAG-generated answer.
        all_chunks: All chunk texts from the chunker (before retrieval).
        retrieved_chunk_texts: Chunk texts returned by retrieval.
        context_sent_to_llm: The exact context string in the final prompt.
        skipped_retrieval: True when strategy bypassed retrieval entirely
            (AdaptiveRAG simple path, SelfRAG "no retrieval" decision).

    Returns:
        Tuple of (stage, confidence) where stage is one of: "none",
        "chunker", "retrieval", "filtering", "generation",
        "no_retrieval", "unknown"; and confidence is "high", "low",
        or "n/a".
    """
    # No gold answer — can't attribute
    if not gold_answer:
        return ("unknown", "n/a")

    # Gold found in answer — no failure
    if _gold_in_text(gold_answer, rag_answer):
        return ("none", "n/a")

    # Strategy skipped retrieval entirely
    if skipped_retrieval:
        return ("no_retrieval", "high")

    # Gold not in ANY chunk — chunker lost the information
    if not any(_gold_in_text(gold_answer, chunk) for chunk in all_chunks):
        return ("chunker", "low")

    # Gold in chunks but not in any retrieved chunk — retrieval failure
    if not any(_gold_in_text(gold_answer, chunk) for chunk in retrieved_chunk_texts):
        return ("retrieval", "high")

    # Gold in retrieved chunks but not in context — filtering removed it
    if not _gold_in_text(gold_answer, context_sent_to_llm):
        return ("filtering", "high")

    # Gold in context but not in answer — generation failure
    return ("generation", "high")
