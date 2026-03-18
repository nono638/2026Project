"""SQuAD 2.0 dataset loader for RAGBench.

Converts SQuAD 2.0 (HuggingFace ``datasets`` library) into RAGBench's
Document + Query format.  Each answerable SQuAD example becomes one
Document (single context paragraph) and one Query (with ``reference_answer``
set to the first gold answer span).

Unanswerable questions (empty ``answers["text"]``) are skipped — RAG
evaluation needs a reference to compare against.

This is an "easy baseline" gold-standard dataset alongside HotpotQA's
harder multi-hop questions, enabling calibration across difficulty levels.

See architecture-decisions.md: "HotpotQA as primary evaluation corpus"
(SQuAD is the secondary calibration dataset).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from datasets import load_dataset as hf_load_dataset  # type: ignore[import-untyped]

from src.document import Document
from src.query import Query

logger = logging.getLogger(__name__)


def _build_document(example: dict) -> Document | None:
    """Convert a single SQuAD 2.0 example into a Document.

    Each SQuAD example has a single context paragraph (unlike HotpotQA's
    10-passage concatenation), so the text is used directly without headers
    or separators.

    Args:
        example: A single SQuAD 2.0 example dict from the HuggingFace dataset.

    Returns:
        A Document, or None if the context is empty.
    """
    context = example.get("context", "")
    if not context or not context.strip():
        logger.warning("Skipping example %s: empty context", example.get("id"))
        return None

    return Document(
        title=f"squad:{example['id']}",
        text=context,
        metadata={
            "article_title": example.get("title", ""),
        },
    )


def _build_query(example: dict) -> Query:
    """Convert a single SQuAD 2.0 example into a Query.

    Uses the first answer span as the reference answer. All SQuAD questions
    are extractive factoid — unlike HotpotQA's bridge/comparison distinction,
    SQuAD has no type labels.

    Args:
        example: A single SQuAD 2.0 example dict from the HuggingFace dataset.

    Returns:
        A Query with reference_answer set to the first answer span.
    """
    answer_texts = example["answers"]["text"]

    return Query(
        text=example["question"],
        query_type="factoid",  # All SQuAD questions are extractive factoid
        source_doc_title=f"squad:{example['id']}",
        reference_answer=answer_texts[0],  # Use first span — multiple spans are same answer at different positions
        generator_name="squad",
        metadata={
            "article_title": example.get("title", ""),
            "num_answer_spans": len(answer_texts),
        },
    )


def load_squad(
    split: str = "train",
) -> tuple[list[Document], list[Query]]:
    """Load SQuAD 2.0 and convert to Document + Query lists.

    Each answerable SQuAD example produces one Document (single context
    paragraph) and one Query (with gold reference answer).  Unanswerable
    questions are skipped because RAG evaluation needs a reference to
    compare against.

    Args:
        split: Which split to load (default ``"train"``).

    Returns:
        Tuple of (documents, queries) — one Document and one Query per valid
        example.  The two lists are parallel (same length, same order).
    """
    dataset = hf_load_dataset("squad_v2")[split]

    documents: list[Document] = []
    queries: list[Query] = []

    for example in dataset:
        # Skip unanswerable questions — empty answers list means no gold answer
        answer_texts = example.get("answers", {}).get("text", [])
        if not answer_texts:
            logger.warning("Skipping unanswerable example %s", example.get("id"))
            continue

        doc = _build_document(example)
        if doc is None:
            # Empty context already logged in _build_document
            continue

        documents.append(doc)
        queries.append(_build_query(example))

    return documents, queries


def sample_squad(
    documents: list[Document],
    queries: list[Query],
    n: int = 200,
    seed: int = 42,
) -> tuple[list[Document], list[Query]]:
    """Return a stratified subset of SQuAD documents and queries.

    Stratifies by article title (442 unique articles in SQuAD 2.0) to
    ensure topic diversity — unlike HotpotQA which stratifies by
    (type, difficulty), SQuAD lacks those fields so article title is
    the natural sampling axis.

    Args:
        documents: Full document list from ``load_squad()``.
        queries: Full query list from ``load_squad()`` (parallel to documents).
        n: Target sample size.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled_documents, sampled_queries) — parallel lists.
    """
    if n <= 0:
        return [], []

    if n >= len(documents):
        return list(documents), list(queries)

    rng = random.Random(seed)

    # Group indices by article title for stratified sampling
    strata: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(queries):
        meta = q.metadata or {}
        article = meta.get("article_title", "unknown")
        strata[article].append(i)

    # Allocate proportionally to each stratum
    total = len(documents)
    allocations: dict[str, int] = {}
    for key, indices in strata.items():
        alloc = round(len(indices) / total * n)
        alloc = min(alloc, len(indices))
        allocations[key] = alloc

    # Adjust to hit exactly n — add to largest strata first, remove from smallest
    current = sum(allocations.values())
    sorted_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)

    while current < n:
        adjusted = False
        for key in sorted_keys:
            if allocations[key] < len(strata[key]):
                allocations[key] += 1
                current += 1
                adjusted = True
                if current >= n:
                    break
        if not adjusted:
            break

    while current > n:
        adjusted = False
        for key in sorted(sorted_keys, key=lambda k: allocations[k]):
            if allocations[key] > 0:
                allocations[key] -= 1
                current -= 1
                adjusted = True
                if current <= n:
                    break
        if not adjusted:
            break

    # Sample from each stratum, skipping strata with 0 allocation
    sampled_indices: list[int] = []
    for key in sorted(strata.keys()):
        k = allocations[key]
        if k > 0:
            sampled_indices.extend(rng.sample(strata[key], k))

    sampled_indices.sort()

    sampled_docs = [documents[i] for i in sampled_indices]
    sampled_queries = [queries[i] for i in sampled_indices]

    return sampled_docs, sampled_queries
