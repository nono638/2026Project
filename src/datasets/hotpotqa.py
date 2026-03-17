"""HotpotQA dataset loader for RAGBench.

Converts HotpotQA (HuggingFace ``datasets`` library) into RAGBench's
Document + Query format.  Each HotpotQA example becomes one Document
(all 10 passages concatenated with markdown headers and separators) and
one Query (with ``reference_answer`` set to the gold answer).

Uses the "distractor" config (2 gold + 8 distractor passages per question),
which is the standard evaluation setup.  The "fullwiki" config requires
searching all of Wikipedia and is out of scope.

See architecture-decisions.md: "HotpotQA as primary evaluation corpus."
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
    """Convert a single HotpotQA example into a Document.

    Concatenates all non-empty passages with markdown headers and ``---``
    separators.  Passages with empty sentence lists are skipped — the
    Document is still created from the remaining passages.

    Args:
        example: A single HotpotQA example dict from the HuggingFace dataset.

    Returns:
        A Document, or None if all passages are empty.
    """
    titles: list[str] = example["context"]["title"]
    sentences_per_passage: list[list[str]] = example["context"]["sentences"]

    passage_blocks: list[str] = []
    included_titles: list[str] = []

    for title, sentences in zip(titles, sentences_per_passage):
        # Skip passages with empty text
        if not sentences:
            continue
        text = " ".join(sentences)
        passage_blocks.append(f"## {title}\n{text}")
        included_titles.append(title)

    if not passage_blocks:
        return None

    # Join passages with clear separators — helps chunkers respect boundaries
    full_text = "\n\n---\n\n".join(passage_blocks)

    return Document(
        title=f"hotpotqa:{example['id']}",
        text=full_text,
        metadata={
            "passage_titles": included_titles,
            "num_passages": len(included_titles),
        },
    )


def _build_query(example: dict) -> Query:
    """Convert a single HotpotQA example into a Query.

    Args:
        example: A single HotpotQA example dict from the HuggingFace dataset.

    Returns:
        A Query with reference_answer, difficulty, question_type, and
        supporting_titles in metadata.
    """
    supporting_titles = list(example["supporting_facts"]["title"])
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_supporting: list[str] = []
    for t in supporting_titles:
        if t not in seen:
            seen.add(t)
            unique_supporting.append(t)

    return Query(
        text=example["question"],
        query_type=example["type"],
        source_doc_title=f"hotpotqa:{example['id']}",
        reference_answer=example["answer"],
        generator_name="hotpotqa",
        metadata={
            "difficulty": example["level"],
            "question_type": example["type"],
            "supporting_titles": unique_supporting,
        },
    )


def load_hotpotqa(
    split: str = "train",
) -> tuple[list[Document], list[Query]]:
    """Load HotpotQA and convert to Document + Query lists.

    Each HotpotQA example produces one Document (10 passages concatenated)
    and one Query (with gold reference answer).  Examples with empty answers
    are skipped.

    Args:
        split: Which split to load (default ``"train"``).

    Returns:
        Tuple of (documents, queries) — one Document and one Query per valid
        example.  The two lists are parallel (same length, same order).
    """
    dataset = hf_load_dataset("hotpot_qa", "distractor")[split]

    documents: list[Document] = []
    queries: list[Query] = []

    for example in dataset:
        # Skip examples with empty answers — no gold standard to evaluate against
        if not example.get("answer"):
            logger.warning("Skipping example %s: empty answer", example.get("id"))
            continue

        doc = _build_document(example)
        if doc is None:
            logger.warning("Skipping example %s: all passages empty", example.get("id"))
            continue

        documents.append(doc)
        queries.append(_build_query(example))

    return documents, queries


def sample_hotpotqa(
    documents: list[Document],
    queries: list[Query],
    n: int = 200,
    seed: int = 42,
) -> tuple[list[Document], list[Query]]:
    """Return a stratified subset of HotpotQA documents and queries.

    Stratifies by (question_type, difficulty) — 6 strata total:
    bridge×easy, bridge×medium, bridge×hard, comparison×easy,
    comparison×medium, comparison×hard.  Samples proportionally from each
    stratum so the subset isn't dominated by one category.

    Args:
        documents: Full document list from ``load_hotpotqa()``.
        queries: Full query list from ``load_hotpotqa()`` (parallel to documents).
        n: Target sample size.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled_documents, sampled_queries) — parallel lists.
    """
    if n >= len(documents):
        return list(documents), list(queries)

    rng = random.Random(seed)

    # Group indices by (question_type, difficulty)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, q in enumerate(queries):
        meta = q.metadata or {}
        key = (meta.get("question_type", "unknown"), meta.get("difficulty", "unknown"))
        strata[key].append(i)

    # Allocate proportionally to each stratum
    total = len(documents)
    allocations: dict[tuple[str, str], int] = {}
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
        # Remove from smallest strata first (allows dropping a stratum entirely)
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
