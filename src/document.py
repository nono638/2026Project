"""Document representation for the RAG research tool.

Separate module (not in protocols.py) because Document will grow to support
multimodal content (images, video) in the future. Keeping it isolated means
multimodal changes don't ripple through protocol definitions.
See architecture-decisions.md: "every pipeline stage gets its own protocol."
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Document:
    """A document in the evaluation corpus.

    Attributes:
        title: Human-readable document identifier.
        text: Full document text content.
        metadata: Optional extensible metadata (e.g., source, word_count, domain).
    """

    title: str
    text: str
    metadata: dict | None = field(default=None)


def load_corpus_from_csv(
    path: str | Path,
    title_col: str = "title",
    text_col: str = "text",
    metadata_cols: list[str] | None = None,
) -> list[Document]:
    """Load a document corpus from a CSV file.

    Designed for the Wikipedia dataset but works with any CSV that has title
    and text columns. Skips rows with null text — the Wikipedia dataset has
    36 null texts that would break downstream chunking.

    Args:
        path: Path to the CSV file.
        title_col: Column name containing document titles.
        text_col: Column name containing document text.
        metadata_cols: Optional list of column names to include as metadata.

    Returns:
        List of Document objects, one per valid row.
    """
    df = pd.read_csv(path)

    # Skip rows where text is null/NaN — these break downstream processing
    df = df.dropna(subset=[text_col])

    documents: list[Document] = []
    for _, row in df.iterrows():
        metadata = None
        if metadata_cols:
            metadata = {col: row[col] for col in metadata_cols if col in row.index}

        documents.append(
            Document(
                title=str(row[title_col]),
                text=str(row[text_col]),
                metadata=metadata,
            )
        )

    return documents


def sample_corpus(
    documents: list[Document],
    n: int = 200,
    seed: int = 42,
    stratify_by: str = "length",
) -> list[Document]:
    """Sample a subset of documents, stratified by document length.

    Stratified sampling ensures the sample includes both short and long
    documents proportionally, which matters because chunking behavior and
    retrieval difficulty vary with document length.

    Args:
        documents: Full list of documents to sample from.
        n: Target number of documents to return.
        seed: Random seed for reproducibility.
        stratify_by: Stratification criterion. Only "length" is supported.

    Returns:
        Sampled list of documents. Returns all documents if n >= len(documents).
    """
    if n <= 0:
        return []

    if n >= len(documents):
        return list(documents)

    rng = random.Random(seed)

    if stratify_by == "length":
        # Bin documents into quartiles by text length
        sorted_docs = sorted(documents, key=lambda d: len(d.text))
        quartile_size = len(sorted_docs) // 4 or 1
        bins: list[list[Document]] = []
        for i in range(0, len(sorted_docs), quartile_size):
            bins.append(sorted_docs[i : i + quartile_size])

        # Sample proportionally from each bin, then adjust to hit exactly n
        allocations: list[int] = []
        for bin_docs in bins:
            bin_n = round(len(bin_docs) / len(documents) * n)
            bin_n = max(1, min(bin_n, len(bin_docs)))
            allocations.append(bin_n)

        # Adjust for rounding: add/remove from largest bins to hit n exactly
        total_alloc = sum(allocations)
        while total_alloc < n:
            # Add one sample to the largest under-capacity bin
            for i in sorted(range(len(bins)), key=lambda j: len(bins[j]), reverse=True):
                if allocations[i] < len(bins[i]):
                    allocations[i] += 1
                    total_alloc += 1
                    break
            else:
                break  # All bins at capacity
        while total_alloc > n:
            # Remove one sample from the largest bin
            for i in sorted(range(len(bins)), key=lambda j: allocations[j], reverse=True):
                if allocations[i] > 1:
                    allocations[i] -= 1
                    total_alloc -= 1
                    break
            else:
                break

        sampled: list[Document] = []
        for bin_docs, bin_n in zip(bins, allocations):
            sampled.extend(rng.sample(bin_docs, bin_n))

        return sampled

    # Fallback: simple random sample for unknown stratify_by values
    return rng.sample(documents, n)


def documents_to_dicts(documents: list[Document]) -> list[dict]:
    """Convert Document objects to the dict format Experiment.load_corpus expects.

    Bridge helper so the new Document type works with the existing experiment
    runner without modifying experiment.py.

    Args:
        documents: List of Document objects.

    Returns:
        List of dicts with 'title' and 'text' keys.
    """
    return [{"title": d.title, "text": d.text} for d in documents]
