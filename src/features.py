"""Feature extraction for the meta-learner.

Extracts properties from the query and document before any model runs.
These become the feature matrix X for the XGBoost classifier.

Features:
- query_length: token count (approximated by word count)
- query_type: lookup / synthesis / multi_hop (one-hot encoded at training time)
- num_named_entities: proxy for specificity
- doc_length: token count of source document
- doc_vocab_entropy: lexical diversity proxy
- mean_retrieval_score: how confidently embedding matched query to doc
- var_retrieval_score: high variance = scattered context (Zhang et al., 2025)

Adapted from src/data/features.py to use the new Retriever interface instead
of raw (chunks, index) pairs. This decouples feature extraction from the
specific FAISS implementation details.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.retriever import Retriever


def extract_features(query: str, document: str, retriever: Retriever) -> dict[str, float]:
    """Extract features for a (query, document) pair.

    Args:
        query: The question text.
        document: The full document text.
        retriever: A Retriever instance (used to get retrieval scores).

    Returns:
        Dict of feature name -> value.
    """
    retrieved = retriever.retrieve(query)
    scores = [r["score"] for r in retrieved]

    return {
        "query_length": len(query.split()),
        "num_named_entities": _count_entities(query),
        "doc_length": len(document.split()),
        "doc_vocab_entropy": _vocab_entropy(document),
        "mean_retrieval_score": float(np.mean(scores)) if scores else 0.0,
        "var_retrieval_score": float(np.var(scores)) if scores else 0.0,
    }


def _count_entities(text: str) -> int:
    """Count capitalized multi-word sequences as a rough NER proxy.

    Args:
        text: Input text to scan for named entities.

    Returns:
        Count of likely named entity words.
    """
    words = text.split()
    count = 0
    for word in words:
        if word[0].isupper() and word not in ("What", "When", "Where", "Who",
                                                "Why", "How", "Is", "Are",
                                                "Does", "Do", "Can", "The", "A"):
            count += 1
    return count


def _vocab_entropy(text: str) -> float:
    """Calculate Shannon entropy of word frequency distribution.

    Args:
        text: Input text to analyze.

    Returns:
        Shannon entropy value (higher = more diverse vocabulary).
    """
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
