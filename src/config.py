"""Shared configuration for SmallModelBigStrategy.

Centralizes constants used across training, inference, and the API.
Add new project-wide settings here rather than inlining them in modules.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---
MODELS_DIR = Path("models")

# --- Meta-learner ---
DEFAULT_QUALITY_THRESHOLD = 3.0

FEATURE_COLS = [
    # Query-level
    "query_length",
    "num_named_entities",
    # Document-level (basic)
    "doc_length",
    "doc_vocab_entropy",
    # Document characterization (content-aware)
    "doc_ner_density",
    "doc_ner_repetition",
    "doc_topic_count",
    "doc_topic_density",
    "doc_semantic_coherence",
    # Extended document features
    "doc_readability_score",
    "doc_embedding_spread",
    # Query-document features
    "query_doc_similarity",
    "query_doc_lexical_overlap",
    # Retrieval-level
    "mean_retrieval_score",
    "var_retrieval_score",
]
