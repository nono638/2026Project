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
    "query_length",
    "num_named_entities",
    "doc_length",
    "doc_vocab_entropy",
    "mean_retrieval_score",
    "var_retrieval_score",
]
