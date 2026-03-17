"""Inference for the meta-learner — used by the FastAPI endpoint.

Loads the trained XGBoost model and predicts the optimal 4-axis configuration
(chunker, embedder, strategy, model) for a given query and document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb

from src.config import MODELS_DIR


_model: xgb.XGBClassifier | None = None
_label_classes: list[str] | None = None
_feature_columns: list[str] | None = None


def _load_model(model_dir: Path | None = None) -> None:
    """Load model artifacts from disk (lazy, once).

    Args:
        model_dir: Directory containing model artifacts. Defaults to 'models/'.
    """
    global _model, _label_classes, _feature_columns
    if model_dir is None:
        model_dir = MODELS_DIR

    _model = xgb.XGBClassifier()
    _model.load_model(str(model_dir / "xgb_meta_learner.json"))

    with open(model_dir / "label_encoder.json") as f:
        _label_classes = json.load(f)["classes"]

    with open(model_dir / "feature_columns.json") as f:
        _feature_columns = json.load(f)["columns"]


def predict(features: dict) -> dict:
    """Predict the optimal config for given features.

    The config label format is chunker__embedder__strategy__model (4 axes),
    matching the training format from train.prepare_data.

    Args:
        features: Dict of feature values from extract_features(),
                  plus 'query_type'.

    Returns:
        Dict with 'chunker', 'embedder', 'strategy', 'model', 'confidence'.
    """
    if _model is None:
        _load_model()

    # Build feature row matching training columns
    row: dict[str, float] = {}
    for col in _feature_columns:
        if col.startswith("qt_"):
            # One-hot encode query type to match training format
            query_type = features.get("query_type", "lookup")
            row[col] = 1.0 if col == f"qt_{query_type}" else 0.0
        else:
            row[col] = features.get(col, 0.0)

    X = pd.DataFrame([row])
    pred_idx = _model.predict(X)[0]
    probas = _model.predict_proba(X)[0]

    config = _label_classes[pred_idx]
    # Config format: chunker__embedder__strategy__model (4 parts)
    parts = config.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"Expected config with 4 '__'-separated parts, got {len(parts)}: {config!r}"
        )
    chunker, embedder, strategy, model = parts

    return {
        "chunker": chunker,
        "embedder": embedder,
        "strategy": strategy,
        "model": model,
        "confidence": float(probas[pred_idx]),
    }
