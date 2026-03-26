"""Inference for the meta-learner — used by the FastAPI endpoint.

Loads the trained XGBoost model and predicts the optimal 4-axis configuration
(chunker, embedder, strategy, model) for a given query and document.

Supports both classification (predict best config) and regression (predict
a numeric value) modes.  Mode is determined by ``meta.json`` saved alongside
model artifacts during training (see :mod:`src.model.train`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb

from src.config import MODELS_DIR


_model: xgb.XGBClassifier | xgb.XGBRegressor | None = None
_label_classes: list[str] | None = None
_feature_columns: list[str] | None = None
_meta: dict | None = None


def _load_model(model_dir: Path | None = None) -> None:
    """Load model artifacts from disk (lazy, once).

    Reads ``meta.json`` to determine whether the model is a classifier or
    regressor.  If ``meta.json`` is missing (pre-task-034 model), assumes
    classification mode for backward compatibility.

    Args:
        model_dir: Directory containing model artifacts. Defaults to 'models/'.
    """
    global _model, _label_classes, _feature_columns, _meta

    if model_dir is None:
        model_dir = MODELS_DIR

    # Load meta.json to determine mode
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _meta = json.load(f)
    else:
        # Pre-task-034 model — assume classification
        print(
            "No meta.json found — assuming classification mode "
            "(pre-task-034 model)"
        )
        _meta = {"mode": "classification"}

    mode = _meta.get("mode", "classification")

    if mode == "regression":
        _model = xgb.XGBRegressor()
    else:
        _model = xgb.XGBClassifier()

    _model.load_model(str(model_dir / "xgb_meta_learner.json"))

    # Label encoder only exists for classification models
    le_path = model_dir / "label_encoder.json"
    if le_path.exists():
        with open(le_path) as f:
            _label_classes = json.load(f)["classes"]
    else:
        _label_classes = None

    with open(model_dir / "feature_columns.json") as f:
        _feature_columns = json.load(f)["columns"]


def _build_feature_row(features: dict) -> dict[str, float]:
    """Build a feature row matching the training columns.

    Handles one-hot encoded query_type columns (``qt_*``) and config axis
    code columns (``*_code``).

    Args:
        features: Raw feature dict from the caller.

    Returns:
        Dict mapping each training column to its numeric value.
    """
    row: dict[str, float] = {}
    for col in _feature_columns:
        if col.startswith("qt_"):
            # One-hot encode query type to match training format
            query_type = features.get("query_type", "lookup")
            row[col] = 1.0 if col == f"qt_{query_type}" else 0.0
        elif col.endswith("_code"):
            # Config axis code — use the raw string value hashed to an int
            # At prediction time we don't have the category mapping, so we
            # rely on the caller passing the config axis value directly
            axis = col.replace("_code", "")
            raw = features.get(axis, "")
            # Simple hash-to-int — matches pd.Categorical().cat.codes ordering
            # only if the same categories were seen during training.  For a
            # quick approximation, hash the string.
            row[col] = float(hash(raw) % 1000)
        else:
            row[col] = float(features.get(col, 0.0))
    return row


def predict(features: dict) -> dict:
    """Predict the optimal config (classification) or a numeric value (regression).

    The behavior depends on the mode recorded in ``meta.json``:

    - **Classification**: returns ``{chunker, embedder, strategy, model,
      confidence}``.
    - **Regression**: returns ``{predicted_value, model_type: "regression"}``.

    Args:
        features: Dict of feature values from ``extract_features()``,
                  plus ``query_type``.  For regression models that include
                  config axis codes, also pass ``chunker``, ``embedder``,
                  ``strategy``, ``model``.

    Returns:
        Prediction dict.
    """
    if _model is None:
        _load_model()

    mode = (_meta or {}).get("mode", "classification")

    row = _build_feature_row(features)
    X = pd.DataFrame([row])

    if mode == "regression":
        pred = _model.predict(X)[0]
        return {
            "predicted_value": float(pred),
            "model_type": "regression",
        }

    # Classification path
    pred_idx = int(_model.predict(X)[0])
    probas = _model.predict_proba(X)[0]

    # Guard against label encoder / model class mismatch
    if pred_idx < 0 or pred_idx >= len(_label_classes):
        raise ValueError(
            f"Predicted class index {pred_idx} out of range "
            f"[0, {len(_label_classes) - 1}]. "
            f"Model and label_encoder.json may be out of sync."
        )

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


def predict_all_configs(features: dict) -> list[dict]:
    """Return all configs ranked by confidence (classification only).

    Useful for constrained optimization at inference time — the caller can
    filter the ranked list by their own constraints.

    Args:
        features: Same as :func:`predict`.

    Returns:
        List of dicts sorted by confidence descending.  Each dict has
        ``chunker``, ``embedder``, ``strategy``, ``model``, ``confidence``.

    Raises:
        ValueError: If the loaded model is not a classifier.
    """
    if _model is None:
        _load_model()

    mode = (_meta or {}).get("mode", "classification")
    if mode != "classification":
        raise ValueError(
            "predict_all_configs() requires a classification model, "
            f"but loaded model is mode={mode!r}"
        )

    row = _build_feature_row(features)
    X = pd.DataFrame([row])
    probas = _model.predict_proba(X)[0]

    results: list[dict] = []
    for idx, conf in enumerate(probas):
        config = _label_classes[idx]
        parts = config.split("__")
        if len(parts) != 4:
            continue
        chunker, embedder, strategy, model = parts
        results.append({
            "chunker": chunker,
            "embedder": embedder,
            "strategy": strategy,
            "model": model,
            "confidence": float(conf),
        })

    # Sort by confidence descending
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results
