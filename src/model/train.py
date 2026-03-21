"""XGBoost meta-learner training — flexible objectives.

Trains a classifier or regressor to predict the optimal (chunker, embedder,
strategy, model) configuration from query/document features.  Supports:

- **Classification** (default): predict the best 4-axis config label.
  Winner is selected per (query, doc) group via ``objective`` (maximize or
  minimize the ``target`` column).
- **Regression**: predict a numeric target (quality, latency, etc.) given
  features + config columns.
- **Constraints**: filter rows before training (e.g. ``{"quality": ">3.0"}``).

Auto-detection: ``target="config"`` → classification.  Numeric target with
no explicit mode → regression.  Numeric target + ``mode="classification"``
→ "which config gives best {target}?".

Primary model: XGBoost (XGBClassifier / XGBRegressor).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

from src.config import DEFAULT_QUALITY_THRESHOLD, FEATURE_COLS, MODELS_DIR

# Config axis columns used to build the 4-axis label
_CONFIG_AXES = ["chunker", "embedder", "strategy", "model"]

# Regex for constraint string parsing: operator + value
_CONSTRAINT_RE = re.compile(r"^(>=|<=|!=|==|>|<)(.+)$")


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

def _apply_constraints(
    df: pd.DataFrame,
    constraints: dict[str, str],
) -> pd.DataFrame:
    """Filter *df* according to constraint expressions.

    Each key is a column name, each value is an operator + value string
    (e.g. ``">3.0"``, ``"<=5"``, ``"==NaiveRAG"``).  Numeric values are
    parsed as float; everything else is compared as a string.

    Args:
        df: Input DataFrame.
        constraints: Mapping of column → constraint expression.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If a constraint string cannot be parsed.
    """
    mask = pd.Series(True, index=df.index)

    for col, expr in constraints.items():
        m = _CONSTRAINT_RE.match(expr)
        if m is None:
            raise ValueError(
                f"Invalid constraint format: '{expr}'. "
                "Expected operator + value, e.g., '>3.0', '<=5', '==NaiveRAG'"
            )
        op, raw_value = m.group(1), m.group(2)

        # Try numeric parse; fall back to string comparison
        try:
            value: float | str = float(raw_value)
        except ValueError:
            value = raw_value

        if op == ">":
            mask &= df[col] > value
        elif op == ">=":
            mask &= df[col] >= value
        elif op == "<":
            mask &= df[col] < value
        elif op == "<=":
            mask &= df[col] <= value
        elif op == "==":
            mask &= df[col] == value
        elif op == "!=":
            mask &= df[col] != value

    return df.loc[mask]


# ---------------------------------------------------------------------------
# Data preparation — classification
# ---------------------------------------------------------------------------

def _prepare_classification_data(
    df: pd.DataFrame,
    target: str = "quality",
    objective: str = "maximize",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X/y for classification (predict best config per group).

    Groups by (query_text, doc_title) and selects the winner per group
    according to ``objective`` applied to the ``target`` column.

    Args:
        df: Experiment results DataFrame (already constraint-filtered).
        target: Column used for winner selection within each group.
        objective: ``"maximize"`` or ``"minimize"``.
        features: Feature column names.  Defaults to :data:`FEATURE_COLS`.

    Returns:
        Tuple of (feature DataFrame, target Series of config labels).
    """
    if features is None:
        features = list(FEATURE_COLS)

    # Ensure quality column exists (needed for backward compat default path)
    if "quality" not in df.columns:
        score_cols = ["faithfulness", "relevance", "conciseness"]
        existing = [c for c in score_cols if c in df.columns]
        if existing:
            df = df.copy()
            df["quality"] = df[existing].mean(axis=1)

    # Build 4-axis config label
    if "config" not in df.columns:
        df = df.copy()
        df["config"] = (
            df["chunker"] + "__" + df["embedder"] + "__"
            + df["strategy"] + "__" + df["model"]
        )

    ascending = objective == "minimize"

    targets: list[str] = []
    feature_rows: list[dict] = []

    for (_query, _doc), group in df.groupby(["query_text", "doc_title"]):
        if group.empty:
            continue
        best = group.sort_values(target, ascending=ascending).iloc[0]
        targets.append(best["config"])
        feature_rows.append(best[features + ["query_type"]].to_dict())

    features_df = pd.DataFrame(feature_rows)
    # One-hot encode query_type
    features_df = pd.get_dummies(features_df, columns=["query_type"], prefix="qt")

    return features_df, pd.Series(targets, name="config")


def prepare_data(
    df: pd.DataFrame,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from experiment results.

    Backward-compatible entry point that delegates to
    :func:`_prepare_classification_data` with the original threshold-based
    winner selection logic (best quality among rows clearing threshold).

    Args:
        df: DataFrame from ExperimentResult or loaded from Parquet.
        quality_threshold: Minimum quality score for "acceptable" answer.

    Returns:
        Tuple of (feature DataFrame, target Series).

    Raises:
        ValueError: If no score columns are found in the data.
    """
    if "quality" not in df.columns:
        score_cols = ["faithfulness", "relevance", "conciseness"]
        existing = [c for c in score_cols if c in df.columns]
        if existing:
            df = df.copy()
            df["quality"] = df[existing].mean(axis=1)
        else:
            raise ValueError("No score columns found in data")

    # Build 4-axis config label
    df = df.copy()
    df["config"] = (
        df["chunker"] + "__" + df["embedder"] + "__"
        + df["strategy"] + "__" + df["model"]
    )

    targets: list[str] = []
    feature_rows: list[dict] = []

    for (_query, _doc), group in df.groupby(["query_text", "doc_title"]):
        viable = group[group["quality"] >= quality_threshold]
        if viable.empty:
            best = group.loc[group["quality"].idxmax()]
        else:
            best = viable.sort_values("quality", ascending=False).iloc[0]

        targets.append(best["config"])
        feature_rows.append(best[FEATURE_COLS + ["query_type"]].to_dict())

    features_df = pd.DataFrame(feature_rows)
    features_df = pd.get_dummies(features_df, columns=["query_type"], prefix="qt")

    return features_df, pd.Series(targets, name="config")


# ---------------------------------------------------------------------------
# Data preparation — regression
# ---------------------------------------------------------------------------

def _prepare_regression_data(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X/y for regression (predict a numeric value per row).

    Config columns (chunker, embedder, strategy, model) are encoded as
    pandas categoricals (integer codes) so XGBoost can split on them.

    Args:
        df: Experiment results DataFrame (already constraint-filtered).
        target: Numeric column to predict.
        features: Base feature columns.  Defaults to :data:`FEATURE_COLS`.

    Returns:
        Tuple of (feature DataFrame, target Series).
    """
    if features is None:
        features = list(FEATURE_COLS)

    df = df.copy()

    # Build feature matrix: FEATURE_COLS + query_type one-hot + config axes
    feature_cols_to_use = [c for c in features if c in df.columns]

    # One-hot encode query_type if present
    if "query_type" in df.columns:
        df = pd.get_dummies(df, columns=["query_type"], prefix="qt")
        qt_cols = [c for c in df.columns if c.startswith("qt_")]
    else:
        qt_cols = []

    # Encode config axis columns as integer codes
    config_cols_present = [c for c in _CONFIG_AXES if c in df.columns]
    for col in config_cols_present:
        df[col] = pd.Categorical(df[col])
        df[f"{col}_code"] = df[col].cat.codes

    code_cols = [f"{c}_code" for c in config_cols_present]

    all_feature_cols = feature_cols_to_use + qt_cols + code_cols
    X = df[all_feature_cols].copy()
    y = df[target]

    return X, y


# ---------------------------------------------------------------------------
# Meta-json persistence
# ---------------------------------------------------------------------------

def _save_meta(
    save_dir: Path,
    *,
    mode: str,
    target: str,
    objective: str,
    constraints: dict[str, str] | None,
    features: list[str],
) -> None:
    """Write ``meta.json`` alongside model artifacts for inference-time use.

    Args:
        save_dir: Directory for model artifacts.
        mode: ``"classification"`` or ``"regression"``.
        target: Target column name.
        objective: ``"maximize"`` or ``"minimize"``.
        constraints: Constraint dict (may be ``None``).
        features: Feature column names used during training.
    """
    meta = {
        "mode": mode,
        "target": target,
        "objective": objective,
        "constraints": constraints,
        "features": features,
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Internal training helpers
# ---------------------------------------------------------------------------

def _train_classifier(
    df: pd.DataFrame,
    save_dir: Path,
    target: str,
    objective: str,
    features: list[str] | None,
    constraints: dict[str, str] | None,
    use_legacy_prepare: bool = False,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> dict:
    """Train an XGBClassifier to predict the best config label.

    Args:
        df: Experiment results DataFrame.
        save_dir: Directory for model artifacts.
        target: Column used for winner selection.
        objective: ``"maximize"`` or ``"minimize"``.
        features: Feature column names.
        constraints: Constraint dict.
        use_legacy_prepare: If True, use the original prepare_data() for
            backward compatibility with the threshold-based winner selection.
        quality_threshold: Quality threshold for legacy prepare_data path.

    Returns:
        Dict with evaluation metrics including ``f1_weighted``.
    """
    # Apply constraints
    if constraints:
        df = _apply_constraints(df, constraints)
        if df.empty:
            raise ValueError(
                f"No rows remain after applying constraints: {constraints}"
            )

    if use_legacy_prepare:
        # Backward compat: use original threshold-based winner selection
        X, y = prepare_data(df, quality_threshold=quality_threshold)
    else:
        X, y = _prepare_classification_data(df, target=target, objective=objective, features=features)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Handle edge cases: too few classes or classes with only 1 member
    # — sklearn's stratified split requires at least 2 members per class.
    n_classes = len(le.classes_)
    class_counts = pd.Series(y_encoded).value_counts()
    min_count = class_counts.min() if len(class_counts) > 0 else 0
    can_stratify = n_classes >= 2 and min_count >= 2

    if not can_stratify:
        print(
            f"WARNING: Cannot stratify ({n_classes} class(es), "
            f"min count per class={min_count}) — using non-stratified split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
        )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=n_classes,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    all_labels = list(range(n_classes))
    report = classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=le.classes_, output_dict=True, zero_division=0,
    )
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Save model artifacts
    model.save_model(str(save_dir / "xgb_meta_learner.json"))
    with open(save_dir / "label_encoder.json", "w", encoding="utf-8") as fh:
        json.dump({"classes": le.classes_.tolist()}, fh)
    with open(save_dir / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump({"columns": list(X.columns)}, fh)

    # Save meta.json for predict-time mode detection
    _save_meta(
        save_dir,
        mode="classification",
        target=target,
        objective=objective,
        constraints=constraints,
        features=features if features is not None else list(FEATURE_COLS),
    )

    print(f"Weighted F1: {f1:.3f}")
    print(classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=le.classes_, zero_division=0,
    ))

    return {"f1_weighted": f1, "report": report}


def _train_regressor(
    df: pd.DataFrame,
    save_dir: Path,
    target: str,
    features: list[str] | None,
    constraints: dict[str, str] | None,
) -> dict:
    """Train an XGBRegressor to predict a numeric target.

    Args:
        df: Experiment results DataFrame.
        save_dir: Directory for model artifacts.
        target: Numeric column to predict.
        features: Feature column names.
        constraints: Constraint dict.

    Returns:
        Dict with evaluation metrics including ``r2``, ``rmse``, ``mae``.
    """
    # Apply constraints
    if constraints:
        df = _apply_constraints(df, constraints)
        if df.empty:
            raise ValueError(
                f"No rows remain after applying constraints: {constraints}"
            )

    X, y = _prepare_regression_data(df, target=target, features=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Save model artifacts
    model.save_model(str(save_dir / "xgb_meta_learner.json"))
    with open(save_dir / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump({"columns": list(X.columns)}, fh)

    # Save meta.json
    _save_meta(
        save_dir,
        mode="regression",
        target=target,
        objective="minimize",  # regression doesn't use objective but record for completeness
        constraints=constraints,
        features=features if features is not None else list(FEATURE_COLS),
    )

    print(f"R²: {r2:.3f}  RMSE: {rmse:.3f}  MAE: {mae:.3f}")

    return {"r2": float(r2), "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    data,
    save_dir: Path | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    *,
    target: str = "quality",
    mode: Optional[str] = None,
    objective: str = "maximize",
    features: Optional[list[str]] = None,
    constraints: Optional[dict[str, str]] = None,
) -> dict:
    """Train the XGBoost meta-learner.

    Unified entry point that dispatches to classification or regression
    based on ``mode`` (or auto-detected from ``target``).

    Args:
        data: ExperimentResult, DataFrame, or Path to Parquet file.
        save_dir: Where to save model artifacts.  Defaults to ``models/``.
        quality_threshold: Legacy param — mapped to
            ``constraints={"quality": f">={quality_threshold}"}`` when no
            explicit constraints are provided and mode is classification.
        target: Column to optimize.  ``"config"`` triggers legacy
            classification.  A numeric column auto-detects regression
            when *mode* is ``None``.
        mode: ``"classification"``, ``"regression"``, or ``None`` (auto).
        objective: ``"maximize"`` or ``"minimize"`` (classification only).
        features: List of X column names.  Defaults to :data:`FEATURE_COLS`.
        constraints: Dict of column → filter expression, e.g.
            ``{"quality": ">3.0", "model_param_billions": "<4"}``.

    Returns:
        Dict with evaluation metrics.

    Raises:
        ValueError: If target column is missing, features list is empty,
            or constraints filter out all rows.
    """
    # --- Input coercion ---
    if isinstance(data, (Path, str)):
        df = pd.read_parquet(data)
    elif hasattr(data, "df"):
        df = data.df
    else:
        df = data

    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate features ---
    if features is not None and len(features) == 0:
        raise ValueError("features list cannot be empty")

    # --- Validate target ---
    # Special case: target="config" means legacy classification
    if target == "config":
        # Build the config column so the user doesn't have to
        if "config" not in df.columns:
            df = df.copy()
            df["config"] = (
                df["chunker"] + "__" + df["embedder"] + "__"
                + df["strategy"] + "__" + df["model"]
            )
        # Use quality for winner selection (backward compat)
        return _train_classifier(
            df, save_dir, target="quality", objective=objective,
            features=features, constraints=constraints,
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Auto-detect mode ---
    # When target="quality" (the default) and no explicit mode, default to
    # classification for backward compatibility — the original train(df) call
    # always produced a classifier.  Other numeric targets auto-detect regression.
    if mode is None:
        if target == "quality":
            # Backward compat: default target always means classification
            mode = "classification"
        elif pd.api.types.is_numeric_dtype(df[target]):
            mode = "regression"
        else:
            mode = "classification"

    # --- Backward compat: quality_threshold → constraints ---
    # Only apply when using the default path (no explicit constraints,
    # classification mode, quality target, and threshold differs from default)
    if (
        mode == "classification"
        and constraints is None
        and quality_threshold != DEFAULT_QUALITY_THRESHOLD
    ):
        constraints = {"quality": f">={quality_threshold}"}

    # --- Dispatch ---
    # Use the legacy prepare_data path when called with defaults (backward compat).
    # The old prepare_data uses threshold-based winner selection which differs from
    # the new _prepare_classification_data (pure sort-based).
    _is_default_path = (
        target == "quality"
        and mode == "classification"
        and constraints is None
        and features is None
        and objective == "maximize"
    )
    if mode == "classification":
        return _train_classifier(
            df, save_dir, target=target, objective=objective,
            features=features, constraints=constraints,
            use_legacy_prepare=_is_default_path,
            quality_threshold=quality_threshold,
        )
    elif mode == "regression":
        return _train_regressor(
            df, save_dir, target=target,
            features=features, constraints=constraints,
        )
    else:
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'classification', 'regression', or None.")
