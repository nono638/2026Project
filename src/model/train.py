"""XGBoost meta-learner training.

Trains a classifier to predict the optimal (chunker, embedder, strategy, model)
configuration from query/document features. The config label is now 4 axes
instead of the original 2 (strategy, model), reflecting the pluggable
architecture introduced in task-001/task-002.

Primary model: XGBoost
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

from src.config import DEFAULT_QUALITY_THRESHOLD, FEATURE_COLS, MODELS_DIR


def prepare_data(
    df: pd.DataFrame,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from experiment results.

    Target: minimum viable config (cheapest that clears quality threshold).
    Config is now: chunker__embedder__strategy__model (4 axes).

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
            df["quality"] = df[existing].mean(axis=1)
        else:
            raise ValueError("No score columns found in data")

    # 4-axis config label: chunker__embedder__strategy__model
    df["config"] = (
        df["chunker"] + "__" + df["embedder"] + "__" +
        df["strategy"] + "__" + df["model"]
    )

    targets: list[str] = []
    feature_rows: list[dict] = []

    for (query, doc_title), group in df.groupby(["query_text", "doc_title"]):
        viable = group[group["quality"] >= quality_threshold]
        if viable.empty:
            # Nothing clears threshold — use the best available
            best = group.loc[group["quality"].idxmax()]
        else:
            # Sort by quality descending, pick the best viable config
            # Could be refined with actual timing data for cost ordering
            best = viable.sort_values("quality", ascending=False).iloc[0]

        targets.append(best["config"])
        feature_rows.append(best[FEATURE_COLS + ["query_type"]].to_dict())

    features_df = pd.DataFrame(feature_rows)
    # One-hot encode query_type
    features_df = pd.get_dummies(features_df, columns=["query_type"], prefix="qt")

    return features_df, pd.Series(targets, name="config")


def train(
    data,
    save_dir: Path | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> dict:
    """Train the XGBoost meta-learner.

    Args:
        data: ExperimentResult, DataFrame, or Path to Parquet file.
        save_dir: Where to save model artifacts.
        quality_threshold: Minimum quality score for "acceptable" answer.

    Returns:
        Dict with evaluation metrics.
    """
    # Accept multiple input types for flexibility
    if isinstance(data, (Path, str)):
        df = pd.read_parquet(data)
    elif hasattr(data, "df"):
        # ExperimentResult — access its underlying DataFrame
        df = data.df
    else:
        df = data

    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y = prepare_data(df, quality_threshold)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(le.classes_),
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Pass labels explicitly so classification_report handles classes that
    # may not appear in the test set (common with small datasets or many classes)
    all_labels = list(range(len(le.classes_)))
    report = classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=le.classes_, output_dict=True, zero_division=0,
    )
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Save model artifacts for inference
    model.save_model(str(save_dir / "xgb_meta_learner.json"))
    with open(save_dir / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    with open(save_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"columns": list(X.columns)}, f)

    print(f"Weighted F1: {f1:.3f}")
    print(classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=le.classes_, zero_division=0,
    ))

    return {"f1_weighted": f1, "report": report}
