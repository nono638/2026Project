# Task 005: Wire Meta-Learner and FastAPI to New Architecture

**Depends on:** task-002 (migrated components), task-004 (ExperimentResult)

## What
Update `src/model/train.py`, `src/model/predict.py`, and `src/app.py` to work with
the new pluggable architecture. The meta-learner now predicts across all four axes
(chunker, embedder, strategy, model) instead of just (strategy, model).

## Why
The meta-learner and API are the "product" layer sitting on top of the research tool.
They need to consume ExperimentResult data and predict the optimal full configuration
for a given query and document.

## Exact Changes

### `src/model/train.py`

**Key changes:**
1. Accept `ExperimentResult` or Parquet path (not just CSV)
2. The config label is now `"chunker__embedder__strategy__model"` (4 axes)
3. Feature columns unchanged — they're query/doc properties, not config properties

```python
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb


FEATURE_COLS = [
    "query_length",
    "num_named_entities",
    "doc_length",
    "doc_vocab_entropy",
    "mean_retrieval_score",
    "var_retrieval_score",
]

DEFAULT_QUALITY_THRESHOLD = 3.0


def prepare_data(df: pd.DataFrame,
                 quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from experiment results.

    Target: minimum viable config (cheapest that clears quality threshold).
    Config is now: chunker__embedder__strategy__model (4 axes).
    """
    if "quality" not in df.columns:
        score_cols = ["faithfulness", "relevance", "conciseness"]
        existing = [c for c in score_cols if c in df.columns]
        if existing:
            df["quality"] = df[existing].mean(axis=1)
        else:
            raise ValueError("No score columns found in data")

    df["config"] = (df["chunker"] + "__" + df["embedder"] + "__" +
                    df["strategy"] + "__" + df["model"])

    targets = []
    feature_rows = []

    for (query, doc_title), group in df.groupby(["query_text", "doc_title"]):
        viable = group[group["quality"] >= quality_threshold]
        if viable.empty:
            best = group.loc[group["quality"].idxmax()]
        else:
            # Sort by a cost heuristic: prefer simpler strategy, smaller model
            # This is approximate — could be refined with actual timing data
            best = viable.sort_values("quality", ascending=False).iloc[0]

        targets.append(best["config"])
        feature_rows.append(best[FEATURE_COLS + ["query_type"]].to_dict())

    features_df = pd.DataFrame(feature_rows)
    features_df = pd.get_dummies(features_df, columns=["query_type"], prefix="qt")

    return features_df, pd.Series(targets, name="config")


def train(data, save_dir: Path = None,
          quality_threshold: float = DEFAULT_QUALITY_THRESHOLD) -> dict:
    """Train the XGBoost meta-learner.

    Args:
        data: ExperimentResult, DataFrame, or Path to Parquet file.
        save_dir: Where to save model artifacts.
        quality_threshold: Minimum quality score for "acceptable" answer.

    Returns:
        Dict with evaluation metrics.
    """
    # Accept multiple input types
    if isinstance(data, Path) or isinstance(data, str):
        df = pd.read_parquet(data)
    elif hasattr(data, "df"):
        # ExperimentResult
        df = data.df
    else:
        df = data

    if save_dir is None:
        save_dir = Path("models")
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
    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                   output_dict=True)
    f1 = f1_score(y_test, y_pred, average="weighted")

    model.save_model(str(save_dir / "xgb_meta_learner.json"))
    with open(save_dir / "label_encoder.json", "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    with open(save_dir / "feature_columns.json", "w") as f:
        json.dump({"columns": list(X.columns)}, f)

    print(f"Weighted F1: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {"f1_weighted": f1, "report": report}
```

### `src/model/predict.py`

**Key change:** config now splits into 4 parts, not 2.

```python
def predict(features: dict) -> dict:
    """Predict the optimal config for given features.

    Returns:
        Dict with 'chunker', 'embedder', 'strategy', 'model', 'confidence'.
    """
    if _model is None:
        _load_model()

    row = {}
    for col in _feature_columns:
        if col.startswith("qt_"):
            query_type = features.get("query_type", "lookup")
            row[col] = 1.0 if col == f"qt_{query_type}" else 0.0
        else:
            row[col] = features.get(col, 0.0)

    X = pd.DataFrame([row])
    pred_idx = _model.predict(X)[0]
    probas = _model.predict_proba(X)[0]

    config = _label_classes[pred_idx]
    parts = config.split("__")
    # config format: chunker__embedder__strategy__model
    chunker, embedder, strategy, model = parts[0], parts[1], parts[2], parts[3]

    return {
        "chunker": chunker,
        "embedder": embedder,
        "strategy": strategy,
        "model": model,
        "confidence": float(probas[pred_idx]),
    }
```

### `src/app.py`

**Key changes:** response now includes all 4 axes. The endpoint still takes a document
and question, chunks/embeds/extracts features, and returns a recommendation.

The app needs to know which chunker and embedder to use for feature extraction at
inference time. Default to the most common ones from training, or accept them as
optional request parameters.

```python
from fastapi import FastAPI
from pydantic import BaseModel

from src.chunkers import SemanticChunker
from src.embedders import OllamaEmbedder
from src.retriever import Retriever
from src.features import extract_features
from src.model.predict import predict


app = FastAPI(
    title="SmallModelBigStrategy",
    description="Predicts the optimal RAG configuration for your query.",
)


class QueryRequest(BaseModel):
    document: str
    question: str
    query_type: str = "lookup"


class RecommendationResponse(BaseModel):
    chunker: str
    embedder: str
    strategy: str
    model: str
    confidence: float


# Default components for inference-time feature extraction
_default_chunker = None
_default_embedder = None

def _get_defaults():
    global _default_chunker, _default_embedder
    if _default_chunker is None:
        _default_chunker = SemanticChunker()
        _default_embedder = OllamaEmbedder()
    return _default_chunker, _default_embedder


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: QueryRequest):
    chunker, embedder = _get_defaults()
    chunks = chunker.chunk(req.document)
    retriever = Retriever(chunks, embedder)
    features = extract_features(req.question, req.document, retriever)
    features["query_type"] = req.query_type
    result = predict(features)
    return RecommendationResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
```

## Tests to Add: `tests/test_model.py`

Use a synthetic DataFrame (similar to test_analysis.py fixture) to test training:

1. `test_prepare_data` — verify features and targets are extracted correctly
2. `test_train_runs` — train on synthetic data, verify model files are saved
3. `test_predict_returns_4_axes` — load saved model, predict, verify response has
   chunker, embedder, strategy, model, confidence keys
4. `test_config_label_format` — verify the `__` separator works for 4-part configs

## What NOT to Touch
- `src/protocols.py`, `src/retriever.py`, `src/experiment.py`
- `src/chunkers/`, `src/embedders/`, `src/strategies/`, `src/scorers/`
- `src/features.py`
