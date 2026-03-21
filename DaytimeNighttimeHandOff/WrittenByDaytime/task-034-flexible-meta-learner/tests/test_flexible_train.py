"""Tests for the flexible meta-learner training API (task-034).

These tests extend the existing test_model.py with new test classes for
regression mode, constraints, auto-detection, and backward compatibility.

Append these classes to tests/test_model.py after the existing test classes.
The _make_synthetic_df() fixture from test_model.py is reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper — synthetic data with latency and model metadata columns
# ---------------------------------------------------------------------------

def _make_synthetic_df_with_latency() -> pd.DataFrame:
    """Extend _make_synthetic_df with latency and model_param_billions columns.

    Adds total_latency_ms (correlated with model size) and model_param_billions
    (parsed from model name) to enable regression and constrained optimization
    tests.
    """
    rng = np.random.RandomState(42)
    feat_rng = np.random.RandomState(99)

    docs = [f"doc_{i}" for i in range(5)]
    queries = [f"question {i}" for i in range(8)]
    chunkers = ["FixedSizeChunker", "SemanticChunker"]
    embedders = ["OllamaEmbedder", "HuggingFaceEmbedder"]
    strategies = ["NaiveRAG", "SelfRAG"]
    models = ["qwen3:0.6b", "qwen3:4b"]
    query_types = ["lookup", "synthesis", "multi_hop"]

    model_params = {"qwen3:0.6b": 0.6, "qwen3:4b": 4.0}
    model_latency_base = {"qwen3:0.6b": 200, "qwen3:4b": 800}

    config_bonus = {
        ("FixedSizeChunker", "OllamaEmbedder", "NaiveRAG", "qwen3:0.6b"): 2.0,
        ("SemanticChunker", "HuggingFaceEmbedder", "SelfRAG", "qwen3:4b"): 1.8,
        ("FixedSizeChunker", "HuggingFaceEmbedder", "SelfRAG", "qwen3:0.6b"): 1.5,
        ("SemanticChunker", "OllamaEmbedder", "NaiveRAG", "qwen3:4b"): 1.3,
    }

    rows = []
    for doc in docs:
        for query in queries:
            qt = rng.choice(query_types)
            feats = {
                "query_length": feat_rng.randint(3, 20),
                "num_named_entities": feat_rng.randint(0, 5),
                "doc_length": feat_rng.randint(100, 2000),
                "doc_vocab_entropy": feat_rng.uniform(3.0, 8.0),
                "doc_ner_density": feat_rng.uniform(0.0, 20.0),
                "doc_ner_repetition": feat_rng.uniform(0.0, 5.0),
                "doc_topic_count": feat_rng.randint(1, 8),
                "doc_topic_density": feat_rng.uniform(0.0, 5.0),
                "doc_semantic_coherence": feat_rng.uniform(0.3, 0.95),
                "doc_readability_score": feat_rng.uniform(4.0, 16.0),
                "doc_embedding_spread": feat_rng.uniform(0.0, 1.0),
                "query_doc_similarity": feat_rng.uniform(-0.1, 0.9),
                "query_doc_lexical_overlap": feat_rng.uniform(0.0, 0.3),
                "mean_retrieval_score": feat_rng.uniform(0.3, 0.9),
                "var_retrieval_score": feat_rng.uniform(0.01, 0.2),
            }
            for chunker in chunkers:
                for embedder in embedders:
                    for strategy in strategies:
                        for model in models:
                            bonus = config_bonus.get(
                                (chunker, embedder, strategy, model), 0.0
                            )
                            latency = model_latency_base[model] + rng.uniform(-50, 50)
                            rows.append({
                                "doc_title": doc,
                                "query_text": query,
                                "query_type": qt,
                                "chunker": chunker,
                                "embedder": embedder,
                                "strategy": strategy,
                                "model": model,
                                "answer": f"answer for {query}",
                                "faithfulness": rng.uniform(1, 3) + bonus,
                                "relevance": rng.uniform(1, 3) + bonus,
                                "conciseness": rng.uniform(1, 3) + bonus,
                                "total_latency_ms": latency,
                                "model_param_billions": model_params[model],
                                **feats,
                            })

    df = pd.DataFrame(rows)
    df["quality"] = df[["faithfulness", "relevance", "conciseness"]].mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Regression mode
# ---------------------------------------------------------------------------

class TestRegressionMode:
    """Tests for regression training mode."""

    def test_regression_train_and_predict(self, tmp_path: Path) -> None:
        """Train a regression model on quality, predict returns a numeric value."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, target="quality", mode="regression", save_dir=tmp_path)

        # Regression metrics
        assert "r2" in result or "rmse" in result or "mae" in result
        # Model artifacts saved
        assert (tmp_path / "xgb_meta_learner.json").exists()
        assert (tmp_path / "meta.json").exists()

    def test_regression_saves_meta_json(self, tmp_path: Path) -> None:
        """Verify meta.json records regression mode and target."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        train(df, target="quality", mode="regression", save_dir=tmp_path)

        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)

        assert meta["mode"] == "regression"
        assert meta["target"] == "quality"

    def test_regression_on_latency(self, tmp_path: Path) -> None:
        """Train regression on total_latency_ms — a different target column."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, target="total_latency_ms", mode="regression", save_dir=tmp_path)

        assert (tmp_path / "xgb_meta_learner.json").exists()
        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["target"] == "total_latency_ms"

    def test_regression_predict_returns_numeric(self, tmp_path: Path) -> None:
        """Predict with a regression model returns a predicted numeric value."""
        from src.model.train import train
        import src.model.predict as pred_module

        df = _make_synthetic_df_with_latency()
        train(df, target="quality", mode="regression", save_dir=tmp_path)

        # Reset and reload
        pred_module._model = None
        pred_module._label_classes = None
        pred_module._feature_columns = None
        pred_module._load_model(tmp_path)

        features = {
            "query_length": 5,
            "num_named_entities": 2,
            "doc_length": 500,
            "doc_vocab_entropy": 5.5,
            "doc_ner_density": 8.0,
            "doc_ner_repetition": 2.0,
            "doc_topic_count": 3,
            "doc_topic_density": 1.5,
            "doc_semantic_coherence": 0.7,
            "doc_readability_score": 10.0,
            "doc_embedding_spread": 0.4,
            "query_doc_similarity": 0.6,
            "query_doc_lexical_overlap": 0.1,
            "mean_retrieval_score": 0.7,
            "var_retrieval_score": 0.05,
            "query_type": "lookup",
            "chunker": "FixedSizeChunker",
            "embedder": "OllamaEmbedder",
            "strategy": "NaiveRAG",
            "model": "qwen3:0.6b",
        }
        result = pred_module.predict(features)

        assert "predicted_value" in result
        assert isinstance(result["predicted_value"], float)
        assert "model_type" in result
        assert result["model_type"] == "regression"


# ---------------------------------------------------------------------------
# Classification with constraints
# ---------------------------------------------------------------------------

class TestClassificationWithConstraints:
    """Tests for classification mode with constraint filtering."""

    def test_quality_constraint(self, tmp_path: Path) -> None:
        """Constraints filter out low-quality rows before winner selection."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(
            df,
            target="quality",
            mode="classification",
            objective="maximize",
            constraints={"quality": ">2.0"},
            save_dir=tmp_path,
        )
        assert "f1_weighted" in result

    def test_minimize_objective(self, tmp_path: Path) -> None:
        """Classification with minimize picks smallest target value per group."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(
            df,
            target="model_param_billions",
            mode="classification",
            objective="minimize",
            constraints={"quality": ">2.0"},
            save_dir=tmp_path,
        )
        assert "f1_weighted" in result

    def test_constraint_filters_all_rows_raises(self, tmp_path: Path) -> None:
        """Constraints that filter ALL rows should raise ValueError."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        with pytest.raises(ValueError, match="No rows remain"):
            train(
                df,
                target="quality",
                mode="classification",
                constraints={"quality": ">999"},
                save_dir=tmp_path,
            )

    def test_string_equality_constraint(self, tmp_path: Path) -> None:
        """String equality constraint filters by categorical value."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(
            df,
            target="quality",
            mode="classification",
            constraints={"strategy": "==NaiveRAG"},
            save_dir=tmp_path,
        )
        assert "f1_weighted" in result

    def test_multiple_constraints(self, tmp_path: Path) -> None:
        """Multiple constraints are applied together (AND logic)."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(
            df,
            target="quality",
            mode="classification",
            constraints={"quality": ">2.0", "model_param_billions": "<=4"},
            save_dir=tmp_path,
        )
        assert "f1_weighted" in result


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:
    """Tests for mode auto-detection from target type."""

    def test_config_target_auto_classifies(self, tmp_path: Path) -> None:
        """target='config' auto-detects classification mode."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, target="config", save_dir=tmp_path)

        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["mode"] == "classification"

    def test_numeric_target_auto_regresses(self, tmp_path: Path) -> None:
        """Numeric target with no explicit mode auto-detects regression."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, target="total_latency_ms", save_dir=tmp_path)

        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["mode"] == "regression"

    def test_numeric_target_explicit_classification(self, tmp_path: Path) -> None:
        """Numeric target + explicit mode='classification' trains classifier."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(
            df,
            target="quality",
            mode="classification",
            save_dir=tmp_path,
        )

        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["mode"] == "classification"
        assert "f1_weighted" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestTrainErrors:
    """Tests for error handling in train()."""

    def test_invalid_target_column(self, tmp_path: Path) -> None:
        """Nonexistent target column raises ValueError."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        with pytest.raises(ValueError, match="not found in data"):
            train(df, target="nonexistent_column", save_dir=tmp_path)

    def test_invalid_constraint_operator(self, tmp_path: Path) -> None:
        """Malformed constraint raises ValueError."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        with pytest.raises(ValueError, match="Invalid constraint"):
            train(
                df,
                target="quality",
                constraints={"quality": "~3.0"},
                save_dir=tmp_path,
            )

    def test_empty_features_raises(self, tmp_path: Path) -> None:
        """Empty features list raises ValueError."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        with pytest.raises(ValueError, match="features.*cannot be empty"):
            train(df, target="quality", features=[], save_dir=tmp_path)


# ---------------------------------------------------------------------------
# meta.json persistence
# ---------------------------------------------------------------------------

class TestMetaJson:
    """Tests for meta.json round-trip persistence."""

    def test_meta_json_contains_all_fields(self, tmp_path: Path) -> None:
        """meta.json records mode, target, objective, constraints, features."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        train(
            df,
            target="quality",
            mode="classification",
            objective="maximize",
            constraints={"quality": ">2.0"},
            save_dir=tmp_path,
        )

        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)

        assert "mode" in meta
        assert "target" in meta
        assert "objective" in meta
        assert "constraints" in meta
        assert "features" in meta
        assert meta["mode"] == "classification"
        assert meta["target"] == "quality"
        assert meta["objective"] == "maximize"
        assert meta["constraints"] == {"quality": ">2.0"}

    def test_predict_without_meta_json_assumes_classification(self, tmp_path: Path) -> None:
        """When meta.json is missing, predict() assumes classification mode."""
        from src.model.train import train
        import src.model.predict as pred_module

        df = _make_synthetic_df_with_latency()
        train(df, target="quality", mode="classification", save_dir=tmp_path)

        # Delete meta.json to simulate old model
        (tmp_path / "meta.json").unlink()

        pred_module._model = None
        pred_module._label_classes = None
        pred_module._feature_columns = None
        pred_module._load_model(tmp_path)

        features = {
            "query_length": 5,
            "num_named_entities": 2,
            "doc_length": 500,
            "doc_vocab_entropy": 5.5,
            "doc_ner_density": 8.0,
            "doc_ner_repetition": 2.0,
            "doc_topic_count": 3,
            "doc_topic_density": 1.5,
            "doc_semantic_coherence": 0.7,
            "doc_readability_score": 10.0,
            "doc_embedding_spread": 0.4,
            "query_doc_similarity": 0.6,
            "query_doc_lexical_overlap": 0.1,
            "mean_retrieval_score": 0.7,
            "var_retrieval_score": 0.05,
            "query_type": "lookup",
        }
        result = pred_module.predict(features)

        # Should work as classification (backward compat)
        assert "chunker" in result
        assert "confidence" in result


# ---------------------------------------------------------------------------
# predict_all_configs
# ---------------------------------------------------------------------------

class TestPredictAllConfigs:
    """Tests for predict_all_configs() ranked config list."""

    def test_returns_ranked_list(self, tmp_path: Path) -> None:
        """predict_all_configs returns a list of configs sorted by confidence."""
        from src.model.train import train
        import src.model.predict as pred_module

        df = _make_synthetic_df_with_latency()
        train(df, target="quality", mode="classification", save_dir=tmp_path)

        pred_module._model = None
        pred_module._label_classes = None
        pred_module._feature_columns = None
        pred_module._load_model(tmp_path)

        features = {
            "query_length": 5,
            "num_named_entities": 2,
            "doc_length": 500,
            "doc_vocab_entropy": 5.5,
            "doc_ner_density": 8.0,
            "doc_ner_repetition": 2.0,
            "doc_topic_count": 3,
            "doc_topic_density": 1.5,
            "doc_semantic_coherence": 0.7,
            "doc_readability_score": 10.0,
            "doc_embedding_spread": 0.4,
            "query_doc_similarity": 0.6,
            "query_doc_lexical_overlap": 0.1,
            "mean_retrieval_score": 0.7,
            "var_retrieval_score": 0.05,
            "query_type": "lookup",
        }
        results = pred_module.predict_all_configs(features)

        assert isinstance(results, list)
        assert len(results) > 1

        # Each entry has config axes + confidence
        for r in results:
            assert "chunker" in r
            assert "embedder" in r
            assert "strategy" in r
            assert "model" in r
            assert "confidence" in r

        # Sorted by confidence descending
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Backward compatibility — existing train(df) call
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Verify train(df) with no extra args still works like before."""

    def test_default_train_produces_classification(self, tmp_path: Path) -> None:
        """train(df) with defaults produces a classification model."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, save_dir=tmp_path)

        assert "f1_weighted" in result
        assert (tmp_path / "xgb_meta_learner.json").exists()
        assert (tmp_path / "label_encoder.json").exists()
        assert (tmp_path / "feature_columns.json").exists()

    def test_quality_threshold_backward_compat(self, tmp_path: Path) -> None:
        """Old quality_threshold param still works, mapped to constraints."""
        from src.model.train import train

        df = _make_synthetic_df_with_latency()
        result = train(df, quality_threshold=2.0, save_dir=tmp_path)

        assert "f1_weighted" in result
