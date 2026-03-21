"""Tests for the meta-learner training and prediction pipeline.

Uses synthetic DataFrames (similar to test_analysis.py) to verify that the
4-axis config label format works end-to-end: prepare_data → train → predict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.model.train import prepare_data, train, FEATURE_COLS


def _make_synthetic_df() -> pd.DataFrame:
    """Create a synthetic experiment results DataFrame as a cartesian product.

    Generates data mimicking ExperimentResult.df — a full cartesian product of
    (doc × query × chunker × embedder × strategy × model). This ensures each
    (query, doc) group has rows for every config, and the "winning" config
    per group is deterministic. With few component options and many groups,
    each winning config appears multiple times — required for stratified split.

    Returns:
        DataFrame matching the experiment output format.
    """
    rng = np.random.RandomState(42)
    feat_rng = np.random.RandomState(99)  # separate RNG for features

    # Few configs (2×2×2×2=16) spread across many groups (5 docs × 8 queries = 40)
    # ensures each config "wins" in multiple groups
    docs = [f"doc_{i}" for i in range(5)]
    queries = [f"question {i}" for i in range(8)]
    chunkers = ["FixedSizeChunker", "SemanticChunker"]
    embedders = ["OllamaEmbedder", "HuggingFaceEmbedder"]
    strategies = ["NaiveRAG", "SelfRAG"]
    models = ["qwen3:0.6b", "qwen3:4b"]
    query_types = ["lookup", "synthesis", "multi_hop"]

    # Assign deterministic quality bonuses to configs so only a few configs
    # consistently "win" per group. This ensures each winning config appears
    # at least 2 times — required for sklearn's stratified train/test split.
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
                                **feats,
                            })

    df = pd.DataFrame(rows)
    df["quality"] = df[["faithfulness", "relevance", "conciseness"]].mean(axis=1)
    return df


class TestPrepareData:
    """Tests for the prepare_data function."""

    def test_prepare_data_returns_features_and_targets(self) -> None:
        """Verify prepare_data returns a feature DataFrame and target Series."""
        df = _make_synthetic_df()
        features, targets = prepare_data(df)

        assert isinstance(features, pd.DataFrame)
        assert isinstance(targets, pd.Series)
        assert len(features) == len(targets)
        # All FEATURE_COLS should be in the output (query_type gets one-hot encoded)
        for col in FEATURE_COLS:
            assert col in features.columns

    def test_config_label_format(self) -> None:
        """Verify config labels have 4 parts separated by '__'."""
        df = _make_synthetic_df()
        _, targets = prepare_data(df)

        for config in targets:
            parts = config.split("__")
            assert len(parts) == 4, f"Expected 4 parts in config, got {len(parts)}: {config}"
            # Each part should be non-empty
            for part in parts:
                assert len(part) > 0, f"Empty part in config: {config}"

    def test_quality_column_computed_if_missing(self) -> None:
        """Verify quality is computed from score columns when not present."""
        df = _make_synthetic_df()
        df = df.drop(columns=["quality"])
        features, targets = prepare_data(df)
        # Should succeed without error — quality is derived internally
        assert len(targets) > 0

    def test_query_type_one_hot_encoded(self) -> None:
        """Verify query_type columns are one-hot encoded with qt_ prefix."""
        df = _make_synthetic_df()
        features, _ = prepare_data(df)
        qt_cols = [c for c in features.columns if c.startswith("qt_")]
        assert len(qt_cols) > 0, "No one-hot query_type columns found"


class TestTrain:
    """Tests for the train function."""

    def test_train_runs_and_saves_model(self, tmp_path: Path) -> None:
        """Verify training completes and saves all 3 model artifact files."""
        df = _make_synthetic_df()
        result = train(df, save_dir=tmp_path)

        # Check return value
        assert "f1_weighted" in result
        assert "report" in result
        assert isinstance(result["f1_weighted"], float)

        # Check saved artifacts
        assert (tmp_path / "xgb_meta_learner.json").exists()
        assert (tmp_path / "label_encoder.json").exists()
        assert (tmp_path / "feature_columns.json").exists()

        # Verify label encoder has 4-part config labels
        with open(tmp_path / "label_encoder.json") as f:
            le_data = json.load(f)
        for cls in le_data["classes"]:
            assert len(cls.split("__")) == 4

    def test_train_accepts_experiment_result(self, tmp_path: Path) -> None:
        """Verify train accepts an object with a .df attribute (ExperimentResult)."""

        class FakeResult:
            """Mimics ExperimentResult with a .df attribute."""

            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df

        df = _make_synthetic_df()
        fake = FakeResult(df)
        result = train(fake, save_dir=tmp_path)
        assert "f1_weighted" in result


class TestPredict:
    """Tests for end-to-end train → predict flow."""

    def test_predict_returns_4_axes(self, tmp_path: Path) -> None:
        """Train a model, then predict and verify response has all 4 axes."""
        df = _make_synthetic_df()
        train(df, save_dir=tmp_path)

        # Reset the module-level cached model so it loads from tmp_path
        import src.model.predict as pred_module
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

        # Verify all expected keys are present
        assert "chunker" in result
        assert "embedder" in result
        assert "strategy" in result
        assert "model" in result
        assert "confidence" in result

        # Confidence should be a probability
        assert 0.0 <= result["confidence"] <= 1.0

        # Each axis should be a non-empty string
        for key in ["chunker", "embedder", "strategy", "model"]:
            assert isinstance(result[key], str)
            assert len(result[key]) > 0
