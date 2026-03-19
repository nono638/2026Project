"""Tests for document characterization features in src/features.py.

Tests NER density, NER repetition, topic count, topic density,
semantic coherence, and the full extract_features function.
All external dependencies (spaCy, Ollama) are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNERFeatures:
    """Tests for _ner_features() using mocked spaCy."""

    def _make_mock_nlp(self, entities: list[tuple[str, str]]):
        """Build a mock spaCy nlp that returns given entities.

        Args:
            entities: List of (text, label) tuples for mock entities.
        """
        mock_nlp = MagicMock()
        mock_nlp.max_length = 1_000_000

        mock_doc = MagicMock()
        mock_ents = []
        for text, label in entities:
            ent = MagicMock()
            ent.text = text
            ent.label_ = label
            mock_ents.append(ent)
        mock_doc.ents = mock_ents
        mock_nlp.return_value = mock_doc

        return mock_nlp

    @patch("src.features._get_spacy_nlp")
    def test_entity_density(self, mock_get_nlp: MagicMock) -> None:
        """NER density = distinct entities per 1000 tokens."""
        from src.features import _ner_features

        # 5 distinct entities in a 100-word doc = 50 per 1000 tokens
        mock_get_nlp.return_value = self._make_mock_nlp([
            ("Paris", "GPE"), ("France", "GPE"), ("Eiffel", "PERSON"),
            ("UNESCO", "ORG"), ("Europe", "LOC"),
        ])
        text = " ".join(["word"] * 100)
        density, _ = _ner_features(text)
        assert density == pytest.approx(50.0)

    @patch("src.features._get_spacy_nlp")
    def test_entity_repetition(self, mock_get_nlp: MagicMock) -> None:
        """NER repetition = total mentions / distinct count."""
        from src.features import _ner_features

        # "Paris" mentioned 3 times, "France" once = 4 total / 2 distinct = 2.0
        mock_get_nlp.return_value = self._make_mock_nlp([
            ("Paris", "GPE"), ("Paris", "GPE"), ("Paris", "GPE"),
            ("France", "GPE"),
        ])
        text = " ".join(["word"] * 100)
        _, repetition = _ner_features(text)
        assert repetition == pytest.approx(2.0)

    @patch("src.features._get_spacy_nlp")
    def test_no_entities(self, mock_get_nlp: MagicMock) -> None:
        """No entities → density 0, repetition 0."""
        from src.features import _ner_features

        mock_get_nlp.return_value = self._make_mock_nlp([])
        text = " ".join(["word"] * 50)
        density, repetition = _ner_features(text)
        assert density == 0.0
        assert repetition == 0.0

    @patch("src.features._get_spacy_nlp")
    def test_empty_text(self, mock_get_nlp: MagicMock) -> None:
        """Empty text → both features 0."""
        from src.features import _ner_features

        mock_get_nlp.return_value = self._make_mock_nlp([])
        density, repetition = _ner_features("")
        assert density == 0.0
        assert repetition == 0.0

    @patch("src.features._get_spacy_nlp")
    def test_case_insensitive_dedup(self, mock_get_nlp: MagicMock) -> None:
        """'Paris' and 'paris' should count as the same entity."""
        from src.features import _ner_features

        mock_get_nlp.return_value = self._make_mock_nlp([
            ("Paris", "GPE"), ("paris", "GPE"),
        ])
        text = " ".join(["word"] * 100)
        density, repetition = _ner_features(text)
        # 1 distinct entity → density = 1/100*1000 = 10
        assert density == pytest.approx(10.0)
        # 2 mentions / 1 distinct = 2.0
        assert repetition == pytest.approx(2.0)


class TestEmbeddingFeatures:
    """Tests for _embedding_features() using a mock retriever with real FAISS."""

    def _make_retriever_with_embeddings(self, embeddings: np.ndarray):
        """Build a mock retriever with a real FAISS index.

        Args:
            embeddings: (n, d) array of vectors to index.
        """
        import faiss

        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        retriever = MagicMock()
        retriever._index = index
        return retriever

    def test_single_chunk(self) -> None:
        """Single chunk → 1 topic, 1.0 coherence."""
        from src.features import _embedding_features

        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        retriever = self._make_retriever_with_embeddings(embeddings)
        topic_count, coherence = _embedding_features(retriever)
        assert topic_count == 1
        assert coherence == 1.0

    def test_two_identical_chunks(self) -> None:
        """Two identical embeddings → 1 topic, ~1.0 coherence."""
        from src.features import _embedding_features

        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        retriever = self._make_retriever_with_embeddings(embeddings)
        topic_count, coherence = _embedding_features(retriever)
        assert topic_count == 1
        assert coherence == pytest.approx(1.0, abs=0.01)

    def test_two_orthogonal_clusters(self) -> None:
        """Two well-separated clusters → 2 topics, low coherence."""
        from src.features import _embedding_features

        # Cluster A: vectors near [1,0,0,0], Cluster B: vectors near [0,0,1,0]
        embeddings = np.array([
            [1.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.1, 1.0, 0.0],
            [0.1, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        retriever = self._make_retriever_with_embeddings(embeddings)
        topic_count, coherence = _embedding_features(retriever)
        assert topic_count == 2

    def test_coherence_decreases_with_disorder(self) -> None:
        """Alternating between clusters gives lower coherence than sorted."""
        from src.features import _consecutive_cosine_mean

        # Sorted: A A A B B B (high consecutive similarity)
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        sorted_emb = np.array([a, a, a, b, b, b])
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(sorted_emb)

        # Alternating: A B A B A B (low consecutive similarity)
        alt_emb = np.array([a, b, a, b, a, b])
        alt_copy = alt_emb.copy()
        faiss_module.normalize_L2(alt_copy)

        sorted_coherence = _consecutive_cosine_mean(sorted_emb)
        alt_coherence = _consecutive_cosine_mean(alt_copy)
        assert sorted_coherence > alt_coherence


class TestEstimateTopicCount:
    """Tests for _estimate_topic_count()."""

    def test_separated_clusters_found(self) -> None:
        """Well-separated clusters → more topics than uniform data."""
        from src.features import _estimate_topic_count

        rng = np.random.RandomState(42)

        # Uniform: all near same direction in high-d (more realistic)
        base = np.zeros(64, dtype=np.float32)
        base[0] = 1.0
        uniform = base + rng.normal(0, 0.01, (10, 64)).astype(np.float32)

        # Separated: 3 clear clusters in different directions
        c1 = np.zeros(64); c1[0] = 1.0
        c2 = np.zeros(64); c2[20] = 1.0
        c3 = np.zeros(64); c3[40] = 1.0
        separated = np.array([
            c1 + rng.normal(0, 0.01, 64),
            c1 + rng.normal(0, 0.01, 64),
            c1 + rng.normal(0, 0.01, 64),
            c2 + rng.normal(0, 0.01, 64),
            c2 + rng.normal(0, 0.01, 64),
            c2 + rng.normal(0, 0.01, 64),
            c3 + rng.normal(0, 0.01, 64),
            c3 + rng.normal(0, 0.01, 64),
            c3 + rng.normal(0, 0.01, 64),
        ], dtype=np.float32)

        k_uniform = _estimate_topic_count(uniform)
        k_separated = _estimate_topic_count(separated)
        assert k_separated > k_uniform

    def test_two_chunks_returns_one(self) -> None:
        """Fewer than 3 samples → returns 1 (can't compute silhouette)."""
        from src.features import _estimate_topic_count

        embeddings = np.array([
            [1.0, 0.0], [0.0, 1.0],
        ], dtype=np.float32)
        assert _estimate_topic_count(embeddings) == 1


class TestExtractFeaturesIntegration:
    """Test that extract_features returns all expected keys."""

    @patch("src.features._get_spacy_nlp")
    def test_all_feature_keys_present(self, mock_get_nlp: MagicMock) -> None:
        """extract_features returns all 11 feature keys."""
        import faiss
        from src.features import extract_features

        # Mock spaCy
        mock_nlp = MagicMock()
        mock_nlp.max_length = 1_000_000
        mock_doc = MagicMock()
        ent = MagicMock()
        ent.text = "TestEntity"
        mock_doc.ents = [ent]
        mock_nlp.return_value = mock_doc
        mock_get_nlp.return_value = mock_nlp

        # Mock retriever with real FAISS
        embeddings = np.random.randn(5, 8).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(8)
        index.add(embeddings)

        retriever = MagicMock()
        retriever._index = index
        retriever.retrieve.return_value = [
            {"text": "chunk", "score": 0.5, "index": 0},
            {"text": "chunk", "score": 0.3, "index": 1},
        ]

        features = extract_features("test query", "some document text here", retriever)

        expected_keys = {
            "query_length", "num_named_entities",
            "doc_length", "doc_vocab_entropy",
            "doc_ner_density", "doc_ner_repetition",
            "doc_topic_count", "doc_topic_density", "doc_semantic_coherence",
            "mean_retrieval_score", "var_retrieval_score",
        }
        assert set(features.keys()) == expected_keys

    @patch("src.features._get_spacy_nlp")
    def test_feature_types_are_numeric(self, mock_get_nlp: MagicMock) -> None:
        """All feature values must be int or float (meta-learner requirement)."""
        import faiss
        from src.features import extract_features

        mock_nlp = MagicMock()
        mock_nlp.max_length = 1_000_000
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_get_nlp.return_value = mock_nlp

        embeddings = np.random.randn(4, 8).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(8)
        index.add(embeddings)

        retriever = MagicMock()
        retriever._index = index
        retriever.retrieve.return_value = [
            {"text": "c", "score": 0.5, "index": 0},
        ]

        features = extract_features("query", "doc text", retriever)
        for key, val in features.items():
            assert isinstance(val, (int, float)), f"{key} is {type(val)}, expected numeric"
