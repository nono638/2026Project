"""Tests for extended feature columns (task-032).

Tests readability, embedding spread, query-document similarity, and
query-document lexical overlap features. Uses real HuggingFace embedder
for embedding-based tests (~80MB model, downloads on first run).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chunkers import FixedSizeChunker
from src.embedders import HuggingFaceEmbedder
from src.features import extract_features
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures — shared across test classes
# ---------------------------------------------------------------------------

EASY_TEXT = (
    "The cat sat on the mat. The dog ran in the park. "
    "The sun was bright. The sky was blue. Birds sang in the trees. "
) * 5

HARD_TEXT = (
    "Quantum chromodynamics describes the strong interaction between quarks "
    "and gluons, which are the fundamental constituents of hadrons such as "
    "protons and neutrons. The theory is characterized by asymptotic freedom "
    "and confinement, whereby the coupling constant decreases at short "
    "distances while increasing at large distances, preventing the isolation "
    "of individual color-charged particles. Perturbative calculations in QCD "
    "employ Feynman diagrams with gluon propagators and quark-gluon vertices. "
) * 3

MULTI_TOPIC_TEXT = (
    "Python is a high-level programming language used for web development. "
    "Machine learning algorithms identify patterns in training data. "
    "The Eiffel Tower in Paris attracts millions of visitors annually. "
    "Photosynthesis converts sunlight into chemical energy in plants. "
    "The stock market fluctuates based on supply and demand dynamics. "
    "Beethoven composed nine symphonies during his career. "
    "Quantum computing uses qubits to perform parallel computations. "
    "The Amazon rainforest contains the most biodiversity on Earth. "
) * 3


@pytest.fixture(scope="module")
def embedder():
    return HuggingFaceEmbedder()


def _make_retriever(text: str, embedder, chunk_size: int = 80):
    """Helper to build a retriever from text."""
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=10)
    chunks = chunker.chunk(text)
    return Retriever(chunks, embedder, top_k=3)


# ---------------------------------------------------------------------------
# Readability score
# ---------------------------------------------------------------------------

class TestReadabilityScore:
    """Tests for doc_readability_score feature (Flesch-Kincaid grade)."""

    def test_returns_float(self, embedder):
        """Readability score should be a float."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("What happened?", EASY_TEXT, retriever)
        assert isinstance(features["doc_readability_score"], float)

    def test_easy_text_lower_than_hard(self, embedder):
        """Simple text should have a lower FK grade than complex text."""
        r_easy = _make_retriever(EASY_TEXT, embedder)
        r_hard = _make_retriever(HARD_TEXT, embedder)

        f_easy = extract_features("What happened?", EASY_TEXT, r_easy)
        f_hard = extract_features("What happened?", HARD_TEXT, r_hard)

        assert f_easy["doc_readability_score"] < f_hard["doc_readability_score"]

    def test_empty_text_returns_zero(self, embedder):
        """Empty document should return 0.0 readability."""
        # Use a non-empty retriever but empty document text
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("test", "", retriever)
        assert features["doc_readability_score"] == 0.0


# ---------------------------------------------------------------------------
# Embedding spread
# ---------------------------------------------------------------------------

class TestEmbeddingSpread:
    """Tests for doc_embedding_spread feature (intra-cluster distance)."""

    def test_returns_float(self, embedder):
        """Embedding spread should be a float."""
        retriever = _make_retriever(MULTI_TOPIC_TEXT, embedder)
        features = extract_features("test", MULTI_TOPIC_TEXT, retriever)
        assert isinstance(features["doc_embedding_spread"], float)

    def test_non_negative(self, embedder):
        """Embedding spread should always be >= 0."""
        retriever = _make_retriever(MULTI_TOPIC_TEXT, embedder)
        features = extract_features("test", MULTI_TOPIC_TEXT, retriever)
        assert features["doc_embedding_spread"] >= 0.0

    def test_single_chunk_is_zero(self, embedder):
        """Single chunk should have 0 spread."""
        short_text = "This is a single short chunk."
        chunker = FixedSizeChunker(chunk_size=500, overlap=0)
        chunks = chunker.chunk(short_text)
        assert len(chunks) == 1
        retriever = Retriever(chunks, embedder, top_k=1)
        features = extract_features("test", short_text, retriever)
        assert features["doc_embedding_spread"] == 0.0

    def test_multi_topic_has_higher_spread(self, embedder):
        """Document covering many topics should have higher spread than
        a repetitive single-topic document."""
        single_topic = "Python is a programming language. " * 20
        r_single = _make_retriever(single_topic, embedder, chunk_size=60)
        r_multi = _make_retriever(MULTI_TOPIC_TEXT, embedder, chunk_size=60)

        f_single = extract_features("test", single_topic, r_single)
        f_multi = extract_features("test", MULTI_TOPIC_TEXT, r_multi)

        assert f_multi["doc_embedding_spread"] > f_single["doc_embedding_spread"]


# ---------------------------------------------------------------------------
# Query-document similarity (embedding)
# ---------------------------------------------------------------------------

class TestQueryDocSimilarity:
    """Tests for query_doc_similarity feature (embedding cosine sim)."""

    def test_returns_float(self, embedder):
        """Should return a float."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("What is a cat?", EASY_TEXT, retriever)
        assert isinstance(features["query_doc_similarity"], float)

    def test_range(self, embedder):
        """Should be in [-1, 1] range."""
        retriever = _make_retriever(MULTI_TOPIC_TEXT, embedder)
        features = extract_features("What is Python?", MULTI_TOPIC_TEXT, retriever)
        assert -1.0 <= features["query_doc_similarity"] <= 1.0

    def test_relevant_query_higher_than_irrelevant(self, embedder):
        """A query about the document's topic should have higher similarity
        than a completely unrelated query."""
        python_text = (
            "Python is a high-level programming language. "
            "It supports object-oriented and functional programming. "
            "Python is widely used for data science and web development. "
        ) * 5
        retriever = _make_retriever(python_text, embedder)

        f_relevant = extract_features(
            "What is Python used for?", python_text, retriever
        )
        f_irrelevant = extract_features(
            "What is the weather on Mars?", python_text, retriever
        )

        assert (f_relevant["query_doc_similarity"]
                > f_irrelevant["query_doc_similarity"])


# ---------------------------------------------------------------------------
# Query-document lexical overlap
# ---------------------------------------------------------------------------

class TestQueryDocLexicalOverlap:
    """Tests for query_doc_lexical_overlap feature (Jaccard word similarity)."""

    def test_returns_float(self, embedder):
        """Should return a float."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("cat mat", EASY_TEXT, retriever)
        assert isinstance(features["query_doc_lexical_overlap"], float)

    def test_range(self, embedder):
        """Should be in [0, 1] range."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("cat mat", EASY_TEXT, retriever)
        assert 0.0 <= features["query_doc_lexical_overlap"] <= 1.0

    def test_no_overlap_is_zero(self, embedder):
        """Query with no common words should give 0.0."""
        text = "alpha beta gamma delta epsilon"
        retriever = _make_retriever(text * 5, embedder, chunk_size=50)
        features = extract_features("xyz qqq zzz", text * 5, retriever)
        assert features["query_doc_lexical_overlap"] == 0.0

    def test_full_overlap(self, embedder):
        """Query identical to document should give 1.0."""
        text = "hello world"
        doc = " ".join([text] * 10)
        retriever = _make_retriever(doc, embedder, chunk_size=50)
        features = extract_features("hello world", doc, retriever)
        assert features["query_doc_lexical_overlap"] == 1.0

    def test_partial_overlap(self, embedder):
        """Some common words should give value between 0 and 1."""
        text = "Python is a programming language for data science"
        retriever = _make_retriever(text * 5, embedder, chunk_size=50)
        features = extract_features(
            "What is Python programming?", text * 5, retriever
        )
        overlap = features["query_doc_lexical_overlap"]
        assert 0.0 < overlap < 1.0

    def test_case_insensitive(self, embedder):
        """Overlap should be case-insensitive."""
        text = "Python Programming Language"
        doc = " ".join([text] * 10)
        retriever = _make_retriever(doc, embedder, chunk_size=50)
        features = extract_features(
            "python programming language", doc, retriever
        )
        assert features["query_doc_lexical_overlap"] == 1.0


# ---------------------------------------------------------------------------
# All features present in extract_features output
# ---------------------------------------------------------------------------

class TestExtractFeaturesIncludesNewKeys:
    """Verify extract_features returns all new feature keys."""

    def test_all_new_keys_present(self, embedder):
        """extract_features dict should contain all 4 new feature keys."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("What is a cat?", EASY_TEXT, retriever)

        new_keys = [
            "doc_readability_score",
            "doc_embedding_spread",
            "query_doc_similarity",
            "query_doc_lexical_overlap",
        ]
        for key in new_keys:
            assert key in features, f"Missing feature key: {key}"

    def test_existing_keys_still_present(self, embedder):
        """Existing features should not be removed."""
        retriever = _make_retriever(EASY_TEXT, embedder)
        features = extract_features("What is a cat?", EASY_TEXT, retriever)

        existing_keys = [
            "query_length", "num_named_entities",
            "doc_length", "doc_vocab_entropy",
            "doc_ner_density", "doc_ner_repetition",
            "doc_topic_count", "doc_topic_density", "doc_semantic_coherence",
            "mean_retrieval_score", "var_retrieval_score",
        ]
        for key in existing_keys:
            assert key in features, f"Missing existing key: {key}"
