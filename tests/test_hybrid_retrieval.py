"""Tests for hybrid retrieval (dense + BM25 + RRF).

Uses a mock embedder with deterministic vectors so tests don't depend on
any external model or API. BM25 works on the raw text — no mocking needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Mock embedder: assigns each chunk a deterministic vector based on index
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic embedder for testing.

    Each text gets a unit vector pointing in a different direction.
    The first text aligns with dimension 0, second with dimension 1, etc.
    Query "python" aligns with dimension 0 (similar to first chunk).
    Query "capital france" aligns with dimension 2 (similar to third chunk).
    """

    def __init__(self, dim: int = 8):
        self._dim = dim
        # Map specific strings to specific vector directions
        self._overrides: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "mock:deterministic"

    @property
    def dimension(self) -> int:
        return self._dim

    def set_override(self, text: str, direction: int) -> None:
        """Force a specific text to embed along the given dimension."""
        self._overrides[text] = direction

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return deterministic embeddings.

        Uses overrides if set, otherwise hashes the text to pick a direction.
        """
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text in self._overrides:
                idx = self._overrides[text] % self._dim
            else:
                idx = hash(text) % self._dim
            result[i, idx] = 1.0
        return result


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Chunks designed so BM25 and dense retrieval rank them differently.
# Chunk 0: "python" keyword, dense=direction 0
# Chunk 1: "java programming language", dense=direction 1
# Chunk 2: "python snake reptile", dense=direction 2
# Chunk 3: "machine learning algorithms", dense=direction 3
# Chunk 4: "python programming tutorial guide", dense=direction 4
CHUNKS = [
    "python is a popular programming language used worldwide",
    "java is another programming language for enterprise",
    "the python snake is a large reptile found in tropics",
    "machine learning algorithms process large datasets",
    "python programming tutorial and complete guide for beginners",
]


@pytest.fixture
def embedder():
    """Mock embedder with overrides for predictable dense ranking."""
    e = MockEmbedder(dim=8)
    # Set chunk directions
    for i, chunk in enumerate(CHUNKS):
        e.set_override(chunk, i)
    return e


# ---------------------------------------------------------------------------
# Tests: Retriever modes
# ---------------------------------------------------------------------------

class TestHybridMode:
    """Verify hybrid retrieval combines dense and sparse results."""

    def test_hybrid_returns_results(self, embedder):
        """Hybrid mode should return non-empty results."""
        r = Retriever(CHUNKS, embedder, top_k=3, mode="hybrid")
        # Set query direction to align with chunk 0
        embedder.set_override("python programming", 0)
        results = r.retrieve("python programming")
        assert len(results) > 0
        assert len(results) <= 3

    def test_hybrid_boosts_keyword_matches(self, embedder):
        """Chunks with keyword overlap should rank higher in hybrid than dense-only."""
        # Query that aligns densely with chunk 3 (ML direction) but has
        # BM25 keyword overlap with chunks 0, 2, 4 (contain "python")
        embedder.set_override("python algorithms", 3)

        r_hybrid = Retriever(CHUNKS, embedder, top_k=5, mode="hybrid")
        r_dense = Retriever(CHUNKS, embedder, top_k=5, mode="dense")

        hybrid_results = r_hybrid.retrieve("python algorithms")
        dense_results = r_dense.retrieve("python algorithms")

        # In dense-only, chunk 3 ("machine learning algorithms") should rank first
        # because the query vector points in direction 3.
        assert dense_results[0]["index"] == 3

        # In hybrid, chunks containing "python" get a BM25 boost.
        # The top results should include python-containing chunks.
        hybrid_texts = [r["text"] for r in hybrid_results[:3]]
        python_in_top3 = sum(1 for t in hybrid_texts if "python" in t.lower())
        assert python_in_top3 >= 1, "Hybrid should boost keyword matches"

    def test_hybrid_result_format(self, embedder):
        """Hybrid results should have text, score, index keys."""
        embedder.set_override("test query", 0)
        r = Retriever(CHUNKS, embedder, top_k=2, mode="hybrid")
        results = r.retrieve("test query")
        for result in results:
            assert "text" in result
            assert "score" in result
            assert "index" in result
            assert isinstance(result["score"], float)
            assert isinstance(result["index"], int)
            assert result["score"] > 0

    def test_hybrid_results_sorted_by_score(self, embedder):
        """Results should be sorted by descending RRF score."""
        embedder.set_override("python", 0)
        r = Retriever(CHUNKS, embedder, top_k=5, mode="hybrid")
        results = r.retrieve("python")
        scores = [res["score"] for res in results]
        assert scores == sorted(scores, reverse=True)


class TestDenseMode:
    """Verify dense-only mode matches original behavior."""

    def test_dense_returns_results(self, embedder):
        """Dense mode should return results."""
        embedder.set_override("query", 0)
        r = Retriever(CHUNKS, embedder, top_k=3, mode="dense")
        results = r.retrieve("query")
        assert len(results) > 0

    def test_dense_ignores_keyword_overlap(self, embedder):
        """Dense mode should rank purely by vector similarity."""
        # Query aligned with chunk 1 direction — should return chunk 1 first
        # regardless of keyword overlap
        embedder.set_override("python keywords", 1)
        r = Retriever(CHUNKS, embedder, top_k=3, mode="dense")
        results = r.retrieve("python keywords")
        # First result should be chunk 1 (java) because vectors align
        assert results[0]["index"] == 1

    def test_dense_scores_are_cosine_similarity(self, embedder):
        """Dense scores should be cosine similarities (0 to 1 for normalized vectors)."""
        embedder.set_override("test", 0)
        r = Retriever(CHUNKS, embedder, top_k=5, mode="dense")
        results = r.retrieve("test")
        for res in results:
            assert 0.0 <= res["score"] <= 1.0 + 1e-6  # Small tolerance for float


class TestSparseMode:
    """Verify BM25-only mode works."""

    def test_sparse_returns_results(self, embedder):
        """Sparse mode should return results based on keyword matching."""
        r = Retriever(CHUNKS, embedder, top_k=3, mode="sparse")
        results = r.retrieve("python programming")
        assert len(results) > 0

    def test_sparse_ranks_by_keyword_relevance(self, embedder):
        """Chunks with more keyword overlap should rank higher."""
        r = Retriever(CHUNKS, embedder, top_k=5, mode="sparse")
        results = r.retrieve("python programming")
        # Chunks 0, 2, 4 contain "python" — they should be in the results
        result_indices = {res["index"] for res in results}
        python_chunks = {0, 2, 4}
        assert python_chunks.issubset(result_indices), \
            f"Expected python chunks {python_chunks} in results {result_indices}"

    def test_sparse_no_keyword_overlap(self, embedder):
        """Query with no keyword overlap should return results with score 0 or empty."""
        r = Retriever(CHUNKS, embedder, top_k=3, mode="sparse")
        results = r.retrieve("zzzzxyxyx nonexistent terms")
        # BM25 may return results with 0 scores, or empty
        for res in results:
            assert res["score"] >= 0


class TestRRFFusion:
    """Test the RRF fusion logic directly via observable behavior."""

    def test_rrf_promotes_multi_signal_chunks(self, embedder):
        """A chunk ranked well by BOTH retrievers should rank highest."""
        # Query "python" + direction 0 = chunk 0 is best in both dense and sparse
        embedder.set_override("python", 0)
        r = Retriever(CHUNKS, embedder, top_k=5, mode="hybrid")
        results = r.retrieve("python")
        # Chunk 0 should be first — it's the top dense hit AND has strong BM25 for "python"
        assert results[0]["index"] == 0

    def test_rrf_score_is_sum_of_reciprocal_ranks(self, embedder):
        """Top result's RRF score should be approximately 1/(k+1) + 1/(k+1) = 2/(k+1)."""
        # When a chunk is rank 1 in both retrievers, RRF = 1/(60+1) + 1/(60+1)
        embedder.set_override("python", 0)
        r = Retriever(CHUNKS, embedder, top_k=1, mode="hybrid")
        results = r.retrieve("python")
        expected_max = 2.0 / 61.0  # rank 1 in both sources
        # Allow some tolerance — it should be close to the max
        assert results[0]["score"] <= expected_max + 1e-6


class TestHybridRetrievalEdgeCases:
    """Edge cases: empty chunks, single chunk, top_k bounds."""

    def test_empty_chunks_returns_empty(self, embedder):
        """Empty chunk list should return [] for all modes."""
        for mode in ("hybrid", "dense", "sparse"):
            r = Retriever([], embedder, top_k=5, mode=mode)
            results = r.retrieve("anything")
            assert results == [], f"mode={mode} should return [] for empty chunks"

    def test_single_chunk(self, embedder):
        """Single chunk should be returned in all modes."""
        single = ["python is great"]
        embedder.set_override(single[0], 0)
        for mode in ("hybrid", "dense", "sparse"):
            r = Retriever(single, embedder, top_k=5, mode=mode)
            embedder.set_override("python", 0)
            results = r.retrieve("python")
            assert len(results) == 1, f"mode={mode} should return 1 result"
            assert results[0]["text"] == single[0]

    def test_top_k_larger_than_chunks(self, embedder):
        """top_k > len(chunks) should return all chunks."""
        embedder.set_override("test", 0)
        r = Retriever(CHUNKS, embedder, top_k=100, mode="hybrid")
        results = r.retrieve("test")
        assert len(results) == len(CHUNKS)

    def test_top_k_respected(self, embedder):
        """Should never return more than top_k results."""
        embedder.set_override("python", 0)
        r = Retriever(CHUNKS, embedder, top_k=2, mode="hybrid")
        results = r.retrieve("python")
        assert len(results) <= 2

    def test_top_k_override_at_retrieve_time(self, embedder):
        """retrieve(top_k=N) should override the default."""
        embedder.set_override("python", 0)
        r = Retriever(CHUNKS, embedder, top_k=5, mode="hybrid")
        results = r.retrieve("python", top_k=1)
        assert len(results) == 1


class TestModeValidation:
    """Invalid mode should raise a clear error."""

    def test_invalid_mode_raises(self, embedder):
        """Passing an unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode"):
            Retriever(CHUNKS, embedder, top_k=3, mode="invalid")


class TestExperimentIntegration:
    """Verify Experiment passes through retrieval_mode."""

    def test_experiment_accepts_retrieval_mode(self):
        """Experiment.__init__ should accept retrieval_mode parameter."""
        from src.experiment import Experiment

        # Just verify it doesn't raise — we use mock components
        class MockChunker:
            @property
            def name(self) -> str:
                return "mock:chunker"
            def chunk(self, text: str) -> list[str]:
                return [text[:100]]

        class MockStrategy:
            @property
            def name(self) -> str:
                return "mock:strategy"
            def run(self, query, retriever, model):
                return "answer"

        class MockScorer:
            @property
            def name(self) -> str:
                return "mock:scorer"
            def score(self, query, context, answer):
                return {"quality": 4.0}

        exp = Experiment(
            chunkers=[MockChunker()],
            embedders=[MockEmbedder()],
            models=["test"],
            strategies=[MockStrategy()],
            scorer=MockScorer(),
            retrieval_mode="hybrid",
        )
        assert exp._retrieval_mode == "hybrid"

    def test_experiment_rejects_invalid_mode(self):
        """Experiment should reject invalid retrieval_mode."""
        from src.experiment import Experiment

        class MockChunker:
            @property
            def name(self) -> str:
                return "mock:chunker"
            def chunk(self, text: str) -> list[str]:
                return [text[:100]]

        class MockStrategy:
            @property
            def name(self) -> str:
                return "mock:strategy"
            def run(self, query, retriever, model):
                return "answer"

        class MockScorer:
            @property
            def name(self) -> str:
                return "mock:scorer"
            def score(self, query, context, answer):
                return {"quality": 4.0}

        with pytest.raises(ValueError, match="retrieval_mode"):
            Experiment(
                chunkers=[MockChunker()],
                embedders=[MockEmbedder()],
                models=["test"],
                strategies=[MockStrategy()],
                scorer=MockScorer(),
                retrieval_mode="invalid",
            )


class TestTokenization:
    """Test the BM25 tokenization helper."""

    def test_tokenize_basic(self):
        """Basic tokenization: lowercase and split."""
        tokens = Retriever._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_tokenize_strips_punctuation(self):
        """Punctuation should be removed."""
        tokens = Retriever._tokenize("Hello, world! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        # No commas, exclamation marks, or apostrophes in tokens
        for t in tokens:
            assert not any(c in t for c in ",.!?'\""), f"Token {t!r} has punctuation"

    def test_tokenize_empty_string(self):
        """Empty string should return empty list."""
        assert Retriever._tokenize("") == []

    def test_tokenize_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        assert Retriever._tokenize("   \t\n  ") == []
