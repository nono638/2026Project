"""Tests for RoundTripFilter.

Uses simple test chunker and HuggingFaceEmbedder for real semantic similarity,
plus a hash-based embedder for tests that don't need meaningful similarity.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.document import Document
from src.protocols import QueryFilter
from src.query import Query
from src.query_filters.round_trip import RoundTripFilter


# ---------------------------------------------------------------------------
# Test helpers (not in src/ — for testing only)
# ---------------------------------------------------------------------------


class SimpleChunker:
    """Chunks by splitting on double newlines. For testing only."""

    @property
    def name(self) -> str:
        """Return test chunker name."""
        return "test:simple"

    def chunk(self, text: str) -> list[str]:
        """Split on double newlines, returning non-empty paragraphs."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class HashEmbedder:
    """Deterministic embedder using MD5 hashing. For testing only.

    Produces vectors that are deterministic but NOT semantically meaningful.
    Use only for tests that check structure, not retrieval quality.
    """

    @property
    def name(self) -> str:
        """Return test embedder name."""
        return "test:hash"

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return 64

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using MD5 hash for deterministic but meaningless vectors."""
        result = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hashlib.md5(text.encode()).digest()
            result[i, :16] = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(text: str, source: str = "Doc1") -> Query:
    """Create a minimal Query for testing."""
    return Query(text=text, query_type="factoid", source_doc_title=source)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundTripFilter:
    """Tests for the RoundTripFilter class."""

    def test_filter_keeps_good_query(self) -> None:
        """Query about document content retrieves its source and passes.

        Uses HuggingFaceEmbedder for real semantic similarity.
        """
        from src.embedders.huggingface import HuggingFaceEmbedder

        chunker = SimpleChunker()
        embedder = HuggingFaceEmbedder("all-MiniLM-L6-v2")

        doc = Document(
            title="Cooking",
            text=(
                "Pasta is made from flour and water.\n\n"
                "Italian cuisine is known for its pasta dishes.\n\n"
                "Tomato sauce originated in southern Italy."
            ),
        )
        # Query about pasta should retrieve chunks from the Cooking doc
        query = _q("What is pasta made from?", source="Cooking")

        f = RoundTripFilter(chunker, embedder, top_k=3)
        result = f.filter([query], [doc])
        assert len(result) == 1

    def test_filter_removes_bad_query(self) -> None:
        """Query unrelated to document content fails round-trip.

        Uses HuggingFaceEmbedder for real semantic similarity.
        """
        from src.embedders.huggingface import HuggingFaceEmbedder

        chunker = SimpleChunker()
        embedder = HuggingFaceEmbedder("all-MiniLM-L6-v2")

        doc_cooking = Document(
            title="Cooking",
            text=(
                "Pasta is made from flour and water.\n\n"
                "Italian cuisine is known for its pasta dishes."
            ),
        )
        doc_physics = Document(
            title="Physics",
            text=(
                "Quantum entanglement is a phenomenon in quantum mechanics.\n\n"
                "Electrons can be entangled across vast distances."
            ),
        )
        # Query about physics but source_doc_title says Cooking → should fail
        query = _q("What is quantum entanglement?", source="Cooking")

        f = RoundTripFilter(chunker, embedder, top_k=2)
        result = f.filter([query], [doc_cooking, doc_physics])
        assert len(result) == 0

    def test_filter_empty_queries(self) -> None:
        """Empty query list returns empty list."""
        f = RoundTripFilter(SimpleChunker(), HashEmbedder(), top_k=3)
        result = f.filter([], [Document(title="D", text="T")])
        assert result == []

    def test_filter_empty_documents(self) -> None:
        """Empty document list returns empty list."""
        f = RoundTripFilter(SimpleChunker(), HashEmbedder(), top_k=3)
        result = f.filter([_q("Q?")], [])
        assert result == []

    def test_filter_unknown_source_doc(self) -> None:
        """Query referencing unknown doc title is filtered out."""
        f = RoundTripFilter(SimpleChunker(), HashEmbedder(), top_k=3)
        doc = Document(title="KnownDoc", text="Some content here.\n\nMore content.")
        query = _q("What is this?", source="UnknownDoc")
        result = f.filter([query], [doc])
        assert len(result) == 0

    def test_name_format(self) -> None:
        """Name includes embedder name and top_k."""
        f = RoundTripFilter(SimpleChunker(), HashEmbedder(), top_k=10)
        assert f.name == "round_trip:test:hash:k=10"

    def test_protocol_compliance(self) -> None:
        """RoundTripFilter satisfies the QueryFilter protocol."""
        f = RoundTripFilter(SimpleChunker(), HashEmbedder())
        assert isinstance(f, QueryFilter)
