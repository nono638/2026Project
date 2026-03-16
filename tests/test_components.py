"""Tests for the remaining pluggable components.

Tests chunkers and embedders that don't require external services (Ollama).
Strategy tests that require Ollama are marked @pytest.mark.slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.protocols import Chunker, Embedder, Strategy
from src.chunkers.fixed import FixedSizeChunker
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.sentence import SentenceChunker
from src.chunkers.semantic import SemanticChunker
from src.embedders.huggingface import HuggingFaceEmbedder
from src.strategies.multi_query import MultiQueryRAG
from src.strategies.corrective import CorrectiveRAG
from src.strategies.adaptive import AdaptiveRAG


# ---------------------------------------------------------------------------
# Sample text for chunker tests
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Python is a popular programming language. "
    "FAISS is a library for efficient similarity search. "
    "Machine learning models can be large or small. "
    "Retrieval-augmented generation combines search with language models. "
    "Semantic chunking uses embeddings to find meaning boundaries. "
    "Fixed-size chunking splits text by word count. "
    "Sentence chunking groups sentences together. "
    "The recursive approach uses separator hierarchies. "
    "Each approach has trade-offs in terms of quality and speed."
)


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    """Test FixedSizeChunker behavior."""

    def test_basic_chunking(self) -> None:
        """Verify chunks are produced with expected word counts."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk(SAMPLE_TEXT)

        assert len(chunks) > 1
        # First chunk should have exactly 10 words
        assert len(chunks[0].split()) == 10

    def test_overlap(self) -> None:
        """Verify overlap: last words of chunk N appear at start of chunk N+1."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk(SAMPLE_TEXT)

        if len(chunks) >= 2:
            first_words = chunks[0].split()
            second_words = chunks[1].split()
            # Last 3 words of chunk 0 should be first 3 words of chunk 1
            assert first_words[-3:] == second_words[:3]

    def test_name(self) -> None:
        """Verify name format includes parameters."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        assert chunker.name == "fixed:200:50"

    def test_empty_text(self) -> None:
        """Empty text should return empty list."""
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []


class TestSentenceChunker:
    """Test SentenceChunker behavior."""

    def test_sentence_grouping(self) -> None:
        """Verify sentences are grouped correctly."""
        chunker = SentenceChunker(sentences_per_chunk=2)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        assert "First sentence." in chunks[0]
        assert "Second sentence." in chunks[0]
        assert "Third sentence." in chunks[1]
        assert "Fourth sentence." in chunks[1]

    def test_name(self) -> None:
        """Verify name format."""
        chunker = SentenceChunker(sentences_per_chunk=5)
        assert chunker.name == "sentence:5"


class TestRecursiveChunker:
    """Test RecursiveChunker behavior."""

    def test_chunks_under_target_size(self) -> None:
        """Verify chunks are at or under the target character size."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        # Use a longer text to ensure multiple chunks
        long_text = SAMPLE_TEXT * 5
        chunks = chunker.chunk(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance since LangChain may slightly exceed the target
            assert len(chunk) <= 200, f"Chunk too large: {len(chunk)} chars"

    def test_name(self) -> None:
        """Verify name format includes parameters."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.name == "recursive:500:100"


class TestProtocolCompliance:
    """Verify all new components pass isinstance checks against Protocols."""

    def test_all_chunkers_protocol(self) -> None:
        """All chunkers should satisfy the Chunker protocol."""
        assert isinstance(FixedSizeChunker(), Chunker)
        assert isinstance(RecursiveChunker(), Chunker)
        assert isinstance(SentenceChunker(), Chunker)
        assert isinstance(SemanticChunker(), Chunker)

    def test_all_strategies_protocol(self) -> None:
        """All strategies should satisfy the Strategy protocol."""
        assert isinstance(MultiQueryRAG(), Strategy)
        assert isinstance(CorrectiveRAG(), Strategy)
        assert isinstance(AdaptiveRAG(), Strategy)

    def test_huggingface_embedder_protocol(self) -> None:
        """HuggingFaceEmbedder should satisfy the Embedder protocol."""
        # Use a small model for fast test execution
        embedder = HuggingFaceEmbedder(model="all-MiniLM-L6-v2")
        assert isinstance(embedder, Embedder)


class TestHuggingFaceEmbedder:
    """Test HuggingFaceEmbedder functionality."""

    def test_embed_shape(self) -> None:
        """Verify embed returns correct shape."""
        embedder = HuggingFaceEmbedder(model="all-MiniLM-L6-v2")
        texts = ["hello world", "test sentence"]
        result = embedder.embed(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
        assert result.shape[1] == embedder.dimension
        assert result.dtype == np.float32

    def test_dimension(self) -> None:
        """Verify dimension property returns a positive integer."""
        embedder = HuggingFaceEmbedder(model="all-MiniLM-L6-v2")
        assert embedder.dimension > 0

    def test_name(self) -> None:
        """Verify name format."""
        embedder = HuggingFaceEmbedder(model="all-MiniLM-L6-v2")
        assert embedder.name == "hf:all-MiniLM-L6-v2"
