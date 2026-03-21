"""Tests for src/metadata.py — pure parsing functions for pipeline metadata.

Each parse function is tested with every known format plus edge cases
(unknown format, missing colons, empty strings).
"""

from __future__ import annotations

import pytest

from src.metadata import (
    parse_chunker_name,
    parse_embedder_name,
    parse_model_name,
    parse_scorer_name,
    parse_llm_name,
    build_retrieval_metadata,
    build_context_metadata,
    build_reranker_metadata,
    build_dataset_metadata,
)


# ---------------------------------------------------------------------------
# parse_chunker_name
# ---------------------------------------------------------------------------

class TestParseChunkerName:
    """Test chunker name parsing for all known formats."""

    def test_recursive(self):
        result = parse_chunker_name("recursive:500:100")
        assert result == {"chunk_type": "recursive", "chunk_size": 500, "chunk_overlap": 100}

    def test_fixed(self):
        result = parse_chunker_name("fixed:200:50")
        assert result == {"chunk_type": "fixed", "chunk_size": 200, "chunk_overlap": 50}

    def test_sentence(self):
        result = parse_chunker_name("sentence:5")
        assert result == {"chunk_type": "sentence", "chunk_size": 5, "chunk_overlap": None}

    def test_semantic(self):
        result = parse_chunker_name("semantic:mxbai-embed-large")
        assert result == {"chunk_type": "semantic", "chunk_size": None, "chunk_overlap": None}

    def test_unknown_format_no_colon(self):
        result = parse_chunker_name("custom_chunker")
        assert result == {"chunk_type": "custom_chunker", "chunk_size": None, "chunk_overlap": None}

    def test_unknown_format_with_colon(self):
        """Unknown type with colon but not enough parts falls through to default."""
        result = parse_chunker_name("new_type:abc")
        assert result["chunk_type"] == "new_type:abc"
        assert result["chunk_size"] is None

    def test_empty_string(self):
        result = parse_chunker_name("")
        assert result["chunk_type"] == ""
        assert result["chunk_size"] is None


# ---------------------------------------------------------------------------
# parse_embedder_name
# ---------------------------------------------------------------------------

class TestParseEmbedderName:
    """Test embedder name parsing."""

    def test_ollama(self):
        result = parse_embedder_name("ollama:mxbai-embed-large", 1024)
        assert result == {
            "embed_provider": "ollama",
            "embed_model": "mxbai-embed-large",
            "embed_dimension": 1024,
        }

    def test_huggingface(self):
        result = parse_embedder_name("hf:all-MiniLM-L6-v2", 384)
        assert result == {
            "embed_provider": "hf",
            "embed_model": "all-MiniLM-L6-v2",
            "embed_dimension": 384,
        }

    def test_google(self):
        result = parse_embedder_name("google:text-embedding-005", 768)
        assert result == {
            "embed_provider": "google",
            "embed_model": "text-embedding-005",
            "embed_dimension": 768,
        }

    def test_no_dimension(self):
        result = parse_embedder_name("ollama:nomic-embed-text")
        assert result["embed_dimension"] is None

    def test_no_colon(self):
        result = parse_embedder_name("custom", 256)
        assert result == {"embed_provider": "custom", "embed_model": "custom", "embed_dimension": 256}


# ---------------------------------------------------------------------------
# parse_scorer_name
# ---------------------------------------------------------------------------

class TestParseScorerName:
    """Test scorer name parsing."""

    def test_google(self):
        result = parse_scorer_name("google:gemini-2.5-flash")
        assert result == {"scorer_provider": "google", "scorer_model": "gemini-2.5-flash"}

    def test_anthropic(self):
        result = parse_scorer_name("anthropic:claude-haiku-4-5-20251001")
        assert result == {"scorer_provider": "anthropic", "scorer_model": "claude-haiku-4-5-20251001"}

    def test_no_colon(self):
        result = parse_scorer_name("custom_scorer")
        assert result == {"scorer_provider": "custom_scorer", "scorer_model": "custom_scorer"}


# ---------------------------------------------------------------------------
# parse_llm_name
# ---------------------------------------------------------------------------

class TestParseLlmName:
    """Test LLM name parsing."""

    def test_ollama(self):
        result = parse_llm_name("ollama")
        assert result == {"llm_provider": "ollama"}

    def test_openai_compat(self):
        result = parse_llm_name("openai-compat:http://localhost:1234/v1")
        assert result == {"llm_provider": "openai-compat"}


# ---------------------------------------------------------------------------
# parse_model_name
# ---------------------------------------------------------------------------

class TestParseModelName:
    """Test model name parsing for Ollama-style names."""

    def test_qwen3_small(self):
        result = parse_model_name("qwen3:0.6b")
        assert result == {"model_family": "qwen3", "model_param_billions": 0.6}

    def test_qwen3_large(self):
        result = parse_model_name("qwen3:8b")
        assert result == {"model_family": "qwen3", "model_param_billions": 8.0}

    def test_gemma3(self):
        result = parse_model_name("gemma3:4b")
        assert result == {"model_family": "gemma3", "model_param_billions": 4.0}

    def test_uppercase_b(self):
        result = parse_model_name("qwen3:4B")
        assert result == {"model_family": "qwen3", "model_param_billions": 4.0}

    def test_no_colon(self):
        result = parse_model_name("custom-model")
        assert result == {"model_family": "custom-model", "model_param_billions": None}

    def test_non_numeric_size(self):
        result = parse_model_name("llama3:latest")
        assert result == {"model_family": "llama3", "model_param_billions": None}

    def test_empty_string(self):
        result = parse_model_name("")
        assert result == {"model_family": "", "model_param_billions": None}


# ---------------------------------------------------------------------------
# build_* functions
# ---------------------------------------------------------------------------

class TestBuildRetrievalMetadata:
    """Test retrieval metadata builder."""

    def test_basic(self):
        result = build_retrieval_metadata("hybrid", 5, 5)
        assert result == {
            "retrieval_mode": "hybrid",
            "retrieval_top_k": 5,
            "num_chunks_retrieved": 5,
        }

    def test_fewer_retrieved_than_top_k(self):
        result = build_retrieval_metadata("dense", 10, 3)
        assert result["num_chunks_retrieved"] == 3


class TestBuildContextMetadata:
    """Test context metadata builder."""

    def test_basic(self):
        chunks = [{"text": "hello", "score": 0.9}, {"text": "world!", "score": 0.8}]
        result = build_context_metadata(chunks)
        assert result == {"context_char_length": 11}

    def test_empty_chunks(self):
        result = build_context_metadata([])
        assert result == {"context_char_length": 0}

    def test_missing_text_key(self):
        """Chunks without 'text' key should contribute 0 chars."""
        result = build_context_metadata([{"score": 0.5}])
        assert result == {"context_char_length": 0}


class TestBuildRerankerMetadata:
    """Test reranker metadata builder."""

    def test_with_values(self):
        """When reranker is used, should return name and top_k."""
        result = build_reranker_metadata("minilm:ms-marco-MiniLM-L-6-v2", 3)
        assert result == {"reranker_model": "minilm:ms-marco-MiniLM-L-6-v2", "reranker_top_k": 3}

    def test_no_reranker(self):
        """When no reranker, both fields should be None."""
        result = build_reranker_metadata()
        assert result == {"reranker_model": None, "reranker_top_k": None}


class TestBuildDatasetMetadata:
    """Test dataset metadata builder."""

    def test_with_values(self):
        result = build_dataset_metadata("hotpotqa", 42)
        assert result == {"dataset_name": "hotpotqa", "dataset_sample_seed": 42}

    def test_defaults(self):
        result = build_dataset_metadata()
        assert result == {"dataset_name": None, "dataset_sample_seed": None}
