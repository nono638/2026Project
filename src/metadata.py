"""Pipeline metadata parsing for RAGBench CSVs.

Pure parsing functions that extract structured metadata from component .name
strings. No side effects, no imports from other project modules.

Every experimental axis gets its own column so analysis can slice by any
variable — chunker type, embedding model, retrieval mode, scorer, etc.
"""

from __future__ import annotations


def parse_chunker_name(name: str) -> dict:
    """Parse a chunker .name string into structured metadata.

    Known formats:
        "recursive:500:100"       → type=recursive, size=500, overlap=100
        "fixed:200:50"            → type=fixed, size=200, overlap=50
        "sentence:5"              → type=sentence, size=5, overlap=None
        "semantic:mxbai-embed-large" → type=semantic, size=None, overlap=None

    Args:
        name: Chunker .name property string.

    Returns:
        Dict with keys: chunk_type, chunk_size, chunk_overlap.
    """
    parts = name.split(":")
    chunk_type = parts[0]

    if chunk_type in ("recursive", "fixed") and len(parts) >= 3:
        return {
            "chunk_type": chunk_type,
            "chunk_size": int(parts[1]),
            "chunk_overlap": int(parts[2]),
        }
    elif chunk_type == "sentence" and len(parts) >= 2:
        return {
            "chunk_type": "sentence",
            "chunk_size": int(parts[1]),
            "chunk_overlap": None,
        }
    elif chunk_type == "semantic":
        return {
            "chunk_type": "semantic",
            "chunk_size": None,
            "chunk_overlap": None,
        }
    else:
        # Unknown format — use the full name as type
        return {
            "chunk_type": name,
            "chunk_size": None,
            "chunk_overlap": None,
        }


def parse_embedder_name(name: str, dimension: int | None = None) -> dict:
    """Parse an embedder .name string into structured metadata.

    Known formats:
        "ollama:mxbai-embed-large"    → provider=ollama, model=mxbai-embed-large
        "hf:all-MiniLM-L6-v2"        → provider=hf, model=all-MiniLM-L6-v2
        "google:text-embedding-005"   → provider=google, model=text-embedding-005

    Args:
        name: Embedder .name property string.
        dimension: Embedding vector dimension (from embedder.dimension).

    Returns:
        Dict with keys: embed_provider, embed_model, embed_dimension.
    """
    if ":" in name:
        provider, model = name.split(":", 1)
    else:
        provider, model = name, name

    return {
        "embed_provider": provider,
        "embed_model": model,
        "embed_dimension": dimension,
    }


def parse_scorer_name(name: str) -> dict:
    """Parse a scorer .name string into structured metadata.

    Known formats:
        "google:gemini-2.5-flash"             → provider=google, model=gemini-2.5-flash
        "anthropic:claude-haiku-4-5-20251001"  → provider=anthropic, model=...

    Args:
        name: Scorer .name property string.

    Returns:
        Dict with keys: scorer_provider, scorer_model.
    """
    if ":" in name:
        provider, model = name.split(":", 1)
    else:
        provider, model = name, name

    return {
        "scorer_provider": provider,
        "scorer_model": model,
    }


def parse_llm_name(name: str) -> dict:
    """Parse an LLM .name string into structured metadata.

    Known formats:
        "ollama"                              → provider=ollama
        "openai-compat:http://localhost:1234/v1" → provider=openai-compat

    Args:
        name: LLM .name property string.

    Returns:
        Dict with key: llm_provider.
    """
    # The provider is the first colon-separated segment
    provider = name.split(":")[0]
    return {"llm_provider": provider}


def build_retrieval_metadata(
    mode: str, top_k: int, num_retrieved: int
) -> dict:
    """Build retrieval metadata dict.

    Args:
        mode: Retrieval mode (hybrid, dense, sparse).
        top_k: Configured top_k value.
        num_retrieved: Actual number of chunks retrieved.

    Returns:
        Dict with keys: retrieval_mode, retrieval_top_k, num_chunks_retrieved.
    """
    return {
        "retrieval_mode": mode,
        "retrieval_top_k": top_k,
        "num_chunks_retrieved": num_retrieved,
    }


def build_context_metadata(retrieved_chunks: list[dict]) -> dict:
    """Build context metadata from retrieved chunks.

    Args:
        retrieved_chunks: List of dicts with 'text' key from retriever.retrieve().

    Returns:
        Dict with key: context_char_length.
    """
    total = sum(len(chunk.get("text", "")) for chunk in retrieved_chunks)
    return {"context_char_length": total}


def build_reranker_placeholder() -> dict:
    """Return placeholder reranker metadata (not yet implemented).

    Returns:
        Dict with keys: reranker_model, reranker_top_k (both None).
    """
    return {"reranker_model": None, "reranker_top_k": None}


def build_dataset_metadata(
    name: str | None = None, seed: int | None = None
) -> dict:
    """Build dataset metadata dict.

    Args:
        name: Dataset name (e.g., "hotpotqa", "squad"), or None for CSV corpus.
        seed: Random seed used for sampling, or None.

    Returns:
        Dict with keys: dataset_name, dataset_sample_seed.
    """
    return {
        "dataset_name": name,
        "dataset_sample_seed": seed,
    }
