"""Pipeline metadata parsing for RAGBench CSVs.

Pure parsing functions that extract structured metadata from component .name
strings. No side effects, no imports from other project modules.

Every experimental axis gets its own column so analysis can slice by any
variable — chunker type, embedding model, retrieval mode, scorer, etc.
"""

from __future__ import annotations

# Ollama client import — used for context window queries.
# Imported at module level so tests can mock 'src.metadata.Client'.
try:
    from ollama import Client
except ImportError:
    Client = None  # type: ignore[misc,assignment]


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


def build_reranker_metadata(
    name: str | None = None, top_k: int | None = None
) -> dict:
    """Build reranker metadata dict.

    Args:
        name: Reranker .name property string, or None if no reranker used.
        top_k: Reranker output top_k, or None.

    Returns:
        Dict with keys: reranker_model, reranker_top_k.
    """
    return {"reranker_model": name, "reranker_top_k": top_k}


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


# Module-level cache for context window lookups — avoids repeated API calls
# for the same model across different rows in the experiment loop.
_context_window_cache: dict[str, int | None] = {}


def get_llm_context_window(
    model: str, provider: str | None = None, host: str | None = None,
) -> int | None:
    """Query the LLM's context window size in tokens.

    For Ollama models, queries the Ollama API via client.show(). For other
    providers, returns None (no standard API for this).

    Results are cached per model name to avoid repeated API calls within
    a single experiment run.

    Args:
        model: Model name (e.g., "qwen3:4b").
        provider: LLM provider (e.g., "ollama"). If not "ollama", returns None.
        host: Ollama host URL, or None for localhost default.

    Returns:
        Context window size in tokens, or None if unknown.
    """
    if model in _context_window_cache:
        return _context_window_cache[model]

    ctx = None
    if provider == "ollama":
        ctx = _query_ollama_context_window(model, host)

    _context_window_cache[model] = ctx
    return ctx


def _query_ollama_context_window(model: str, host: str | None) -> int | None:
    """Query Ollama API for a model's context window size.

    Uses ollama.Client().show() which returns model metadata including
    parameters. The context window is in the 'num_ctx' model parameter.

    Args:
        model: Ollama model name.
        host: Ollama server URL, or None for default.

    Returns:
        Context window size in tokens, or None if unavailable.
    """
    try:
        client = Client(host=host) if host else Client()
        info = client.show(model)
        # Ollama returns model info with parameter details.
        # The context window is typically in model_info or parameters.
        # Try multiple locations since Ollama API structure varies by version.
        if hasattr(info, 'model_info') and info.model_info:
            # Look for context_length or num_ctx in model_info dict
            for key in info.model_info:
                if 'context_length' in key.lower():
                    return int(info.model_info[key])
        # Fallback: check modelfile parameters
        if hasattr(info, 'parameters') and info.parameters:
            for line in info.parameters.split('\n'):
                if 'num_ctx' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return int(parts[-1])
        return None
    except Exception:
        # Ollama not running, model not found, network error, etc.
        return None


def build_llm_context_metadata(
    model: str,
    provider: str | None = None,
    host: str | None = None,
    context_char_length: int = 0,
) -> dict:
    """Build LLM context window metadata.

    Queries the model's context window size and computes a utilization ratio
    to detect "lost in the middle" effects — when too much context is stuffed
    into the window, answer quality degrades.

    Args:
        model: Model name.
        provider: LLM provider string.
        host: LLM host URL.
        context_char_length: Total character length of retrieved context.

    Returns:
        Dict with keys: llm_context_window, context_utilization_ratio.
    """
    ctx_window = get_llm_context_window(model, provider, host)

    if ctx_window is not None and ctx_window > 0:
        # Rough char-to-token conversion: ~4 chars per token for English.
        # Not perfect for all tokenizers, but close enough for a relative
        # signal used by the meta-learner.
        approx_tokens = context_char_length / 4
        ratio = approx_tokens / ctx_window
    else:
        ratio = None

    return {
        "llm_context_window": ctx_window,
        "context_utilization_ratio": round(ratio, 4) if ratio is not None else None,
    }
