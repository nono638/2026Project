"""Embedding via Ollama-hosted embedding models.

Wraps the Ollama client to implement the Embedder protocol.
Logic migrated from src/pipeline/retrieval.py embed_texts().

Dimension is detected lazily on the first embed() call by inspecting the
shape of the returned array. This avoids requiring the user to know the
dimension in advance.
"""

from __future__ import annotations

import numpy as np
from ollama import Client


class OllamaEmbedder:
    """Embedding via any Ollama-hosted embedding model.

    Implements the Embedder protocol from src.protocols.
    """

    # 7000 chars (~1750 tokens at 4 chars/token avg) sits below embeddinggemma's
    # 8K-token context, so truncation is a defensive guard rather than the hot
    # path. Embedder history: mxbai-embed-large (512-token cap, pre-2026-05-09)
    # → qwen3-embedding:4b (40K context, fixed Exp 2 chunker overflow but was
    # disproportionately large) → embeddinggemma:300m (2026-05-13, 8K context,
    # ~13x smaller). See docs/methodology.html "Embedder" section for rationale.
    DEFAULT_MAX_CHARS = 7000

    def __init__(
        self,
        model: str = "embeddinggemma:300m",
        host: str | None = None,
        max_chars: int | None = None,
    ) -> None:
        """Initialize with the Ollama model name and optional remote host.

        Args:
            model: Name of the Ollama embedding model to use.
            host: Ollama server URL. None uses the default localhost:11434.
                  Pass a RunPod proxy URL for remote GPU embeddings.
            max_chars: Per-input character cap; inputs longer than this are
                truncated before embedding. None uses DEFAULT_MAX_CHARS.
        """
        self._model = model
        # Match the pattern used by OllamaLLM — pass host only when specified
        self._client = Client(host=host) if host else Client()
        self._dimension: int | None = None
        self._max_chars = max_chars if max_chars is not None else self.DEFAULT_MAX_CHARS

    @property
    def name(self) -> str:
        """Return unique identifier for this embedder."""
        return f"ollama:{self._model}"

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension, detecting lazily if needed.

        Returns:
            Integer dimension of the embedding vectors.
        """
        if self._dimension is None:
            # Detect dimension by embedding a probe string
            probe = self.embed(["hello"])
            self._dimension = probe.shape[1]
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the Ollama model.

        Each input is truncated to ``self._max_chars`` before embedding so a
        single oversized chunk in a batch doesn't fail the entire request.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        clipped = [t[: self._max_chars] for t in texts]
        response = self._client.embed(model=self._model, input=clipped)
        result = np.array(response.embeddings, dtype=np.float32)
        if self._dimension is None:
            self._dimension = result.shape[1]
        return result
