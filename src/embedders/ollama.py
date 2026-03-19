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

    def __init__(self, model: str = "mxbai-embed-large", host: str | None = None) -> None:
        """Initialize with the Ollama model name and optional remote host.

        Args:
            model: Name of the Ollama embedding model to use.
            host: Ollama server URL. None uses the default localhost:11434.
                  Pass a RunPod proxy URL for remote GPU embeddings.
        """
        self._model = model
        # Match the pattern used by OllamaLLM — pass host only when specified
        self._client = Client(host=host) if host else Client()
        self._dimension: int | None = None

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

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        response = self._client.embed(model=self._model, input=texts)
        result = np.array(response.embeddings, dtype=np.float32)
        if self._dimension is None:
            self._dimension = result.shape[1]
        return result
