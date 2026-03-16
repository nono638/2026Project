"""Embedding via sentence-transformers models (runs locally).

Useful for users who want to compare Ollama embeddings against
HuggingFace models, or who don't have Ollama installed. Runs entirely
on the local machine using the sentence-transformers library.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    """Embedding via sentence-transformers models (runs locally).

    Implements the Embedder protocol from src.protocols.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with the HuggingFace model name.

        Args:
            model: Name of the sentence-transformers model to use.
        """
        self._model_name = model
        self._model = SentenceTransformer(model)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        """Return unique identifier for this embedder."""
        return f"hf:{self._model_name}"

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the sentence-transformers model.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
