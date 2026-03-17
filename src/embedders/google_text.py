"""Embedding via Google's text-embedding-005 model.

Uses the google-generativeai SDK (free tier available via Google AI Studio).
768 dimensions, up to 2048 tokens context.

Docs: https://ai.google.dev/gemini-api/docs/embeddings
"""

from __future__ import annotations

import os
import time

import numpy as np

# google-generativeai is deprecated in favor of google.genai, but the spec
# explicitly requires this SDK. Flag for morning review if migration is needed.
import google.generativeai as genai


class GoogleTextEmbedder:
    """Embedding via Google's text-embedding-005 (cloud-hosted, free tier).

    Implements the Embedder protocol from src.protocols. Provides a cloud-hosted
    comparison point against local models (Ollama, HuggingFace) in the experiment
    framework's embedder axis.
    """

    def __init__(
        self,
        model: str = "models/text-embedding-005",
        task_type: str = "retrieval_document",
        api_key: str | None = None,
    ) -> None:
        """Initialize with Google AI Studio API key and model config.

        Args:
            model: The Google embedding model ID.
            task_type: Embedding task type — "retrieval_document" for indexing,
                "retrieval_query" for queries. Affects embedding quality.
            api_key: Google AI Studio API key. Falls back to GOOGLE_API_KEY
                environment variable if not provided.

        Raises:
            ValueError: If no API key is found via parameter or environment.
        """
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Set GOOGLE_API_KEY environment variable or pass api_key "
                "to GoogleTextEmbedder"
            )
        self._model = model
        self._task_type = task_type
        # Configure the SDK with the API key once at init time
        genai.configure(api_key=resolved_key)

    @property
    def name(self) -> str:
        """Return unique identifier in format 'google:<model_name>'.

        Strips the 'models/' prefix to match the naming pattern of other
        embedders (e.g., 'ollama:mxbai-embed-large', 'hf:all-MiniLM-L6-v2').
        """
        # Extract model name after "models/" prefix
        model_name = self._model.removeprefix("models/")
        return f"google:{model_name}"

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (always 768 for text-embedding-005)."""
        return 768

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using Google's text-embedding-005.

        Calls genai.embed_content for each text individually because the SDK's
        batch behavior varies across versions. Single-text calls are reliable.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), 768) with dtype float32.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings = []
        for text in texts:
            embedding = self._embed_single(text, self._task_type)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Embed query texts with task_type="retrieval_query".

        Convenience method NOT part of the Embedder protocol. The Google API
        produces better retrieval results when document and query embeddings
        use different task types. The standard embed() uses "retrieval_document";
        this method uses "retrieval_query" for search-time embedding.

        Args:
            texts: List of query strings to embed.

        Returns:
            numpy array of shape (len(texts), 768) with dtype float32.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings = []
        for text in texts:
            embedding = self._embed_single(text, "retrieval_query")
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def _embed_single(self, text: str, task_type: str) -> list[float]:
        """Embed a single text with one retry on rate limiting.

        Args:
            text: The text to embed.
            task_type: The embedding task type for the API call.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            Exception: If the API call fails twice (after one retry).
        """
        try:
            result = genai.embed_content(
                model=self._model,
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            # Simple retry with 1-second sleep for rate limiting.
            # No exponential backoff per spec — single retry is sufficient
            # for the free tier's rate limits during experiment runs.
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(1)
                result = genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type=task_type,
                )
                return result["embedding"]
            raise
