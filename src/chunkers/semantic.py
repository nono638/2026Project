"""Semantic chunking using LangChain's SemanticChunker.

Wraps LangChain's SemanticChunker to implement the Chunker protocol.
Logic migrated from src/pipeline/chunking.py.

The embedding model used for chunking is independent of the embedding model
used for retrieval. They serve different purposes — chunking embeddings find
meaning breakpoints in text, retrieval embeddings find query-to-chunk similarity.
"""

from __future__ import annotations

from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_community.embeddings import OllamaEmbeddings


class SemanticChunker:
    """Semantic chunking using LangChain's SemanticChunker.

    Uses an embedding model to find meaning boundaries in text.
    """

    def __init__(self, embedding_model: str = "mxbai-embed-large") -> None:
        """Initialize with the embedding model to use for chunk boundary detection.

        Args:
            embedding_model: Ollama model name for computing embeddings.
        """
        self._model = embedding_model

    @property
    def name(self) -> str:
        """Return unique identifier for this chunker config."""
        return f"semantic:{self._model}"

    def chunk(self, text: str) -> list[str]:
        """Split document text into semantic chunks.

        Args:
            text: The full document text to chunk.

        Returns:
            List of chunk strings.
        """
        embeddings = OllamaEmbeddings(model=self._model)
        chunker = LCSemanticChunker(embeddings)
        docs = chunker.create_documents([text])
        return [doc.page_content for doc in docs]
