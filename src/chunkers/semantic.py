"""Semantic chunking using LangChain's SemanticChunker.

Wraps LangChain's SemanticChunker to implement the Chunker protocol.

The embedding model used for chunking is independent of the embedding model
used for retrieval. They serve different purposes — chunking embeddings find
meaning breakpoints in text, retrieval embeddings find query-to-chunk similarity.
"""

from __future__ import annotations

from typing import Any

from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_community.embeddings import OllamaEmbeddings


class _TruncatingOllamaEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings that clips each input to ``max_chars`` before sending.

    LangChain's SemanticChunker splits documents on punctuation and embeds
    each "sentence." On small-context embedders (e.g., all-minilm:22m at 512
    tokens) a single ill-punctuated sentence can overflow. Without this
    wrapper LangChain's embedder raises HTTP 500, retries 3x, and the
    chunker fails for the whole document. Truncating to a safe char cap
    keeps the boundary-detection pass robust across embedders of any size.
    """

    max_chars: int = 1500

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([t[: self.max_chars] for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(text[: self.max_chars])


class SemanticChunker:
    """Semantic chunking using LangChain's SemanticChunker.

    Uses an embedding model to find meaning boundaries in text.
    """

    def __init__(
        self,
        embedding_model: str = "embeddinggemma:300m",
        embedder_max_chars: int = 1500,
    ) -> None:
        """Initialize with the embedding model to use for chunk boundary detection.

        Args:
            embedding_model: Ollama model name for computing embeddings.
            embedder_max_chars: Per-sentence truncation cap. 1500 chars
                (~375 tokens) fits even all-minilm:22m's 512-token context;
                most natural sentences are well under this so the cap rarely
                bites for cleanly-punctuated text. Lower for tighter
                embedders if needed.
        """
        self._model = embedding_model
        self._embedder_max_chars = embedder_max_chars

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
        embeddings = _TruncatingOllamaEmbeddings(model=self._model)
        embeddings.max_chars = self._embedder_max_chars
        chunker = LCSemanticChunker(embeddings)
        docs = chunker.create_documents([text])
        return [doc.page_content for doc in docs]
