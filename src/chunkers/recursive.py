"""Recursive character text splitting via LangChain.

Splits on a hierarchy of separators (paragraphs -> sentences -> words)
to keep chunks at a target size while respecting text structure. A good
middle ground between fixed-size and semantic chunking.
"""

from __future__ import annotations


class RecursiveChunker:
    """Recursive character text splitting via LangChain.

    Splits on a hierarchy of separators (paragraphs -> sentences -> words)
    to keep chunks at a target size while respecting text structure.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize with chunk size and overlap parameters.

        Args:
            chunk_size: Target character count per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    @property
    def name(self) -> str:
        """Return unique identifier for this chunker config."""
        return f"recursive:{self._chunk_size}:{self._overlap}"

    def chunk(self, text: str) -> list[str]:
        """Split text using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: The full document text to chunk.

        Returns:
            List of chunk strings.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._overlap,
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]
