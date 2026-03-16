"""Sentence-level chunking.

Splits text into individual sentences, then groups them into chunks
of N sentences each. Simple and interpretable — good for corpora
where sentences are self-contained units of meaning.
"""

from __future__ import annotations

import re


class SentenceChunker:
    """Sentence-level chunking.

    Splits text into individual sentences, then groups them into chunks
    of N sentences each. Simple and interpretable.
    """

    def __init__(self, sentences_per_chunk: int = 5) -> None:
        """Initialize with the number of sentences per chunk.

        Args:
            sentences_per_chunk: How many sentences to group into each chunk.
        """
        self._n = sentences_per_chunk

    @property
    def name(self) -> str:
        """Return unique identifier for this chunker config."""
        return f"sentence:{self._n}"

    def chunk(self, text: str) -> list[str]:
        """Split text into sentence-grouped chunks.

        Args:
            text: The full document text to chunk.

        Returns:
            List of chunk strings, each containing N sentences.
        """
        # Split on sentence boundaries (period, exclamation, question mark followed by space)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self._n):
            chunks.append(" ".join(sentences[i:i + self._n]))
        return chunks
