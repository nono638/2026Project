"""Fixed-size chunking by token count (approximated by words).

The simplest chunking strategy — splits text into chunks of approximately
equal size with optional overlap. No semantic awareness. Useful as a
baseline and for users who want predictable chunk sizes.
"""

from __future__ import annotations


class FixedSizeChunker:
    """Fixed-size chunking by token count (approximated by words).

    The simplest chunking strategy — splits text into chunks of approximately
    equal size with optional overlap. No semantic awareness.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50) -> None:
        """Initialize with chunk size and overlap parameters.

        Args:
            chunk_size: Number of words per chunk.
            overlap: Number of overlapping words between consecutive chunks.

        Raises:
            ValueError: If chunk_size <= 0 or overlap is not in [0, chunk_size).
                Reject misconfiguration up-front; otherwise the step
                ``chunk_size - overlap`` would be <= 0 and chunk() would
                loop forever.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap must be in [0, chunk_size); got overlap={overlap}, "
                f"chunk_size={chunk_size}"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def name(self) -> str:
        """Return unique identifier for this chunker config."""
        return f"fixed:{self._chunk_size}:{self._overlap}"

    def chunk(self, text: str) -> list[str]:
        """Split text into fixed-size word chunks with overlap.

        Args:
            text: The full document text to chunk.

        Returns:
            List of chunk strings.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._overlap
        return chunks
