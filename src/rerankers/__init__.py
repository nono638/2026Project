"""Reranker implementations for RAGBench.

Cross-encoder rerankers re-score retrieved chunks using joint query-passage
encoding for more precise relevance ranking than embedding similarity alone.
"""

from __future__ import annotations

from src.rerankers.minilm import MiniLMReranker
from src.rerankers.bge import BGEReranker

__all__ = ["MiniLMReranker", "BGEReranker"]
