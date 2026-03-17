"""Built-in dataset loaders for RAGBench.

Provides gold-standard datasets so users can calibrate their pipeline
before running on their own data. Each loader converts external datasets
into RAGBench's Document + Query format.
"""

from __future__ import annotations

from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa

__all__ = ["load_hotpotqa", "sample_hotpotqa"]
