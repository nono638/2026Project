"""Built-in dataset loaders for RAGBench.

Provides gold-standard datasets so users can calibrate their pipeline
before running on their own data. Each loader converts external datasets
into RAGBench's Document + Query format.
"""

from __future__ import annotations

from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
from src.datasets.squad import load_squad, sample_squad

__all__ = ["load_hotpotqa", "sample_hotpotqa", "load_squad", "sample_squad"]
