"""Query set analysis utilities.

Tools for analyzing the quality and distribution of generated query sets
at the set level (not per-query filtering).
"""

from src.query_analysis.distribution import DistributionAnalyzer

__all__ = ["DistributionAnalyzer"]
