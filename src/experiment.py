"""Experiment runner and result container for the RAG research tool.

The central orchestrator that replaces src/data/generate.py. Runs the full
cartesian product of (document x chunker x embedder x query x strategy x model),
caches FAISS indexes to avoid redundant embedding work, and collects scored
results into an ExperimentResult for analysis.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.protocols import Chunker, Embedder, Strategy, Scorer
from src.retriever import Retriever
from src.features import extract_features


class Experiment:
    """Runs a cartesian-product experiment across all component combinations.

    Validates that all components implement the correct Protocol at init time
    for clear error messages, rather than failing deep inside the run loop.
    """

    def __init__(
        self,
        chunkers: list[Chunker],
        embedders: list[Embedder],
        models: list[str],
        strategies: list[Strategy],
        scorer: Scorer,
        top_k: int = 5,
    ) -> None:
        """Initialize the experiment with component lists.

        Args:
            chunkers: List of Chunker implementations to test.
            embedders: List of Embedder implementations to test.
            models: List of Ollama model names (e.g., ['qwen3:0.6b', 'qwen3:4b']).
            strategies: List of Strategy implementations to test.
            scorer: A single Scorer implementation for evaluating answers.
            top_k: Number of chunks to retrieve per query.

        Raises:
            TypeError: If any component doesn't implement its Protocol.
        """
        # Validate protocols at init time for clear error messages
        for c in chunkers:
            if not isinstance(c, Chunker):
                raise TypeError(f"{c} does not implement the Chunker protocol")
        for e in embedders:
            if not isinstance(e, Embedder):
                raise TypeError(f"{e} does not implement the Embedder protocol")
        for s in strategies:
            if not isinstance(s, Strategy):
                raise TypeError(f"{s} does not implement the Strategy protocol")
        if not isinstance(scorer, Scorer):
            raise TypeError(f"{scorer} does not implement the Scorer protocol")

        self._chunkers = chunkers
        self._embedders = embedders
        self._models = models
        self._strategies = strategies
        self._scorer = scorer
        self._top_k = top_k
        self._documents: list[dict] = []
        self._queries: list[dict] = []

    def load_corpus(self, documents: list[dict], queries: list[dict]) -> None:
        """Load documents and queries for the experiment.

        Args:
            documents: List of dicts with 'title' and 'text' keys.
            queries: List of dicts with 'text' and 'type' keys.
                     Type is one of: 'lookup', 'synthesis', 'multi_hop'.
        """
        self._documents = documents
        self._queries = queries

    def run(self, progress: bool = True) -> ExperimentResult:
        """Run the full experiment matrix and return results.

        Iterates: doc x chunker x embedder x query x strategy x model.
        Caches FAISS indexes per (doc_hash, chunker.name, embedder.name)
        to avoid redundant embedding work.

        Args:
            progress: Whether to print progress updates.

        Returns:
            ExperimentResult wrapping a DataFrame of all scored runs.
        """
        rows: list[dict] = []
        index_cache: dict[tuple, Retriever] = {}

        # Handle empty corpus gracefully
        if not self._documents or not self._queries:
            return ExperimentResult(pd.DataFrame(rows))

        total = (len(self._documents) * len(self._chunkers) *
                 len(self._embedders) * len(self._queries) *
                 len(self._strategies) * len(self._models))
        count = 0

        for doc in self._documents:
            doc_hash = hashlib.md5(doc["text"].encode()).hexdigest()[:8]

            for chunker in self._chunkers:
                # Chunk once per (doc, chunker)
                chunks = chunker.chunk(doc["text"])

                for embedder in self._embedders:
                    # Build/cache retriever per (doc, chunker, embedder)
                    cache_key = (doc_hash, chunker.name, embedder.name)
                    if cache_key not in index_cache:
                        index_cache[cache_key] = Retriever(
                            chunks, embedder, self._top_k
                        )
                    retriever = index_cache[cache_key]

                    for query in self._queries:
                        # Extract features once per (query, doc, retriever)
                        features = extract_features(
                            query["text"], doc["text"], retriever
                        )

                        for strategy in self._strategies:
                            for model in self._models:
                                count += 1
                                if progress:
                                    print(f"[{count}/{total}] {strategy.name} / "
                                          f"{model} / {query['text'][:50]}...")

                                answer = strategy.run(
                                    query["text"], retriever, model
                                )
                                scores = self._scorer.score(
                                    query["text"], doc["text"], answer
                                )

                                row = {
                                    "doc_title": doc["title"],
                                    "query_text": query["text"],
                                    "query_type": query["type"],
                                    "chunker": chunker.name,
                                    "embedder": embedder.name,
                                    "model": model,
                                    "strategy": strategy.name,
                                    "answer": answer,
                                    **scores,
                                    "quality": sum(scores.values()) / len(scores) if scores else 0,
                                    **features,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                rows.append(row)

        return ExperimentResult(pd.DataFrame(rows))


class ExperimentResult:
    """Wraps a DataFrame of experiment results with analysis methods."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with a results DataFrame.

        Args:
            df: DataFrame containing experiment results.
        """
        self.df = df

    def compare(self) -> pd.DataFrame:
        """Print and return summary comparison grouped by strategy x model.

        Returns:
            Summary DataFrame with mean, std, count of quality scores.
        """
        if self.df.empty or "quality" not in self.df.columns:
            print("No quality scores found.")
            return self.df
        summary = self.df.groupby(["strategy", "model"])["quality"].agg(
            ["mean", "std", "count"]
        ).round(3)
        print(summary.to_string())
        return summary

    def pivot(self, rows: str, cols: str, values: str = "quality") -> pd.DataFrame:
        """Create a pivot table for analysis.

        Args:
            rows: Column name to use as pivot table rows.
            cols: Column name to use as pivot table columns.
            values: Column name for the values (default: 'quality').

        Returns:
            Pivot table DataFrame.
        """
        return self.df.pivot_table(index=rows, columns=cols, values=values, aggfunc="mean")

    def to_parquet(self, path: Path) -> None:
        """Save results to Parquet.

        Args:
            path: File path for the Parquet output.
        """
        self.df.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: Path) -> ExperimentResult:
        """Load results from Parquet.

        Args:
            path: File path of the Parquet file to load.

        Returns:
            ExperimentResult with loaded data.
        """
        return cls(pd.read_parquet(path))

    def best_config(self, metric: str = "quality") -> tuple:
        """Return the config with the highest mean score for the given metric.

        Args:
            metric: The metric column to optimize (default: 'quality').

        Returns:
            Tuple of (chunker, embedder, strategy, model) for the best config.
        """
        return self.df.groupby(
            ["chunker", "embedder", "strategy", "model"]
        )[metric].mean().idxmax()
