"""Experiment runner and result container for the RAG research tool.

The central orchestrator that replaces src/data/generate.py. Runs the full
cartesian product of (document x chunker x embedder x query x strategy x model),
caches FAISS indexes to avoid redundant embedding work, and collects scored
results into an ExperimentResult for analysis.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.protocols import Chunker, Embedder, Strategy, Scorer, Reranker
from src.retriever import Retriever
from src.features import extract_features
from src.metadata import (
    parse_chunker_name,
    parse_embedder_name,
    parse_model_name,
    parse_scorer_name,
    parse_llm_name,
    build_retrieval_metadata,
    build_context_metadata,
    build_reranker_metadata,
    build_dataset_metadata,
    build_llm_context_metadata,
)


# Constraint parsing — same syntax as src/model/train.py, copied here to avoid
# coupling the analysis layer to the training module. Both modules share the
# syntax ">3.5", "<=1000", "==qwen3:4b" so users learn it once.
_CONSTRAINT_RE = re.compile(r"^(>=|<=|!=|==|>|<)(.+)$")

# The 4 axes that define a configuration in the experiment matrix.
_CONFIG_AXES = ["chunker", "embedder", "strategy", "model"]


def _apply_constraints(df: pd.DataFrame, constraints: dict[str, str]) -> pd.DataFrame:
    """Filter a DataFrame by constraint expressions.

    Each key is a column name, each value is an operator + value string
    (e.g. ">3.0", "<=5000", "==qwen3:4b"). String values without an
    operator prefix do exact match.

    Args:
        df: Input DataFrame to filter.
        constraints: Mapping of column name to constraint expression.

    Returns:
        Filtered DataFrame (may be empty if no rows match).

    Raises:
        KeyError: If a constraint references a column not in the DataFrame.
        ValueError: If a constraint string cannot be parsed.
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    for col, expr in constraints.items():
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. "
                f"Available: {sorted(df.columns.tolist())}"
            )

        m = _CONSTRAINT_RE.match(str(expr))
        if m is None:
            # No operator prefix — treat as exact string match
            mask &= df[col] == expr
            continue

        op, raw_value = m.group(1), m.group(2)

        # Try numeric parse; fall back to string comparison
        try:
            value: float | str = float(raw_value)
        except ValueError:
            value = raw_value

        if op == ">":
            mask &= df[col] > value
        elif op == ">=":
            mask &= df[col] >= value
        elif op == "<":
            mask &= df[col] < value
        elif op == "<=":
            mask &= df[col] <= value
        elif op == "==":
            mask &= df[col] == value
        elif op == "!=":
            mask &= df[col] != value

    return df[mask]


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Raise KeyError if column is not in the DataFrame.

    Args:
        df: DataFrame to check.
        column: Column name to validate.

    Raises:
        KeyError: With a message listing available columns.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found. "
            f"Available: {sorted(df.columns.tolist())}"
        )


class _TimedRetriever:
    """Transparent wrapper that accumulates wall-clock time spent in retrieve().

    Used by Experiment.run() to measure retrieval latency without modifying
    the Retriever class itself. The wrapper delegates all attribute access to
    the inner retriever so strategies can access .chunks, ._embedder, etc.

    Why a wrapper instead of modifying Retriever: timing is an experiment-runner
    concern, not a retriever concern. Keeping it here avoids polluting the
    Retriever interface and makes the instrumentation easy to remove.
    """

    def __init__(self, retriever: Retriever) -> None:
        """Wrap a Retriever to accumulate timing on retrieve() calls.

        Args:
            retriever: The real Retriever instance to delegate to.
        """
        self._inner = retriever
        self._accumulated_s: float = 0.0

    @property
    def retrieval_ms(self) -> float:
        """Total accumulated retrieval time in milliseconds."""
        return self._accumulated_s * 1000

    def reset(self) -> None:
        """Reset accumulated retrieval time to zero."""
        self._accumulated_s = 0.0

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Delegate to inner retriever while timing the call.

        Args:
            query: The query string to retrieve for.
            top_k: Optional override for number of results.

        Returns:
            List of retrieved chunk dicts from the inner retriever.
        """
        t0 = time.perf_counter()
        result = self._inner.retrieve(query, top_k=top_k)
        t1 = time.perf_counter()
        self._accumulated_s += (t1 - t0)
        return result

    def __getattr__(self, name: str):
        """Delegate attribute access to inner retriever for transparency.

        Strategies may access retriever.chunks or other attributes — this
        ensures the wrapper is invisible to downstream code.
        """
        return getattr(self._inner, name)


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
        retrieval_top_k: int = 5,
        reranker: Reranker | None = None,
        reranker_top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        dataset_name: str | None = None,
        dataset_sample_seed: int | None = None,
        llm_provider: str | None = None,
        llm_host: str | None = None,
    ) -> None:
        """Initialize the experiment with component lists.

        Args:
            chunkers: List of Chunker implementations to test.
            embedders: List of Embedder implementations to test.
            models: List of Ollama model names (e.g., ['qwen3:0.6b', 'qwen3:4b']).
            strategies: List of Strategy implementations to test.
            scorer: A single Scorer implementation for evaluating answers.
            retrieval_top_k: Number of chunks to retrieve per query.
            reranker: Optional Reranker to re-score retrieved chunks before
                      passing to the LLM. When None, behavior is identical to
                      the pre-reranker pipeline.
            reranker_top_k: Number of top chunks to keep after reranking.
                            Required when reranker is not None.
            retrieval_mode: Retrieval mode — "hybrid", "dense", or "sparse".
            dataset_name: Name of the dataset (e.g., "hotpotqa"), or None for CSV.
            dataset_sample_seed: Random seed used for dataset sampling, or None.
            llm_provider: LLM backend provider name (e.g., "ollama").
            llm_host: LLM host URL, or None for local.

        Raises:
            TypeError: If any component doesn't implement its Protocol.
            ValueError: If retrieval_mode is invalid, or reranker is set
                        without reranker_top_k.
        """
        if retrieval_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"Invalid retrieval_mode '{retrieval_mode}'. "
                "Must be one of: ('hybrid', 'dense', 'sparse')"
            )
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
        # Reranker validation: if provided, must implement protocol and have top_k
        if reranker is not None:
            if not isinstance(reranker, Reranker):
                raise TypeError(f"{reranker} does not implement the Reranker protocol")
            if reranker_top_k is None:
                raise ValueError(
                    "reranker_top_k is required when reranker is provided"
                )

        self._chunkers = chunkers
        self._embedders = embedders
        self._models = models
        self._strategies = strategies
        self._scorer = scorer
        self._retrieval_top_k = retrieval_top_k
        self._reranker = reranker
        self._reranker_top_k = reranker_top_k
        self._retrieval_mode = retrieval_mode
        self._dataset_name = dataset_name
        self._dataset_sample_seed = dataset_sample_seed
        self._llm_provider = llm_provider
        self._llm_host = llm_host or "local"
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
                num_chunks = len(chunks)

                for embedder in self._embedders:
                    # Build/cache retriever per (doc, chunker, embedder)
                    cache_key = (doc_hash, chunker.name, embedder.name)
                    if cache_key not in index_cache:
                        index_cache[cache_key] = Retriever(
                            chunks, embedder, self._retrieval_top_k,
                            mode=self._retrieval_mode,
                        )
                    retriever = index_cache[cache_key]

                    # Pre-compute embedder metadata once per embedder
                    embedder_meta = parse_embedder_name(
                        embedder.name, embedder.dimension
                    )

                    for query in self._queries:
                        # Retrieve once, share with features and metadata
                        retrieved = retriever.retrieve(query["text"])

                        # Optional reranking stage: re-score and truncate
                        # Time reranking separately — tells users if reranking
                        # is a significant cost vs retrieval and generation
                        if self._reranker is not None:
                            rerank_t0 = time.perf_counter()
                            reranked = self._reranker.rerank(
                                query["text"], retrieved, self._reranker_top_k
                            )
                            rerank_t1 = time.perf_counter()
                            reranking_latency_ms = (rerank_t1 - rerank_t0) * 1000
                            # Compute rerank feature columns from reranked output
                            rerank_scores = [r["rerank_score"] for r in reranked]
                            if rerank_scores:
                                mean_rerank = sum(rerank_scores) / len(rerank_scores)
                                var_rerank = (
                                    sum((s - mean_rerank) ** 2 for s in rerank_scores)
                                    / len(rerank_scores)
                                )
                            else:
                                mean_rerank = None
                                var_rerank = None
                            # Use reranked chunks for features and context
                            final_chunks = reranked
                        else:
                            mean_rerank = None
                            var_rerank = None
                            # None (not 0.0) distinguishes "not configured" from
                            # "configured but instant" — matches mean_rerank_score pattern
                            reranking_latency_ms = None
                            final_chunks = retrieved

                        # Extract features using final chunks (reranked or original)
                        features = extract_features(
                            query["text"], doc["text"], retriever,
                            retrieved=final_chunks,
                        )

                        # Wrap retriever for per-call retrieval timing —
                        # strategies call retriever.retrieve() internally, and
                        # this wrapper accumulates that time transparently
                        timed_retriever = _TimedRetriever(retriever)

                        for strategy in self._strategies:
                            for model in self._models:
                                count += 1

                                # Reset retrieval timer before each strategy call
                                # so we measure only this (strategy, model) pair
                                timed_retriever.reset()

                                # Time strategy execution — perf_counter for monotonic, sub-μs resolution
                                t0 = time.perf_counter()
                                answer = strategy.run(
                                    query["text"], timed_retriever, model
                                )
                                t1 = time.perf_counter()
                                strategy_latency_ms = (t1 - t0) * 1000

                                # Granular breakdown: retrieval vs generation
                                retrieval_latency_ms = timed_retriever.retrieval_ms
                                # Generation ≈ strategy time minus retrieval time
                                # Includes LLM inference + prompt construction + filtering
                                generation_latency_ms = strategy_latency_ms - retrieval_latency_ms

                                # Time scorer execution separately — users need to know
                                # where latency comes from (generation vs evaluation)
                                t2 = time.perf_counter()
                                scores = self._scorer.score(
                                    query["text"], doc["text"], answer
                                )
                                t3 = time.perf_counter()
                                scorer_latency_ms = (t3 - t2) * 1000

                                total_latency_ms = strategy_latency_ms + scorer_latency_ms

                                if progress:
                                    print(f"[{count}/{total}] {strategy.name} / "
                                          f"{model} / {query['text'][:50]}... "
                                          f"({total_latency_ms:.0f}ms)")

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
                                    "strategy_latency_ms": strategy_latency_ms,
                                    "retrieval_latency_ms": retrieval_latency_ms,
                                    "generation_latency_ms": generation_latency_ms,
                                    "reranking_latency_ms": reranking_latency_ms,
                                    "scorer_latency_ms": scorer_latency_ms,
                                    "total_latency_ms": total_latency_ms,
                                    "timestamp": datetime.now().isoformat(),
                                    # Pipeline metadata
                                    **parse_chunker_name(chunker.name),
                                    "num_chunks": num_chunks,
                                    **embedder_meta,
                                    "mean_rerank_score": mean_rerank,
                                    "var_rerank_score": var_rerank,
                                    **build_retrieval_metadata(
                                        self._retrieval_mode,
                                        self._retrieval_top_k,
                                        len(retrieved),
                                    ),
                                    **build_context_metadata(final_chunks),
                                    **parse_model_name(model),
                                    **build_llm_context_metadata(
                                        model,
                                        self._llm_provider,
                                        self._llm_host,
                                        sum(len(c.get("text", "")) for c in final_chunks),
                                    ),
                                    **build_reranker_metadata(
                                        self._reranker.name if self._reranker else None,
                                        self._reranker_top_k,
                                    ),
                                    **parse_scorer_name(self._scorer.name),
                                    "llm_provider": self._llm_provider,
                                    "llm_host": self._llm_host,
                                    **build_dataset_metadata(
                                        self._dataset_name,
                                        self._dataset_sample_seed,
                                    ),
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

    def filter(self, constraints: dict[str, str]) -> ExperimentResult:
        """Return a new ExperimentResult with rows matching all constraints.

        Constraint syntax: ``{"quality": ">3.5", "total_latency_ms": "<5000",
        "model": "qwen3:4b"}``. Operators: ``>``, ``>=``, ``<``, ``<=``,
        ``==``, ``!=``. String values without operators do exact match.

        Args:
            constraints: Mapping of column name to constraint expression.

        Returns:
            New ExperimentResult with matching rows (may be empty).

        Raises:
            KeyError: If a constraint references a non-existent column.
        """
        filtered = _apply_constraints(self.df, constraints)
        return ExperimentResult(filtered.reset_index(drop=True))

    def best_config(
        self,
        metric: str = "quality",
        *,
        maximize: bool = True,
        constraints: dict[str, str] | None = None,
    ) -> dict:
        """Return the config with the best mean score for the given metric.

        Enhanced version: returns a dict (not tuple) with all 4 axis values
        plus the metric's mean value. Accepts optional constraints and a
        maximize/minimize flag.

        Args:
            metric: The metric column to optimize (default: 'quality').
            maximize: If True, find highest mean. If False, find lowest
                (useful for latency where lower is better).
            constraints: Optional constraint dict to filter rows first.

        Returns:
            Dict with keys: chunker, embedder, strategy, model,
            and ``mean_{metric}`` with the winning config's mean value.

        Raises:
            KeyError: If metric column not found.
            ValueError: If no rows match constraints or DataFrame is empty.
        """
        _validate_column(self.df, metric)

        df = self.df
        if constraints:
            df = _apply_constraints(df, constraints)

        if df.empty:
            raise ValueError(
                f"No rows match the given constraints. "
                f"Cannot find best config for '{metric}'."
            )

        # Find config axes present in the data
        axes = [a for a in _CONFIG_AXES if a in df.columns]
        if not axes:
            raise ValueError("No config axes (chunker, embedder, strategy, model) found in data.")

        grouped = df.groupby(axes)[metric].mean()
        # Drop NaN means (configs with all NaN for this metric)
        grouped = grouped.dropna()
        if grouped.empty:
            raise ValueError(f"All values for '{metric}' are NaN.")

        best_idx = grouped.idxmax() if maximize else grouped.idxmin()
        best_value = grouped[best_idx]

        # Build result dict
        if len(axes) == 1:
            best_idx = (best_idx,)
        result = dict(zip(axes, best_idx))
        result[f"mean_{metric}"] = best_value
        return result

    def configs_above(self, metric: str, threshold: float) -> ExperimentResult:
        """Return an ExperimentResult with configs whose mean metric >= threshold.

        Groups by (chunker, embedder, strategy, model), computes the mean,
        and keeps only configs meeting the threshold.

        Args:
            metric: Column to evaluate.
            threshold: Minimum mean value (inclusive).

        Returns:
            ExperimentResult with qualifying rows.

        Raises:
            KeyError: If metric column not found (non-empty DataFrame only).
        """
        if self.df.empty:
            return ExperimentResult(pd.DataFrame())
        _validate_column(self.df, metric)

        axes = [a for a in _CONFIG_AXES if a in self.df.columns]
        means = self.df.groupby(axes)[metric].mean()
        qualifying = means[means >= threshold].index
        if len(qualifying) == 0:
            return ExperimentResult(pd.DataFrame())

        # Filter original rows to only qualifying configs
        if len(axes) == 1:
            mask = self.df[axes[0]].isin(qualifying)
        else:
            keys = set(qualifying)
            mask = self.df[axes].apply(tuple, axis=1).isin(keys)
        return ExperimentResult(self.df[mask].reset_index(drop=True))

    def configs_below(self, metric: str, threshold: float) -> ExperimentResult:
        """Return an ExperimentResult with configs whose mean metric <= threshold.

        Groups by (chunker, embedder, strategy, model), computes the mean,
        and keeps only configs meeting the threshold.

        Args:
            metric: Column to evaluate.
            threshold: Maximum mean value (inclusive).

        Returns:
            ExperimentResult with qualifying rows.

        Raises:
            KeyError: If metric column not found (non-empty DataFrame only).
        """
        if self.df.empty:
            return ExperimentResult(pd.DataFrame())
        _validate_column(self.df, metric)

        axes = [a for a in _CONFIG_AXES if a in self.df.columns]
        means = self.df.groupby(axes)[metric].mean()
        qualifying = means[means <= threshold].index
        if len(qualifying) == 0:
            return ExperimentResult(pd.DataFrame())

        if len(axes) == 1:
            mask = self.df[axes[0]].isin(qualifying)
        else:
            keys = set(qualifying)
            mask = self.df[axes].apply(tuple, axis=1).isin(keys)
        return ExperimentResult(self.df[mask].reset_index(drop=True))

    def budget_analysis(
        self,
        quality_metric: str,
        cost_metric: str,
        budget: float,
        *,
        maximize_quality: bool = True,
    ) -> pd.DataFrame:
        """Find configs within a cost budget, ranked by quality.

        For each config, computes mean quality and mean cost. Filters to
        configs where mean cost <= budget. Returns sorted by quality.

        Args:
            quality_metric: Column for quality (e.g., "quality").
            cost_metric: Column for cost (e.g., "total_latency_ms").
            budget: Maximum mean cost allowed.
            maximize_quality: Sort order for quality (True = best first).

        Returns:
            DataFrame with columns: chunker, embedder, strategy, model,
            mean_{quality_metric}, mean_{cost_metric}. Empty if no
            configs meet budget.

        Raises:
            KeyError: If either metric column not found (non-empty DataFrame only).
        """
        if self.df.empty:
            return pd.DataFrame()
        _validate_column(self.df, quality_metric)
        _validate_column(self.df, cost_metric)

        axes = [a for a in _CONFIG_AXES if a in self.df.columns]
        grouped = self.df.groupby(axes).agg(
            **{
                f"mean_{quality_metric}": (quality_metric, "mean"),
                f"mean_{cost_metric}": (cost_metric, "mean"),
            }
        ).reset_index()

        # Filter by budget
        within_budget = grouped[grouped[f"mean_{cost_metric}"] <= budget]
        if within_budget.empty:
            return pd.DataFrame()

        # Sort by quality
        result = within_budget.sort_values(
            f"mean_{quality_metric}", ascending=not maximize_quality
        ).reset_index(drop=True)
        return result

    def pareto_front(
        self,
        quality_metric: str,
        cost_metric: str,
        *,
        maximize_quality: bool = True,
        minimize_cost: bool = True,
    ) -> pd.DataFrame:
        """Compute the Pareto frontier — non-dominated configs.

        A config is Pareto-optimal if no other config is strictly better on
        both quality and cost. Uses per-config means as the comparison points.

        Args:
            quality_metric: Column for quality dimension.
            cost_metric: Column for cost dimension.
            maximize_quality: Whether higher quality is better (default True).
            minimize_cost: Whether lower cost is better (default True).

        Returns:
            DataFrame with Pareto-optimal configs, sorted by quality descending.
            Columns: chunker, embedder, strategy, model,
            mean_{quality_metric}, mean_{cost_metric}.

        Raises:
            KeyError: If either metric column not found (non-empty DataFrame only).
        """
        if self.df.empty:
            return pd.DataFrame()
        _validate_column(self.df, quality_metric)
        _validate_column(self.df, cost_metric)

        axes = [a for a in _CONFIG_AXES if a in self.df.columns]
        grouped = self.df.groupby(axes).agg(
            **{
                f"mean_{quality_metric}": (quality_metric, "mean"),
                f"mean_{cost_metric}": (cost_metric, "mean"),
            }
        ).reset_index().dropna()

        if grouped.empty:
            return pd.DataFrame()

        q_col = f"mean_{quality_metric}"
        c_col = f"mean_{cost_metric}"

        # For domination check, convert to "higher is better" for both dimensions
        q_vals = grouped[q_col].values if maximize_quality else -grouped[q_col].values
        c_vals = -grouped[c_col].values if minimize_cost else grouped[c_col].values

        n = len(grouped)
        is_pareto = [True] * n
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # j dominates i if j is >= on both and strictly > on at least one
                if (q_vals[j] >= q_vals[i] and c_vals[j] >= c_vals[i] and
                        (q_vals[j] > q_vals[i] or c_vals[j] > c_vals[i])):
                    is_pareto[i] = False
                    break

        pareto = grouped[is_pareto].sort_values(q_col, ascending=False)
        return pareto.reset_index(drop=True)

    def rank(
        self,
        metric: str,
        *,
        ascending: bool = False,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank all configs by mean metric value.

        Args:
            metric: Column to rank by.
            ascending: If True, lowest first (for latency). Default False.
            top_n: Limit output to top N configs. None for all.

        Returns:
            DataFrame with columns: rank, chunker, embedder, strategy, model,
            mean, std, count.

        Raises:
            KeyError: If metric column not found (non-empty DataFrame only).
        """
        if self.df.empty:
            return pd.DataFrame()
        _validate_column(self.df, metric)

        axes = [a for a in _CONFIG_AXES if a in self.df.columns]
        grouped = self.df.groupby(axes)[metric].agg(
            ["mean", "std", "count"]
        ).reset_index().dropna(subset=["mean"])

        if grouped.empty:
            return pd.DataFrame()

        ranked = grouped.sort_values("mean", ascending=ascending).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))

        if top_n is not None:
            ranked = ranked.head(top_n)

        return ranked

    def summary(self) -> pd.DataFrame:
        """Group by all four axes and compute aggregate statistics.

        Returns:
            Summary DataFrame with mean, std, min, max, count for score columns.
        """
        if self.df.empty:
            print("No data.")
            return self.df
        score_cols = [c for c in ["faithfulness", "relevance", "conciseness", "quality"]
                      if c in self.df.columns]
        summary = self.df.groupby(
            ["chunker", "embedder", "strategy", "model"]
        )[score_cols].agg(["mean", "std", "min", "max", "count"]).round(3)
        print(summary.to_string())
        return summary

    def compare_strategies(self, metric: str = "quality") -> pd.DataFrame:
        """Group by strategy only, aggregate metric. Print ranked table.

        Args:
            metric: The metric to compare (default: 'quality').

        Returns:
            DataFrame ranked by mean metric score descending.
        """
        if self.df.empty or metric not in self.df.columns:
            print("No data for comparison.")
            return self.df
        result = self.df.groupby("strategy")[metric].agg(
            ["mean", "std", "count"]
        ).round(3).sort_values("mean", ascending=False)
        print(result.to_string())
        return result

    def compare_models(self, metric: str = "quality") -> pd.DataFrame:
        """Group by model only, aggregate metric. Print ranked table.

        Args:
            metric: The metric to compare (default: 'quality').

        Returns:
            DataFrame ranked by mean metric score descending.
        """
        if self.df.empty or metric not in self.df.columns:
            print("No data for comparison.")
            return self.df
        result = self.df.groupby("model")[metric].agg(
            ["mean", "std", "count"]
        ).round(3).sort_values("mean", ascending=False)
        print(result.to_string())
        return result

    def heatmap(self, rows: str, cols: str, values: str = "quality",
                save_path: Path | None = None) -> None:
        """Create a matplotlib heatmap of the pivot table.

        Args:
            rows: Column name for heatmap rows.
            cols: Column name for heatmap columns.
            values: Column name for cell values (default: 'quality').
            save_path: If provided, save the figure to this path instead of showing.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for headless environments
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting: pip install matplotlib")

        pivot = self.pivot(rows, cols, values)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Annotate cells with values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        plt.colorbar(im)
        plt.title(f"{values} by {rows} × {cols}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def per_query(self, metric: str = "quality") -> pd.DataFrame:
        """For each query, show best/worst config and score spread.

        Useful for identifying queries where strategy choice matters most.

        Args:
            metric: The metric to analyze (default: 'quality').

        Returns:
            DataFrame sorted by spread (descending).
        """
        if self.df.empty or metric not in self.df.columns:
            print("No data.")
            return pd.DataFrame()
        results = []
        for query, group in self.df.groupby("query_text"):
            best = group.loc[group[metric].idxmax()]
            worst = group.loc[group[metric].idxmin()]
            results.append({
                "query": str(query)[:80],
                "best_config": f"{best['strategy']}/{best['model']}",
                "best_score": best[metric],
                "worst_config": f"{worst['strategy']}/{worst['model']}",
                "worst_score": worst[metric],
                "spread": best[metric] - worst[metric],
            })
        result_df = pd.DataFrame(results).sort_values("spread", ascending=False)
        print(result_df.to_string(index=False))
        return result_df

    def strategy_vs_size(self, metric: str = "quality") -> pd.DataFrame:
        """The core research question: when does strategy beat size?

        Pivot strategy x model showing where small models + smart strategies
        beat large models + simple strategies.

        Args:
            metric: The metric to pivot (default: 'quality').

        Returns:
            Pivot table DataFrame of strategy x model.
        """
        pivot = self.pivot("strategy", "model", metric)
        print(pivot.round(3).to_string())
        return pivot

    def to_csv(self, path: Path) -> None:
        """Export results to CSV for sharing or further analysis.

        Args:
            path: File path for the CSV output.
        """
        self.df.to_csv(path, index=False)

    def merge(self, other: ExperimentResult) -> ExperimentResult:
        """Combine two ExperimentResults from different runs.

        Args:
            other: Another ExperimentResult to merge with this one.

        Returns:
            New ExperimentResult with concatenated DataFrames.
        """
        return ExperimentResult(pd.concat([self.df, other.df], ignore_index=True))

    def latency_report(self) -> pd.DataFrame:
        """Mean/std/min/max latency grouped by (strategy, model), sorted fastest-first.

        Includes all available latency columns: total, strategy, retrieval,
        generation, reranking, and scorer. Reranking is excluded when all
        values are None (no reranker configured).

        Returns:
            Grouped DataFrame with latency statistics, or empty DataFrame if no data.
        """
        if self.df.empty or "total_latency_ms" not in self.df.columns:
            return pd.DataFrame()

        # Include all latency columns present in the data
        latency_cols = [
            col for col in [
                "total_latency_ms",
                "strategy_latency_ms",
                "retrieval_latency_ms",
                "generation_latency_ms",
                "reranking_latency_ms",
                "scorer_latency_ms",
            ]
            if col in self.df.columns
            # Skip columns where every value is None (e.g. reranking with no reranker)
            and self.df[col].notna().any()
        ]

        report = self.df.groupby(["strategy", "model"])[latency_cols].agg(
            ["mean", "std", "min", "max"]
        ).round(1).sort_values(("total_latency_ms", "mean"), ascending=True)
        return report

    def time_vs_quality(self) -> pd.DataFrame:
        """Mean quality and mean latency per config, sorted by quality descending.

        Shows the tradeoff between answer quality and response time so users
        can judge whether slower configs are worth the wait.

        Returns:
            DataFrame with mean quality and latency per (strategy, model),
            or empty DataFrame if no data.
        """
        if self.df.empty:
            return pd.DataFrame()

        needed = {"quality", "total_latency_ms"}
        if not needed.issubset(set(self.df.columns)):
            return pd.DataFrame()

        table = self.df.groupby(["strategy", "model"]).agg(
            mean_quality=("quality", "mean"),
            mean_latency_ms=("total_latency_ms", "mean"),
        ).round(3).sort_values("mean_quality", ascending=False)
        return table
