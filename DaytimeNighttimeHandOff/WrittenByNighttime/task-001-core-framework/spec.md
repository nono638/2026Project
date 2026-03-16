# Task 001: Core Framework — Protocols, Retriever, Experiment Runner

## What
Build the foundational framework for the pluggable RAG research tool. Three files:
1. `src/protocols.py` — Protocol definitions for all pluggable component types
2. `src/retriever.py` — Retriever class that wraps FAISS index + embedder + chunks
3. `src/experiment.py` — Experiment runner and ExperimentResult class

Also:
- Move `src/data/features.py` → `src/features.py` (update imports)
- Delete `src/config.py`
- Delete `src/data/` directory (will be replaced by experiment.py)
- Create `tests/test_core.py` with tests using mock Protocol implementations

## Why
The current skeleton has hardcoded imports and a fixed pipeline. The project scope has
shifted to a generalized RAG research tool where users choose their own components.
Every component type (chunker, embedder, strategy, scorer) needs a clean interface so
users can swap them without modifying framework code. This task builds the skeleton that
all other tasks plug into.

## Exact Files to Create

### `src/protocols.py`

Use `typing.Protocol` with `@runtime_checkable` for all interfaces. Protocols over ABCs
because: structural subtyping lets users write classes that happen to have the right methods
without importing or inheriting anything. A research tool should have minimal boilerplate.

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Chunker(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier for this chunker config (e.g., 'semantic:mxbai-embed-large')."""
        ...

    def chunk(self, text: str) -> list[str]:
        """Split document text into chunks."""
        ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'ollama:mxbai-embed-large')."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension. May be detected lazily on first embed() call."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns array of shape (len(texts), dimension)."""
        ...


@runtime_checkable
class Strategy(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'naive', 'self_rag')."""
        ...

    def run(self, query: str, retriever: "Retriever", model: str) -> str:
        """Execute the RAG strategy and return the generated answer.

        Args:
            query: The user's question.
            retriever: A Retriever instance (wraps chunks + index + embedder).
            model: Ollama model name for generation (e.g., 'qwen3:0.6b').
        """
        ...


@runtime_checkable
class Scorer(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'claude:claude-sonnet-4-20250514')."""
        ...

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        """Score a generated answer. Returns dict of metric_name -> score (1-5)."""
        ...
```

**Important:** The `Retriever` forward reference in `Strategy.run` — use `from __future__ import annotations` at the top of the file, or use a string annotation. The Retriever class is defined in `src/retriever.py`, not in protocols.py. Do NOT import Retriever into protocols.py (that would create a circular dependency). Use `TYPE_CHECKING` guard if needed:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.retriever import Retriever
```

### `src/retriever.py`

Concrete class, not a Protocol. Strategies receive this; users don't implement it.

```python
import numpy as np
import faiss

from src.protocols import Embedder


class Retriever:
    """Wraps a FAISS index + embedder + chunks for retrieval.

    Built once per (document, chunker, embedder) triple and cached
    by the Experiment runner to avoid redundant embedding work.
    """

    def __init__(self, chunks: list[str], embedder: Embedder, top_k: int = 5):
        self._chunks = chunks
        self._embedder = embedder
        self._top_k = top_k

        # Build FAISS index
        embeddings = embedder.embed(chunks)
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embedder.dimension)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve top-k chunks for a query.

        Returns list of dicts with 'text', 'score', 'index' keys,
        sorted by descending similarity.
        """
        k = top_k or self._top_k
        query_emb = self._embedder.embed([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self._chunks[idx],
                "score": float(score),
                "index": int(idx),
            })
        return results

    @property
    def chunks(self) -> list[str]:
        """Access the underlying chunks (for building context strings)."""
        return self._chunks
```

### `src/experiment.py`

The central orchestrator. Replaces `src/data/generate.py`.

```python
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.protocols import Chunker, Embedder, Strategy, Scorer
from src.retriever import Retriever
from src.features import extract_features


class Experiment:
    def __init__(
        self,
        chunkers: list[Chunker],
        embedders: list[Embedder],
        models: list[str],
        strategies: list[Strategy],
        scorer: Scorer,
        top_k: int = 5,
    ):
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

    def run(self, progress: bool = True) -> "ExperimentResult":
        """Run the full experiment matrix and return results.

        Iterates: doc × chunker × embedder × query × strategy × model
        Caches FAISS indexes per (doc_hash, chunker.name, embedder.name).
        """
        rows = []
        index_cache: dict[tuple, Retriever] = {}

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

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compare(self) -> pd.DataFrame:
        """Print and return summary comparison grouped by strategy × model."""
        if "quality" not in self.df.columns:
            print("No quality scores found.")
            return self.df
        summary = self.df.groupby(["strategy", "model"])["quality"].agg(
            ["mean", "std", "count"]
        ).round(3)
        print(summary.to_string())
        return summary

    def pivot(self, rows: str, cols: str, values: str = "quality") -> pd.DataFrame:
        """Create a pivot table for analysis."""
        return self.df.pivot_table(index=rows, columns=cols, values=values, aggfunc="mean")

    def to_parquet(self, path: Path) -> None:
        """Save results to Parquet."""
        self.df.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: Path) -> "ExperimentResult":
        """Load results from Parquet."""
        return cls(pd.read_parquet(path))

    def best_config(self, metric: str = "quality") -> pd.Series:
        """Return the config with the highest mean score for the given metric."""
        return self.df.groupby(
            ["chunker", "embedder", "strategy", "model"]
        )[metric].mean().idxmax()
```

### `src/features.py`

Move from `src/data/features.py`. Change the signature to accept a `Retriever` instead
of `(chunks, index)`:

```python
def extract_features(query: str, document: str, retriever: Retriever) -> dict[str, float]:
    """Extract features for a (query, document) pair.

    Args:
        query: The question text.
        document: The full document text.
        retriever: A Retriever instance (used to get retrieval scores).
    """
    retrieved = retriever.retrieve(query)
    scores = [r["score"] for r in retrieved]
    # ... rest unchanged
```

Import `Retriever` from `src.retriever`.

### `tests/test_core.py`

Write tests using mock implementations of each Protocol:

```python
class MockChunker:
    @property
    def name(self) -> str:
        return "mock_chunker"

    def chunk(self, text: str) -> list[str]:
        # Split on double newlines
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class MockEmbedder:
    @property
    def name(self) -> str:
        return "mock_embedder"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        # Deterministic fake embeddings based on text length
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 4).astype(np.float32)


class MockStrategy:
    @property
    def name(self) -> str:
        return "mock_strategy"

    def run(self, query: str, retriever, model: str) -> str:
        retrieved = retriever.retrieve(query)
        return f"Answer based on {len(retrieved)} chunks using {model}"


class MockScorer:
    @property
    def name(self) -> str:
        return "mock_scorer"

    def score(self, query: str, context: str, answer: str) -> dict[str, float]:
        return {"faithfulness": 4.0, "relevance": 3.5, "conciseness": 4.0}
```

**Test cases:**
1. `test_protocol_compliance` — verify mock classes pass `isinstance` checks
2. `test_retriever_build_and_search` — build a Retriever with MockEmbedder, verify retrieve() returns results
3. `test_experiment_runs` — create an Experiment with all mocks, load a tiny corpus (1 doc, 1 query), run it, verify the result DataFrame has the expected columns and 1 row
4. `test_experiment_cartesian_product` — 2 strategies × 2 models = 4 rows for 1 doc × 1 query
5. `test_experiment_result_compare` — verify compare() returns a summary DataFrame
6. `test_experiment_result_parquet_roundtrip` — save to parquet, load back, verify equality

## Files to Delete
- `src/config.py`
- `src/data/__init__.py`
- `src/data/generate.py`
- `src/data/features.py` (moved to `src/features.py`)
- `src/pipeline/__init__.py`
- `src/pipeline/chunking.py`
- `src/pipeline/retrieval.py`
- `src/pipeline/scoring.py`
- `src/pipeline/strategies/__init__.py`
- `src/pipeline/strategies/naive.py`
- `src/pipeline/strategies/self_rag.py`
- `src/pipeline/strategies/multi_query.py`
- `src/pipeline/strategies/corrective.py`
- `src/pipeline/strategies/adaptive.py`

**DO NOT delete these yet** — task-002 will migrate their logic first. In this task, just
create the new framework files alongside the old ones. The old `src/pipeline/` and
`src/data/` directories will be cleaned up in task-002 after migration is complete.

Actually, **revised instruction**: Do NOT delete anything in this task. Only create new
files. Task-002 handles the migration and cleanup.

## What NOT to Touch
- `src/model/` — unchanged until task-005
- `src/app.py` — unchanged until task-005
- `DaytimeNighttimeHandOff/` — don't modify project management files
- `tests/` existing files — only add new test files

## Edge Cases
- `Retriever.__init__` should handle empty chunk lists gracefully (return empty results)
- `Experiment.run()` should handle empty corpus (return ExperimentResult with empty DataFrame)
- `ExperimentResult.compare()` should handle empty DataFrame without crashing
- `extract_features` should handle cases where retriever returns 0 results (scores = [])
