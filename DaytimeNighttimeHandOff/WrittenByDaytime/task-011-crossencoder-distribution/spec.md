# Spec: task-011 — Cross-Encoder Filter and Distribution Analyzer

## What

Add two query validation components:

1. **`CrossEncoderFilter`** — scores each (query, source passage) pair using a small
   cross-encoder reranker model. More precise than round-trip retrieval at identifying
   queries that are semantically disconnected from their source.

2. **`DistributionAnalyzer`** — analyzes a query set as a whole for coverage, diversity,
   type balance, and other set-level quality signals. Produces a report rather than
   filtering individual queries.

### New files to create:

1. `src/query_filters/cross_encoder.py` — CrossEncoderFilter
2. `src/query_analysis/__init__.py` — new package
3. `src/query_analysis/distribution.py` — DistributionAnalyzer
4. `src/query_filters/__init__.py` — update with import
5. `tests/test_cross_encoder_filter.py`
6. `tests/test_distribution_analyzer.py`

## Why

### CrossEncoderFilter

Cross-encoder reranking is the InPars-v2 approach to query quality filtering
(Bonifacio et al., 2023, arxiv:2301.01820). A cross-encoder jointly encodes
the (query, passage) pair and outputs a relevance score, which is more precise than
bi-encoder similarity (round-trip) because it models token-level interactions between
query and passage.

Key research finding from InPars-v2: cross-encoder filtering improved downstream
quality more than generating more queries. Quality filtering > quantity.

The model is small (~100M params, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) and
runs locally with no API calls. Fits in the validation pipeline between the heuristic
pre-filter (Layer 1) and distribution analysis (Layer 3).

### DistributionAnalyzer

Most RAG evaluation pipelines don't do systematic distribution analysis of their test
sets (noted as under-used in Thakur et al., 2021, arxiv:2104.08663 — the BEIR paper).
This catches set-level problems that per-query filters miss: topic skew, difficulty
homogeneity, corpus coverage gaps, type imbalance.

This is NOT a QueryFilter (doesn't filter individual queries). It's an analysis tool
that produces a report — a dict of metrics and optionally a printed summary. Separate
package (`src/query_analysis/`) because analysis utilities will grow over time.

## Dependencies to Install

1. `sentence-transformers` — already installed (includes CrossEncoder support)

The `cross-encoder/ms-marco-MiniLM-L-6-v2` model will be downloaded automatically
by sentence-transformers on first use (~25MB). No separate install step needed.

**Use the install-package skill** if any additional deps are needed.

## File Details

### `src/query_filters/cross_encoder.py`

```python
"""Cross-encoder relevance filter for generated queries.

Scores each (query, source passage) pair using a cross-encoder reranker model.
More precise than bi-encoder round-trip filtering because cross-encoders jointly
encode the query and passage, modeling token-level interactions.

Based on the InPars-v2 filtering methodology (Bonifacio et al., 2023,
arxiv:2301.01820). Key finding: cross-encoder filtering improved downstream
quality more than generating more queries.

Uses a small local model (~100M params) — no API calls required.
Ref: MonoT5 reranker architecture (Nogueira et al., 2020, arxiv:2003.06713).
"""
```

**Class: `CrossEncoderFilter`**

Constructor:
```python
def __init__(
    self,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    threshold: float = 0.5,
    use_full_doc: bool = False,
) -> None:
```

- `model_name` — the cross-encoder model to use. Default is MS MARCO MiniLM, a small
  (~25MB) model trained on query-passage relevance. Well-suited for this task.
- `threshold` — minimum relevance score (0-1) to keep a query. The MS MARCO model
  outputs logits; apply sigmoid to get a 0-1 score. Queries below threshold are filtered.
  Default 0.5 is a reasonable starting point — tunable.
- `use_full_doc` — if True, score against the full document text. If False (default),
  score against individual chunks from the document and take the max score. Chunked
  scoring is more precise (the query may be about one section, not the whole doc) but
  requires a chunker.

**SDK setup:**

```python
from sentence_transformers import CrossEncoder

# In __init__:
self._model = CrossEncoder(model_name)
```

Properties:
- `name` → `f"cross_encoder:{self._model_name.split('/')[-1]}:t={self._threshold}"`
  (e.g., `"cross_encoder:ms-marco-MiniLM-L-6-v2:t=0.5"`)

**`filter` method:**

```python
def filter(
    self,
    queries: list[Query],
    documents: list[Document],
) -> list[Query]:
```

Algorithm:
1. Build document lookup: `{doc.title: doc for doc in documents}`
2. For each query:
   a. Look up source document via `query.source_doc_title`
   b. If `use_full_doc`:
      - Score the pair: `self._model.predict([(query.text, doc.text)])`
      - Apply sigmoid if the model returns logits
   c. If not `use_full_doc`:
      - Split doc.text into passages. Use a simple split: paragraphs (split on `\n\n`).
        Do NOT use a Chunker protocol instance — keep this self-contained. The paragraph
        split is just for scoring granularity, not for retrieval.
      - Score all pairs: `self._model.predict([(query.text, p) for p in paragraphs])`
      - Take the max score across paragraphs
   d. Keep query if score >= threshold
3. Return filtered list

**Edge cases:**
- Source doc not in documents → discard query (log warning to stderr)
- Empty query list → return empty list
- Empty document list → return empty list
- Very long document with `use_full_doc=True` → cross-encoders have a max token limit
  (~512 tokens for MiniLM). The model will truncate automatically. This is acceptable —
  for very long docs, the chunked approach (`use_full_doc=False`) is better anyway.

### `src/query_analysis/__init__.py`

```python
"""Query set analysis utilities.

Tools for analyzing the quality and distribution of generated query sets
at the set level (not per-query filtering).
"""
```

### `src/query_analysis/distribution.py`

```python
"""Distribution analysis for generated query sets.

Analyzes a query set as a whole for coverage, diversity, type balance,
and other set-level quality signals. Produces a report rather than
filtering individual queries.

Set-level analysis is under-used in RAG evaluation pipelines despite being
low-cost and highly informative (noted in Thakur et al., 2021, arxiv:2104.08663).
Catches problems that per-query filters miss: topic skew, corpus coverage gaps,
query type imbalance, and difficulty homogeneity.
"""
```

**Class: `DistributionAnalyzer`**

Constructor:
```python
def __init__(
    self,
    embedder: Embedder | None = None,
) -> None:
```

- `embedder` — optional embedder for corpus coverage and diversity analysis. If None,
  skip embedding-based analyses (only run statistical checks). This keeps the analyzer
  usable even without an embedder.

Properties:
- `name` → `"distribution_analyzer"`

**`analyze` method:**

```python
def analyze(
    self,
    queries: list[Query],
    documents: list[Document] | None = None,
) -> dict:
```

Returns a dict with the following keys (each is a sub-dict or value):

```python
{
    "total_queries": int,
    "type_distribution": {
        "factoid": {"count": int, "fraction": float},
        "reasoning": {"count": int, "fraction": float},
        "multi_context": {"count": int, "fraction": float},
        "conditional": {"count": int, "fraction": float},
        "other": {"count": int, "fraction": float},
    },
    "length_stats": {
        "mean": float,
        "median": float,
        "min": int,
        "max": int,
        "std": float,
    },
    "docs_with_queries": int,         # how many unique source docs are represented
    "docs_without_queries": int,      # how many docs in corpus have no queries (if documents provided)
    "queries_per_doc": {
        "mean": float,
        "min": int,
        "max": int,
    },
    "lexical_diversity": float,       # type-token ratio across all query texts
    "duplicate_count": int,           # exact string duplicates
    "warnings": [str],               # list of detected issues
}
```

**If embedder is provided, add:**

```python
{
    "embedding_diversity": {
        "mean_pairwise_distance": float,  # mean cosine distance between query embeddings
        "cluster_count": int,             # number of clusters (DBSCAN or simple k-means)
    },
    "corpus_coverage": float,  # fraction of doc embeddings that have a query within cosine threshold
}
```

Algorithm:
1. **Type distribution:** Count queries by `query_type`. Compare to expected balance.
   Warning if any type is <5% or >60% of total.
2. **Length stats:** Word count per query. Warning if mean < 7 or > 30.
3. **Document coverage:** Count unique `source_doc_title` values. If `documents` provided,
   compute how many docs have zero queries. Warning if coverage < 80%.
4. **Queries per doc:** Group by `source_doc_title`, compute stats. Warning if max/min
   ratio > 5 (highly unbalanced).
5. **Lexical diversity:** Concatenate all query texts, tokenize, compute type-token ratio
   (unique words / total words). Warning if < 0.3 (very repetitive).
6. **Duplicates:** Count exact string duplicates (case-insensitive, stripped).
   Warning if > 0.
7. **Embedding diversity** (if embedder provided):
   - Embed all queries
   - Compute mean pairwise cosine distance
   - Run simple clustering (use sklearn KMeans with k = min(10, len(queries)//5))
   - Warning if mean pairwise distance < 0.3 (queries are too similar)
8. **Corpus coverage** (if embedder and documents provided):
   - Embed all document titles (or first 100 words of each doc)
   - For each doc embedding, check if any query embedding is within cosine distance 0.5
   - Coverage = fraction of docs with at least one nearby query
   - Warning if < 0.7

**`print_report` method:**

```python
def print_report(self, analysis: dict) -> None:
```

Pretty-print the analysis dict. Format:

```
RAGBench Query Set Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total queries:          500
Type distribution:      factoid: 180 (36%), reasoning: 140 (28%),
                        multi_context: 120 (24%), conditional: 60 (12%)
Length (words):         mean=12.3, median=11, range=[5, 42]
Document coverage:     185/200 docs (92.5%)
Queries per doc:       mean=2.5, min=1, max=8
Lexical diversity:     0.45 (type-token ratio)
Duplicates:            0

Warnings:
  ⚠ 15 documents have no queries — consider regenerating for these docs

Embedding analysis:
  Mean pairwise distance: 0.62
  Cluster count:          8
  Corpus coverage:        0.89
```

### `src/query_filters/__init__.py`

Update to include all filters:

```python
from src.query_filters.round_trip import RoundTripFilter
from src.query_filters.heuristic import HeuristicFilter
from src.query_filters.cross_encoder import CrossEncoderFilter

__all__ = ["RoundTripFilter", "HeuristicFilter", "CrossEncoderFilter"]
```

## Files NOT to Touch

- `src/protocols.py` — DistributionAnalyzer is NOT a QueryFilter (different interface).
  No protocol changes needed.
- `src/experiment.py` — no changes
- `src/query_filters/round_trip.py` — already created in task-008
- `src/query_filters/heuristic.py` — already created in task-008

## Tests

### `tests/test_cross_encoder_filter.py`

The cross-encoder model runs locally (~25MB), so tests can use the real model with
small inputs. No mocking needed for the model itself.

1. **test_keeps_relevant_query** — create a passage about Paris being the capital of
   France. Query: "What is the capital of France?" Assert it passes the filter.
2. **test_rejects_irrelevant_query** — same passage about Paris. Query: "How do you
   bake a chocolate cake?" Assert it's filtered out.
3. **test_threshold_tuning** — with a very high threshold (0.99), even a somewhat
   relevant query may be filtered. With a low threshold (0.1), most queries pass.
   Assert different results at different thresholds.
4. **test_chunked_vs_full_doc** — with `use_full_doc=False`, a query about content in
   paragraph 3 of a multi-paragraph document should score higher (max over paragraphs)
   than with `use_full_doc=True` (diluted by irrelevant paragraphs).
5. **test_empty_queries** — `filter([], docs)` returns `[]`.
6. **test_empty_documents** — `filter(queries, [])` returns `[]`.
7. **test_missing_source_doc** — query references a doc not in the list. Filtered out.
8. **test_name_format** — name includes model name and threshold.
9. **test_protocol_compliance** — `isinstance(CrossEncoderFilter(), QueryFilter)` is True.

### `tests/test_distribution_analyzer.py`

1. **test_type_distribution** — create queries with known type distribution. Assert
   counts and fractions match.
2. **test_length_stats** — queries of known word lengths. Assert mean, median, min, max.
3. **test_document_coverage** — 10 documents, queries for 8 of them. Assert
   `docs_with_queries == 8`, `docs_without_queries == 2`.
4. **test_queries_per_doc** — 3 queries for doc A, 1 for doc B. Assert min=1, max=3.
5. **test_duplicate_detection** — include two identical queries. Assert `duplicate_count == 1`.
6. **test_lexical_diversity** — create queries with repeated vocabulary. Assert
   lexical_diversity is computed correctly.
7. **test_warnings_type_imbalance** — 95% factoid, 5% reasoning. Assert a warning about
   type imbalance.
8. **test_warnings_low_coverage** — queries cover only 50% of documents. Assert warning.
9. **test_no_embedder_skips_embedding_analysis** — analyzer with `embedder=None`. Assert
   result has no `embedding_diversity` or `corpus_coverage` keys.
10. **test_with_embedder** — use HuggingFaceEmbedder. Assert `embedding_diversity` and
    `corpus_coverage` are present and contain reasonable values.
11. **test_print_report_runs** — call `print_report` on a valid analysis dict. Assert
    no exceptions (capture stdout, check it contains "Total queries").

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Research URLs included where research was done
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
