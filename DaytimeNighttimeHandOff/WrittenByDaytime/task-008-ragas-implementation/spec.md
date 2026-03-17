# Spec: task-008 — RAGAS QueryGenerator, Round-Trip Filter, and Heuristic Filter

## What

Implement the first concrete `QueryGenerator` (using RAGAS) and two `QueryFilter`
implementations (round-trip retrieval consistency and heuristic pre-filters). These are
the first implementations of the protocols defined in task-007.

### New files to create:

1. `src/query_generators/ragas.py` — RagasQueryGenerator
2. `src/query_filters/round_trip.py` — RoundTripFilter
3. `src/query_filters/heuristic.py` — HeuristicFilter
4. `src/query_generators/__init__.py` — update with import
5. `src/query_filters/__init__.py` — update with imports
6. `tests/test_ragas_generator.py` — tests (mocked, no real API calls)
7. `tests/test_round_trip_filter.py` — tests
8. `tests/test_heuristic_filter.py` — tests

## Why

RAGAS is the most widely cited synthetic query generation framework for RAG evaluation
(arxiv:2309.15217). It generates seed questions from document chunks and evolves them
into difficulty tiers (factoid, reasoning, multi-context, conditional). This gives us
a credible, citable methodology for our evaluation query set.

Round-trip filtering (from Promptagator, arxiv:2209.11755) is the gold standard quality
filter: a generated query is valid only if it retrieves its source passage. This catches
vague, off-topic, or poorly worded queries cheaply.

Heuristic pre-filters are the universal first layer in any query validation pipeline
(used in InPars, Promptagator, and RAGAS pipelines as standard practice). They catch
degenerate queries — garbled text, copy-paste from source, duplicates, non-questions —
at zero computational cost before more expensive filters run.

All are implementations of protocols from task-007, and all are designed as swappable
components — the experiment pipeline doesn't depend on any specific filter.

## Dependencies to Install

1. `ragas` — the RAGAS evaluation framework
2. Any transitive deps RAGAS pulls in (likely langchain-core, etc.)

**Use the install-package skill** for each: read `.claude/skills/install-package/SKILL.md`.

**Important:** RAGAS has evolved rapidly. Check the installed version's API before coding.
The spec below is based on the RAGAS ~0.1.x / 0.2.x API. If the installed version has a
different API, adapt accordingly but preserve the same external interface (the
`QueryGenerator` protocol). Log any API differences in result.md.

## File Details

### `src/query_generators/ragas.py`

```python
"""RAGAS-based query generation for RAG evaluation.

Uses the RAGAS TestsetGenerator to produce evaluation queries from documents,
with evolution for difficulty diversity. RAGAS is a widely cited framework
(arxiv:2309.15217) which gives the methodology academic credibility.

This is one implementation of the QueryGenerator protocol — the experiment
pipeline doesn't depend on RAGAS specifically.
"""
```

**Class: `RagasQueryGenerator`**

Constructor:
```python
def __init__(
    self,
    generator_model: str = "gpt-4o-mini",
    critic_model: str = "gpt-4o-mini",
    distribution: dict[str, float] | None = None,
) -> None:
```

- `generator_model` — the LLM used to generate seed questions. RAGAS supports OpenAI
  models by default. If the user wants to use a different provider, they can configure
  RAGAS's LLM wrapper before passing it in.
- `critic_model` — the LLM used for evolution/critique.
- `distribution` — controls the proportion of query types. Default:
  ```python
  {"simple": 0.3, "reasoning": 0.3, "multi_context": 0.25, "conditional": 0.15}
  ```

**RAGAS API integration notes:**

RAGAS's `TestsetGenerator` API has changed across versions. Here's the general pattern —
**adapt to whatever version is installed**:

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
# Some versions use different imports — check the installed package

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
testset = generator.generate_with_langchain_docs(documents, test_size=N, distributions=dist)
```

The key things that change between RAGAS versions:
- Import paths for TestsetGenerator and evolution types
- Whether it uses langchain documents or its own document format
- Method names (generate vs generate_with_langchain_docs)
- How distributions are specified

**Approach:** Try the import. If it fails, try the alternative import path. Log which
API version was used. If neither works, raise `ImportError` with a helpful message
suggesting the user check their RAGAS version.

Properties:
- `name` → `f"ragas:{self._generator_model}"` (e.g., `"ragas:gpt-4o-mini"`)

**`generate` method:**

```python
def generate(
    self,
    documents: list[Document],
    queries_per_doc: int = 5,
) -> list[Query]:
```

- Convert Documents to whatever format RAGAS expects (likely langchain Documents or
  RAGAS's own document format). Map `doc.title` to metadata.
- Call RAGAS's testset generator with `test_size = len(documents) * queries_per_doc`
- Map RAGAS's output back to our `Query` dataclass:
  - `text` ← the generated question
  - `query_type` ← map RAGAS evolution type to our taxonomy:
    - RAGAS "simple" → "factoid"
    - RAGAS "reasoning" → "reasoning"
    - RAGAS "multi_context" → "multi_context"
    - RAGAS "conditional" (if available) → "conditional"
    - Anything else → "factoid" (safe default)
  - `source_doc_title` ← extract from RAGAS's context/source metadata
  - `reference_answer` ← RAGAS generates ground truth answers; use them
  - `generator_name` ← `self.name`
- Return list of Query objects

**Environment variable handling:**
RAGAS uses OpenAI by default, so `OPENAI_API_KEY` must be set. If it's not set,
raise `ValueError` with: "Set OPENAI_API_KEY environment variable for RAGAS query
generation. RAGAS uses OpenAI models by default."

### `src/query_filters/round_trip.py`

```python
"""Round-trip consistency filter for generated queries.

Validates that a generated query retrieves its source document/passage when
run through the retriever. Queries that don't retrieve their source are
likely vague, off-topic, or poorly worded.

Based on the Promptagator approach (arxiv:2209.11755). This is one
implementation of the QueryFilter protocol.
"""
```

**Class: `RoundTripFilter`**

Constructor:
```python
def __init__(
    self,
    chunker: Chunker,
    embedder: Embedder,
    top_k: int = 5,
    min_score: float = 0.0,
) -> None:
```

- `chunker` — used to chunk documents for building the retriever
- `embedder` — used to embed chunks and queries
- `top_k` — how many results to check for the source document
- `min_score` — optional minimum similarity score threshold (0.0 = any match counts)

Properties:
- `name` → `f"round_trip:{self._embedder.name}:k={self._top_k}"`

**`filter` method:**

```python
def filter(
    self,
    queries: list[Query],
    documents: list[Document],
) -> list[Query]:
```

Algorithm:
1. Build a lookup: `doc_by_title = {doc.title: doc for doc in documents}`
2. Build a combined retriever across all documents:
   - Concatenate all chunks from all documents, tracking which doc each chunk came from
   - `all_chunks = []` and `chunk_doc_map = []` (parallel list mapping chunk index → doc title)
   - For each document, chunk it, append chunks to `all_chunks`, append doc title
     for each chunk to `chunk_doc_map`
   - Build one `Retriever(all_chunks, embedder, top_k)` — single FAISS index
3. For each query:
   - Retrieve top_k results
   - Check if any retrieved chunk belongs to `query.source_doc_title`
     (look up via `chunk_doc_map[result["index"]]`)
   - If yes (and score >= min_score if set): keep the query
   - If no: discard it
4. Return the filtered list

**Edge cases:**
- Query whose `source_doc_title` isn't in the document list → discard (log a warning
  to stderr)
- Empty query list → return empty list
- Empty document list → return empty list
- Document with no text → skip it during chunking

### `src/query_filters/heuristic.py`

```python
"""Heuristic pre-filters for generated queries.

Fast, zero-cost validation that catches degenerate queries before more
expensive filters (round-trip, cross-encoder) run. Standard practice
across InPars (arxiv:2202.05144), Promptagator (arxiv:2209.11755),
and RAGAS pipelines.

Catches ~10-20% of synthetically generated queries: garbled text,
copy-paste from source, duplicates, non-questions.
"""
```

**Class: `HeuristicFilter`**

Constructor:
```python
def __init__(
    self,
    min_length: int = 5,
    max_length: int = 50,
    require_question_mark: bool = False,
    max_source_overlap: float = 0.8,
    deduplicate: bool = True,
    similarity_threshold: float = 0.9,
) -> None:
```

- `min_length` — minimum query length in words. Below this = degenerate.
- `max_length` — maximum query length in words. Above this = likely copied passage, not a question.
- `require_question_mark` — if True, reject queries not ending with "?".
  Default False because some valid queries are imperative ("Explain..." / "Describe...").
- `max_source_overlap` — maximum fraction of query words (lowercased, stopwords removed)
  that can appear in the source document. Above this = likely copy-paste from source, not
  a genuine question. Use set intersection on word tokens.
- `deduplicate` — if True, remove near-duplicate queries.
- `similarity_threshold` — for deduplication. Two queries are near-duplicates if their
  normalized word sets have Jaccard similarity above this threshold.

Properties:
- `name` → `"heuristic"`

**`filter` method:**

```python
def filter(
    self,
    queries: list[Query],
    documents: list[Document],
) -> list[Query]:
```

Algorithm (apply checks in order, track rejection reasons):
1. Build document lookup: `{doc.title: doc.text for doc in documents}`
2. For each query:
   a. **Length check** — count words in `query.text`. Reject if < min_length or > max_length.
   b. **Question word check** — query must contain at least one of: who, what, where, when,
      why, how, which, is, are, was, were, do, does, did, can, could, would, should, will.
      OR end with "?". Reject if neither.
   c. **Copy detection** — tokenize query and source document (lowercased, strip punctuation).
      Remove English stopwords (use a small hardcoded set: the, a, an, is, are, was, were, in,
      on, at, to, for, of, and, or, but, with, by, from, as, it, this, that). Compute overlap
      ratio = |query_words ∩ doc_words| / |query_words|. Reject if > max_source_overlap.
      Skip if source doc not found in documents.
   d. **Question mark check** — only if `require_question_mark` is True.
3. After individual filtering, **deduplicate** if enabled:
   - For each pair of remaining queries, compute Jaccard similarity on lowercased word sets.
   - If similarity > threshold, keep the first one (by position), discard the duplicate.
   - Use a greedy approach: iterate in order, compare each query against all previously
     kept queries.
4. Return filtered list.

**Edge cases:**
- Empty query list → return empty list
- Empty document list → skip copy detection (can't check overlap without source text),
  apply all other checks
- Query with source_doc_title not in documents → skip copy detection for that query
- All queries filtered out → return empty list (this is informative — the generated
  set was garbage)

### `src/query_generators/__init__.py`

```python
from src.query_generators.ragas import RagasQueryGenerator

__all__ = ["RagasQueryGenerator"]
```

### `src/query_filters/__init__.py`

```python
from src.query_filters.round_trip import RoundTripFilter
from src.query_filters.heuristic import HeuristicFilter

__all__ = ["RoundTripFilter", "HeuristicFilter"]
```

## Files NOT to Touch

- `src/experiment.py` — no changes
- `src/protocols.py` — already updated in task-007
- `src/document.py` — already created in task-007
- `src/query.py` — already created in task-007
- Existing embedder/chunker/strategy/scorer files

## Tests

### `tests/test_ragas_generator.py`

All tests MUST mock RAGAS and LLM calls. Do NOT call real APIs.

1. **test_name_format** — `RagasQueryGenerator().name == "ragas:gpt-4o-mini"`
2. **test_name_custom_model** — `RagasQueryGenerator(generator_model="gpt-4").name == "ragas:gpt-4"`
3. **test_generate_returns_query_objects** — mock RAGAS TestsetGenerator to return a
   fake testset with 3 rows. Call `generate([doc], queries_per_doc=3)`. Assert returns
   3 Query objects with correct fields.
4. **test_generate_maps_evolution_types** — mock RAGAS to return rows with different
   evolution types. Assert query_type mapping: simple→factoid, reasoning→reasoning,
   multi_context→multi_context.
5. **test_generate_sets_generator_name** — all returned queries have
   `generator_name == "ragas:gpt-4o-mini"`.
6. **test_generate_sets_source_doc_title** — returned queries have correct
   `source_doc_title` from the input documents.
7. **test_missing_openai_key** — with no `OPENAI_API_KEY` env var, constructor raises
   `ValueError`.
8. **test_protocol_compliance** — `isinstance(RagasQueryGenerator(...), QueryGenerator)` is True.

**Mocking guidance for RAGAS:**
The RAGAS API may vary by version. Mock at the highest level possible:
- Mock `TestsetGenerator.from_langchain` (or equivalent factory) to return a mock generator
- Mock the generator's `generate` method to return a mock testset
- The mock testset should behave like a pandas DataFrame or have a `to_pandas()` method
  (RAGAS testsets typically convert to DataFrames)
- Set `OPENAI_API_KEY` env var in test fixtures via `monkeypatch`

### `tests/test_round_trip_filter.py`

These tests use real (but simple) mock chunkers and embedders — NOT API-dependent.

1. **test_filter_keeps_good_query** — create a document, chunk it, create a query that
   matches the content. Assert the query passes the filter.
2. **test_filter_removes_bad_query** — create a query with `source_doc_title` pointing to
   doc A, but the query text is completely unrelated (e.g., "What is quantum physics?" for
   a document about cooking). Assert the query is filtered out.
3. **test_filter_empty_queries** — `filter([], docs)` returns `[]`.
4. **test_filter_empty_documents** — `filter(queries, [])` returns `[]`.
5. **test_filter_unknown_source_doc** — query references a doc title not in the document
   list. Assert it's filtered out.
6. **test_name_format** — name includes embedder name and top_k.
7. **test_protocol_compliance** — `isinstance(RoundTripFilter(...), QueryFilter)` is True.

**Mock components for round-trip tests:**

Create simple test helpers (in the test file, not in src/):

```python
class SimpleChunker:
    """Chunks by splitting on double newlines. For testing only."""
    @property
    def name(self) -> str:
        return "test:simple"
    def chunk(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]
```

For the embedder, use `HuggingFaceEmbedder("all-MiniLM-L6-v2")` — it's already installed
and runs locally. This is acceptable for tests that verify retrieval behavior (not API
mocking). If sentence-transformers import is slow, use a mock embedder that returns
deterministic vectors based on text hashing:

```python
class HashEmbedder:
    """Deterministic embedder for testing. NOT for production."""
    @property
    def name(self) -> str:
        return "test:hash"
    @property
    def dimension(self) -> int:
        return 64
    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use hash to seed a deterministic vector
            h = hashlib.md5(text.encode()).digest()
            result[i, :16] = np.frombuffer(h, dtype=np.uint8) / 255.0
        return result
```

The HashEmbedder won't have meaningful similarity, so for test_filter_keeps_good_query,
use the HuggingFaceEmbedder to get real semantic similarity. For the other tests
(empty lists, unknown docs), HashEmbedder is fine.

### `tests/test_heuristic_filter.py`

1. **test_rejects_too_short** — query with 3 words is filtered out.
2. **test_rejects_too_long** — query with 60 words is filtered out.
3. **test_keeps_normal_length** — query with 10 words passes.
4. **test_rejects_no_question_word** — "The mitochondria is the powerhouse" (no question
   word, no "?") is filtered out.
5. **test_keeps_question_word** — "What is the powerhouse of the cell?" passes.
6. **test_keeps_imperative_with_question_word** — "Explain how photosynthesis works" passes
   (contains "how").
7. **test_rejects_copy_paste** — query that is a substring of the source document is
   filtered out (high overlap ratio).
8. **test_keeps_low_overlap** — query with only a few words from the document passes.
9. **test_deduplication** — two near-identical queries ("What is X?" and "What exactly is X?")
   → only one survives.
10. **test_deduplication_preserves_order** — the first of two duplicates is kept, not the second.
11. **test_empty_queries** — `filter([], docs)` returns `[]`.
12. **test_missing_source_doc_skips_copy_check** — query references a doc not in the list;
    copy detection is skipped but other checks still run.
13. **test_protocol_compliance** — `isinstance(HeuristicFilter(), QueryFilter)` is True.
14. **test_name_format** — `HeuristicFilter().name == "heuristic"`.

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Research URLs included where research was done
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
