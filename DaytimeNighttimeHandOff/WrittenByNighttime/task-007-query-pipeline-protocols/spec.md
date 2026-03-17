# Spec: task-007 — Query Pipeline Protocols and Data Models

## What

Add the data models and protocol definitions for the query generation and filtering
stages of the experiment pipeline. This task creates the interfaces only — no
implementations (those come in task-008).

### New files to create:

1. `src/document.py` — Document dataclass
2. `src/query.py` — Query dataclass
3. `src/protocols.py` — add QueryGenerator and QueryFilter protocols
4. `src/query_generators/__init__.py` — empty package
5. `src/query_filters/__init__.py` — empty package
6. `tests/test_query_pipeline.py` — tests for data models and protocol compliance

## Why

The experiment pipeline currently takes raw `list[dict]` for documents and queries.
As the system expands to support pluggable query generation (RAGAS, templates, human-curated
sets) and filtering (round-trip consistency, LLM-based), each stage needs a clean protocol.

**Design principle:** every pipeline stage gets its own protocol in its own module. We favor
more separation over less because multimodal support is an end goal — each stage may
eventually handle different input types. See architecture-decisions.md.

## File Details

### `src/document.py`

```python
"""Document representation for the RAG research tool.

Separate module (not in protocols.py) because Document will grow to support
multimodal content (images, video) in the future. Keeping it isolated means
multimodal changes don't ripple through protocol definitions.
"""
```

**`Document` dataclass:**

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Document:
    """A document in the evaluation corpus.

    Attributes:
        title: Human-readable document identifier.
        text: Full document text content.
        metadata: Optional extensible metadata (e.g., source, word_count, domain).
    """
    title: str
    text: str
    metadata: dict | None = field(default=None)
```

**`load_corpus_from_csv` function:**

A convenience loader for CSV files like the Wikipedia dataset. Not a method on Document —
it's a module-level function.

```python
def load_corpus_from_csv(
    path: str | Path,
    title_col: str = "title",
    text_col: str = "text",
    metadata_cols: list[str] | None = None,
) -> list[Document]:
```

- Read CSV with pandas
- Skip rows where `text_col` is NaN/null (the Wikipedia dataset has 36 null texts)
- For each row, create a `Document` with title, text, and metadata dict from
  `metadata_cols` (e.g., `["word_count"]`)
- Return list of Documents

**`sample_corpus` function:**

For experiments that use a subset of documents. Stratified sampling by document length.

```python
def sample_corpus(
    documents: list[Document],
    n: int = 200,
    seed: int = 42,
    stratify_by: str = "length",
) -> list[Document]:
```

- If `stratify_by == "length"`: bin documents into quartiles by `len(doc.text)`,
  sample proportionally from each bin. Use `random.Random(seed)` for reproducibility.
- If `n >= len(documents)`: return all documents (no sampling needed).
- Return the sampled list.

### `src/query.py`

```python
"""Query representation for the RAG evaluation pipeline.

Separate module because queries carry metadata about their generation,
type classification, and source provenance. Will grow to support
multimodal queries (image-based questions) in the future.
"""
```

**`Query` dataclass:**

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Query:
    """An evaluation query generated from or associated with a document.

    Attributes:
        text: The question text.
        query_type: Category — one of: factoid, reasoning, multi_context, conditional.
        source_doc_title: Title of the document this query was generated from.
        reference_answer: Optional gold-standard answer for evaluation.
        generator_name: Name of the QueryGenerator that created this query.
        metadata: Optional extensible metadata.
    """
    text: str
    query_type: str
    source_doc_title: str
    reference_answer: str | None = None
    generator_name: str | None = None
    metadata: dict | None = field(default=None)
```

**`save_queries` and `load_queries` functions:**

Queries are generated once and frozen. These persist them as JSON.

```python
def save_queries(queries: list[Query], path: str | Path) -> None:
    """Save queries to a JSON file for reproducibility."""
```

- Serialize each Query to a dict via `dataclasses.asdict`
- Write as a JSON array with indent=2

```python
def load_queries(path: str | Path) -> list[Query]:
    """Load queries from a JSON file."""
```

- Read JSON, construct Query objects from each dict
- Validate that required fields (text, query_type, source_doc_title) are present;
  raise `ValueError` with a clear message if not

**Bridge helpers for the existing Experiment runner:**

The current `Experiment.load_corpus()` takes `list[dict]`. Add converter functions
so the new types work with the existing runner without modifying experiment.py:

```python
def queries_to_dicts(queries: list[Query]) -> list[dict]:
    """Convert Query objects to the dict format Experiment.load_corpus expects.

    Returns list of dicts with 'text' and 'type' keys.
    """
    return [{"text": q.text, "type": q.query_type} for q in queries]
```

Put this in `src/query.py`.

Also add to `src/document.py`:

```python
def documents_to_dicts(documents: list[Document]) -> list[dict]:
    """Convert Document objects to the dict format Experiment.load_corpus expects.

    Returns list of dicts with 'title' and 'text' keys.
    """
    return [{"title": d.title, "text": d.text} for d in documents]
```

### `src/protocols.py` — additions

Add two new protocols to the existing file. Do NOT modify existing protocols.

**`QueryGenerator` protocol:**

```python
@runtime_checkable
class QueryGenerator(Protocol):
    """Interface for evaluation query generation backends."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'ragas:gpt-4o-mini', 'template:factoid')."""
        ...

    def generate(
        self,
        documents: list,
        queries_per_doc: int = 5,
    ) -> list:
        """Generate evaluation queries from documents.

        Args:
            documents: List of Document objects from src.document.
            queries_per_doc: Target number of queries to generate per document.

        Returns:
            List of Query objects from src.query.
        """
        ...
```

Note: the type hints use `list` (not `list[Document]` / `list[Query]`) to avoid
importing Document and Query into protocols.py, which would create coupling. The
docstrings specify the expected types. This matches the existing pattern where
`Strategy.run` takes `Retriever` via TYPE_CHECKING guard. Add Document and Query
to the TYPE_CHECKING block for type-checker support:

```python
if TYPE_CHECKING:
    from src.document import Document
    from src.query import Query
    from src.retriever import Retriever
```

Then use the string annotations (which work because of `from __future__ import annotations`):

```python
    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
```

**`QueryFilter` protocol:**

```python
@runtime_checkable
class QueryFilter(Protocol):
    """Interface for query validation/filtering backends."""

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'round_trip:k=5', 'llm_judge')."""
        ...

    def filter(
        self,
        queries: list[Query],
        documents: list[Document],
    ) -> list[Query]:
        """Filter queries, returning only those that pass validation.

        Args:
            queries: List of Query objects to validate.
            documents: The source documents (needed for context-dependent filters
                       like round-trip retrieval).

        Returns:
            Filtered list of Query objects.
        """
        ...
```

### `src/query_generators/__init__.py`

Empty for now. Will hold imports as implementations are added.

```python
"""Query generator implementations.

Each module provides a class implementing the QueryGenerator protocol.
"""
```

### `src/query_filters/__init__.py`

Empty for now.

```python
"""Query filter implementations.

Each module provides a class implementing the QueryFilter protocol.
"""
```

## Files NOT to Touch

- `src/experiment.py` — do not modify. The bridge helpers handle conversion.
- `src/retriever.py` — no changes.
- Existing embedder/chunker/strategy/scorer files — no changes.

## Tests — `tests/test_query_pipeline.py`

### Document tests:

1. **test_document_creation** — create a Document with title, text, metadata. Assert
   all fields accessible.
2. **test_document_metadata_default_none** — Document without metadata kwarg has
   `metadata == None`.
3. **test_load_corpus_from_csv** — create a small temp CSV with title, text, word_count
   columns (use `tmp_path` fixture). Load with `load_corpus_from_csv`. Assert correct
   count, titles, text content.
4. **test_load_corpus_skips_null_text** — CSV with one row having empty/NaN text. Assert
   that row is skipped.
5. **test_load_corpus_metadata_cols** — load with `metadata_cols=["word_count"]`. Assert
   `doc.metadata == {"word_count": <value>}`.
6. **test_sample_corpus_returns_n** — 50 documents in, `sample_corpus(docs, n=10)` returns 10.
7. **test_sample_corpus_deterministic** — same seed produces same sample.
8. **test_sample_corpus_n_exceeds_total** — `sample_corpus(docs, n=100)` with 50 docs
   returns all 50.
9. **test_documents_to_dicts** — converts correctly to `[{"title": ..., "text": ...}]`.

### Query tests:

1. **test_query_creation** — create a Query with all fields. Assert accessible.
2. **test_query_defaults** — Query with only required fields has `reference_answer=None`,
   `generator_name=None`, `metadata=None`.
3. **test_save_and_load_queries** — create queries, save to JSON, load back. Assert
   round-trip equality.
4. **test_load_queries_validates_fields** — JSON missing `text` field raises `ValueError`.
5. **test_queries_to_dicts** — converts correctly to `[{"text": ..., "type": ...}]`.

### Protocol compliance tests:

6. **test_query_generator_protocol** — create a minimal class with `name` property and
   `generate` method. Assert `isinstance(obj, QueryGenerator)` is True.
7. **test_query_filter_protocol** — create a minimal class with `name` property and
   `filter` method. Assert `isinstance(obj, QueryFilter)` is True.

## Edge Cases

- CSV with no rows → `load_corpus_from_csv` returns empty list
- CSV with all null text → returns empty list
- `sample_corpus` with `n=0` → returns empty list
- `save_queries` with empty list → writes `[]` to JSON
- `load_queries` with empty JSON array → returns empty list

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
