# Spec: task-009 — Additional Query Generators: Human, BEIR, Template

## What

Add three more `QueryGenerator` implementations to complement RAGAS (task-008):

1. **`HumanQuerySet`** — loads hand-curated queries from a CSV/JSON file
2. **`BEIRQuerySet`** — loads queries from BEIR benchmark datasets
3. **`TemplateQueryGenerator`** — extracts entities/facts from documents and slots them
   into query templates per type

These give methodological breadth: synthetic (RAGAS), human (HumanQuerySet), benchmark
(BEIR), and structured (Template). Running the same experiment across generators tests
whether RAG rankings are robust to query source — itself a research finding.

### New files to create:

1. `src/query_generators/human.py` — HumanQuerySet
2. `src/query_generators/beir.py` — BEIRQuerySet
3. `src/query_generators/template.py` — TemplateQueryGenerator
4. `src/query_generators/__init__.py` — update with new imports
5. `tests/test_additional_generators.py` — tests for all three

## Why

Single-source evaluation is a methodological weakness. Each generator has different
strengths:
- **HumanQuerySet**: validation anchor — if synthetic and human queries rank RAG configs
  the same way, you can trust the synthetic set. Standard practice per ARES (arxiv:2311.09476).
- **BEIRQuerySet**: gold-standard human-written queries from established benchmarks, gives
  comparability with published results.
- **TemplateQueryGenerator**: maximum uniformity — identical query structures across documents,
  isolating the retrieval variable. Cheap, no LLM cost.

See architecture-decisions.md: "Multiple query generators by design, not one framework."

## File Details

### `src/query_generators/human.py`

```python
"""Human-curated query set loader.

Wraps a CSV or JSON file of hand-written queries in the QueryGenerator
interface. The 'generate' method is a misnomer here — it loads, not generates —
but conforming to the protocol lets human queries plug into the same pipeline
as synthetic generators.

Used as a validation anchor: run the same experiment with synthetic and human
queries, check if they rank RAG configurations the same way.
"""
```

**Class: `HumanQuerySet`**

Constructor:
```python
def __init__(self, path: str | Path) -> None:
```

- `path` — path to a CSV or JSON file containing queries
- CSV format: columns `text`, `query_type`, `source_doc_title`,
  and optionally `reference_answer`
- JSON format: array of objects with the same fields (same as `save_queries` output
  from task-007)
- Detect format by file extension (`.csv` → CSV, `.json` → JSON)
- Load and validate on construction. Raise `ValueError` if required fields are missing.
- Store as `list[Query]` internally.

Properties:
- `name` → `f"human:{Path(path).stem}"` (e.g., `"human:validation_queries"`)

**`generate` method:**

```python
def generate(
    self,
    documents: list[Document],
    queries_per_doc: int = 5,
) -> list[Query]:
```

- Ignores `queries_per_doc` (the set is fixed).
- If `documents` is provided and non-empty: filter to only return queries whose
  `source_doc_title` matches a title in the documents list. This lets you use a
  subset of the corpus without getting queries for missing documents.
- If `documents` is empty: return all queries.
- Set `generator_name` to `self.name` on all returned queries.

### `src/query_generators/beir.py`

```python
"""BEIR benchmark query set loader.

Loads queries from BEIR-format datasets (https://github.com/beir-cellar/beir).
BEIR datasets use a standard directory structure:
  dataset_name/
    corpus.jsonl    — documents (id, title, text)
    queries.jsonl   — queries (id, text)
    qrels/
      test.tsv      — relevance judgments (query_id, corpus_id, score)

This loader reads the queries and maps them to corpus documents via qrels.
The corpus itself can be loaded separately via Document loaders.

Ref: BEIR benchmark (arxiv:2104.08663)
"""
```

**Class: `BEIRQuerySet`**

Constructor:
```python
def __init__(
    self,
    dataset_dir: str | Path,
    split: str = "test",
    query_type: str = "factoid",
) -> None:
```

- `dataset_dir` — path to the BEIR dataset directory
- `split` — which qrels split to use (default "test")
- `query_type` — BEIR doesn't have query type metadata, so assign a default. User can
  override. Common choice: "factoid" for most BEIR datasets, "multi_context" for
  HotpotQA-style datasets.
- On construction:
  - Read `queries.jsonl` — each line is `{"_id": "...", "text": "...", "metadata": {}}`
  - Read `qrels/{split}.tsv` — tab-separated: query_id, corpus_id, score
  - Read `corpus.jsonl` — each line is `{"_id": "...", "title": "...", "text": "..."}`
  - Map each query to its highest-relevance corpus document via qrels
  - Build `list[Query]` with `source_doc_title` from corpus lookup
- Raise `FileNotFoundError` if required files are missing.

Properties:
- `name` → `f"beir:{dataset_dir.name}"` (e.g., `"beir:nfcorpus"`)

**`generate` method:**

```python
def generate(
    self,
    documents: list[Document],
    queries_per_doc: int = 5,
) -> list[Query]:
```

- Ignores `queries_per_doc` (like HumanQuerySet, the set is fixed).
- If `documents` is non-empty: filter to queries whose `source_doc_title` matches.
- Set `generator_name` to `self.name` on all returned queries.

**`load_corpus` convenience method (NOT part of the protocol):**

```python
def load_corpus(self) -> list[Document]:
    """Load the BEIR corpus as Document objects.

    Convenience method so users can load both queries and documents from the
    same BEIR dataset directory.
    """
```

- Read `corpus.jsonl`, return as `list[Document]`
- Include `{"beir_id": id}` in `metadata`

### `src/query_generators/template.py`

```python
"""Template-based query generator using entity/fact extraction.

Extracts named entities and factual statements from documents using spaCy,
then slots them into query templates per type. Produces highly uniform queries
at zero LLM cost — useful for controlled experiments where query structure
should be held constant.

Weakness: limited to patterns the templates cover. Cannot generate genuinely
creative or nuanced questions. Best used alongside RAGAS or human queries
for comparison.
"""
```

**Class: `TemplateQueryGenerator`**

Constructor:
```python
def __init__(
    self,
    spacy_model: str = "en_core_web_sm",
    templates: dict[str, list[str]] | None = None,
) -> None:
```

- `spacy_model` — spaCy model for NER and sentence parsing. `en_core_web_sm` is small
  and fast. Install it if not present (see dependencies section).
- `templates` — query templates per type. If None, use defaults:

```python
DEFAULT_TEMPLATES = {
    "factoid": [
        "What is {entity}?",
        "When was {entity} established?",
        "Where is {entity} located?",
        "Who is {entity}?",
    ],
    "reasoning": [
        "Why is {entity} significant in the context of {topic}?",
        "How does {entity} relate to {entity2}?",
        "What is the significance of {entity} according to the document?",
    ],
    "multi_context": [
        "What do {entity} and {entity2} have in common?",
        "Compare {entity} and {entity2} based on the information provided.",
    ],
    "conditional": [
        "What would happen if {entity} were not involved?",
        "Under what conditions is {entity} relevant to {topic}?",
    ],
}
```

Properties:
- `name` → `"template:spacy"` (or `f"template:{self._spacy_model}"`)

**`generate` method:**

```python
def generate(
    self,
    documents: list[Document],
    queries_per_doc: int = 5,
) -> list[Query]:
```

Algorithm:
1. For each document:
   a. Run spaCy NER on `doc.text` to extract entities (PERSON, ORG, GPE, EVENT, etc.)
   b. Extract the document's topic from the title (use `doc.title` as `{topic}`)
   c. Deduplicate entities, sort by frequency (most common first)
   d. For each query type, select a template and fill slots:
      - `{entity}` — most prominent entity
      - `{entity2}` — second most prominent entity (for multi_context/reasoning)
      - `{topic}` — `doc.title`
   e. Distribute `queries_per_doc` across types proportionally:
      - factoid: 40%, reasoning: 30%, multi_context: 20%, conditional: 10%
      - Round to integers, ensure total = `queries_per_doc`
   f. If document has fewer than 2 entities, skip multi_context and conditional
      templates, fill remaining quota with factoid
2. Create `Query` objects with `generator_name = self.name`
3. Return all queries

**Edge cases:**
- Document with no extractable entities → generate 0 queries for that doc (log warning
  to stderr), continue to next document
- Very short document (< 50 words) → may have few entities, handle gracefully
- Template slot `{entity2}` needed but only 1 entity found → skip that template,
  use a factoid template instead

### `src/query_generators/__init__.py`

Update to include all generators:

```python
from src.query_generators.ragas import RagasQueryGenerator
from src.query_generators.human import HumanQuerySet
from src.query_generators.beir import BEIRQuerySet
from src.query_generators.template import TemplateQueryGenerator

__all__ = [
    "RagasQueryGenerator",
    "HumanQuerySet",
    "BEIRQuerySet",
    "TemplateQueryGenerator",
]
```

## Dependencies to Install

1. `spacy` — for TemplateQueryGenerator's NER
2. `en_core_web_sm` spaCy model — install via `python -m spacy download en_core_web_sm`

**Use the install-package skill** for spacy. The model download is a separate command
(not a pip install) — run it after installing spacy.

No new dependencies for HumanQuerySet or BEIRQuerySet (they use pandas and json, both
already available).

## Files NOT to Touch

- `src/protocols.py` — already updated in task-007
- `src/document.py` — already created in task-007
- `src/query.py` — already created in task-007
- `src/experiment.py` — no changes
- `src/query_generators/ragas.py` — already created in task-008
- `src/query_filters/round_trip.py` — already created in task-008

## Tests — `tests/test_additional_generators.py`

### HumanQuerySet tests:

1. **test_load_from_csv** — create temp CSV with text, query_type, source_doc_title
   columns. Load with HumanQuerySet. Assert correct count and field values.
2. **test_load_from_json** — create temp JSON (same format as `save_queries` output).
   Load. Assert correct.
3. **test_name_format** — name is `"human:<filename_stem>"`.
4. **test_generate_filters_by_documents** — load a set with queries for docs A, B, C.
   Call `generate(documents=[doc_A])`. Assert only doc A's queries returned.
5. **test_generate_empty_documents_returns_all** — `generate([])` returns all queries.
6. **test_missing_required_fields** — CSV missing `text` column raises `ValueError`.
7. **test_protocol_compliance** — `isinstance(HumanQuerySet(...), QueryGenerator)` is True.

### BEIRQuerySet tests:

1. **test_load_beir_dataset** — create temp BEIR directory structure with corpus.jsonl,
   queries.jsonl, qrels/test.tsv (small: 3 docs, 5 queries). Load. Assert correct
   query count and source_doc_title mapping.
2. **test_name_format** — name is `"beir:<dir_name>"`.
3. **test_generate_filters_by_documents** — same pattern as HumanQuerySet.
4. **test_load_corpus** — `load_corpus()` returns Document objects with correct titles
   and beir_id metadata.
5. **test_missing_files** — directory without queries.jsonl raises `FileNotFoundError`.
6. **test_protocol_compliance** — `isinstance(BEIRQuerySet(...), QueryGenerator)` is True.

**BEIR test fixture:**

Create a minimal BEIR dataset in the test using `tmp_path`:

```python
def create_beir_fixture(tmp_path):
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    # corpus.jsonl
    with open(dataset_dir / "corpus.jsonl", "w") as f:
        f.write('{"_id": "d1", "title": "Paris", "text": "Paris is the capital of France."}\n')
        f.write('{"_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany."}\n')
    # queries.jsonl
    with open(dataset_dir / "queries.jsonl", "w") as f:
        f.write('{"_id": "q1", "text": "What is the capital of France?"}\n')
        f.write('{"_id": "q2", "text": "What is the capital of Germany?"}\n')
    # qrels/test.tsv
    (dataset_dir / "qrels").mkdir()
    with open(dataset_dir / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\td1\t1\n")
        f.write("q2\td2\t1\n")
    return dataset_dir
```

### TemplateQueryGenerator tests:

1. **test_generate_returns_queries** — pass a document with known entities (e.g., "Albert
   Einstein was born in Ulm, Germany. He developed the theory of relativity."). Assert
   queries are generated with correct types.
2. **test_query_types_distributed** — with `queries_per_doc=10`, assert approximate
   distribution: ~4 factoid, ~3 reasoning, ~2 multi_context, ~1 conditional.
3. **test_entity_extraction** — document with "New York" and "United Nations". Assert
   both appear in generated query text.
4. **test_document_no_entities** — document with no recognizable entities (e.g., "The
   quick brown fox jumps over the lazy dog."). Assert 0 queries generated, no crash.
5. **test_single_entity_skips_multi_context** — document with only one entity. Assert
   no multi_context queries (they need {entity2}).
6. **test_name_format** — name is `"template:en_core_web_sm"`.
7. **test_custom_templates** — pass custom templates dict. Assert they're used instead
   of defaults.
8. **test_protocol_compliance** — `isinstance(TemplateQueryGenerator(...), QueryGenerator)` is True.

## Quality Checklist
- [x] Exact files to modify are listed
- [x] All edge cases are explicit
- [x] All judgment calls are made
- [x] Why is answered for every non-obvious decision
- [x] Research URLs included where research was done
- [x] Tests cover key behaviors, not just "does it run"
- [x] Scoped to one focused session
