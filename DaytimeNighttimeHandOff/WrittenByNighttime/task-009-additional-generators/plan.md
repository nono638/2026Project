# Plan: task-009 — Additional Query Generators (Human, BEIR, Template)

## Approach

1. Create branch `night/task-009-additional-generators` from main
2. Merge `night/task-007-query-pipeline-protocols` into it (needed for Document, Query, QueryGenerator protocol)
3. Install spacy and download en_core_web_sm model
4. Create `src/query_generators/human.py` — HumanQuerySet (CSV/JSON loader)
5. Create `src/query_generators/beir.py` — BEIRQuerySet (BEIR format loader)
6. Create `src/query_generators/template.py` — TemplateQueryGenerator (spaCy NER + templates)
7. Update `src/query_generators/__init__.py` — add new imports
8. Create `tests/test_additional_generators.py` — 21 tests per spec
9. Run tests

## Files to Create
- `src/query_generators/human.py`
- `src/query_generators/beir.py`
- `src/query_generators/template.py`
- `tests/test_additional_generators.py`

## Files to Modify
- `src/query_generators/__init__.py` — add HumanQuerySet, BEIRQuerySet, TemplateQueryGenerator
- `requirements.txt` — add spacy pin

## Dependencies
- task-007 branch must be merged in (provides Document, Query, QueryGenerator protocol, query_generators package)
- spacy + en_core_web_sm model needed for TemplateQueryGenerator

## Ambiguities
- None — spec is detailed with exact class signatures, algorithms, templates, and test fixtures
