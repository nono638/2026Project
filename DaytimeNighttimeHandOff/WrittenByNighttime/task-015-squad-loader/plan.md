# Plan: task-015 — SQuAD 2.0 Dataset Loader

## Files to Create
- `src/datasets/squad.py` — loader module following hotpotqa.py pattern

## Files to Modify
- `src/datasets/__init__.py` — add load_squad, sample_squad exports

## Approach
1. Create `src/datasets/squad.py` mirroring the structure of `hotpotqa.py`
2. Implement `_build_document()` — maps SQuAD context to Document (single paragraph, no concatenation)
3. Implement `_build_query()` — maps SQuAD question to Query with factoid type
4. Implement `load_squad(split)` — loads HF squad_v2, skips unanswerable/empty, returns parallel lists
5. Implement `sample_squad()` — stratified by article_title (not type/difficulty like HotpotQA)
6. Update `__init__.py` exports
7. Run tests

## Ambiguities
- None — spec is clear and follows established pattern exactly.
