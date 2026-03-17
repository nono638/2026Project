# Plan: task-013 — HotpotQA Dataset Loader

## Approach

1. Create `src/datasets/__init__.py` exporting `load_hotpotqa` and `sample_hotpotqa`
2. Create `src/datasets/hotpotqa.py` with:
   - `load_hotpotqa(split="train")` — downloads HotpotQA distractor config via HuggingFace `datasets`, converts each example to a `Document` + `Query` pair
   - `sample_hotpotqa(docs, queries, n, seed)` — stratified sampling by (type, level)
3. Passage concatenation: `## Title\nSentence text...\n\n---\n\n` per passage
4. Skip empty-answer examples with a warning log
5. Skip passages with empty sentences lists

## Files to Create
- `src/datasets/__init__.py`
- `src/datasets/hotpotqa.py`

## Files Read for Context
- `src/document.py` — Document dataclass
- `src/query.py` — Query dataclass (reference_answer, metadata)
- `src/experiment.py` — load_corpus expects dicts with title/text and text/type

## Ambiguities
- None — spec is clear. The test mocks `hf_load_dataset` which means I should import
  `datasets.load_dataset` aliased as `hf_load_dataset` in `hotpotqa.py`.
