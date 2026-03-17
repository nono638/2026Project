# task-013: HotpotQA Dataset Loader

## Summary

Build a loader for the HotpotQA dataset (HuggingFace `datasets` library) that converts
it into RAGBench's `Document` + `Query` format. This is RAGBench's first built-in
gold-standard dataset, enabling experiments with reference answers and scorer validation.

HotpotQA has 113K multi-hop Wikipedia Q&A pairs with gold answers, difficulty labels
(easy/medium/hard), and question types (bridge/comparison). Each example comes with
10 Wikipedia passages (2 supporting + 8 distractors).

## Requirements

1. A `load_hotpotqa()` function that downloads and converts HotpotQA into `Document` + `Query` lists.
2. Each HotpotQA example produces ONE `Document` (all 10 passages concatenated with
   clear `\n\n---\n\n` separators and passage titles as headers) and ONE `Query`
   (with `reference_answer` set to the gold answer).
3. A `sample_hotpotqa()` function that returns a stratified subset: balanced by
   `type` (bridge/comparison) and `level` (easy/medium/hard).
4. Query metadata includes: `difficulty` (easy/medium/hard), `question_type` (bridge/comparison),
   `supporting_titles` (list of passage titles that contain the answer).
5. Document metadata includes: `passage_titles` (list of all 10 passage titles),
   `num_passages` (always 10 for distractor config).
6. Returns data in the format `Experiment.load_corpus()` expects: the `documents_to_dicts()`
   and `queries_to_dicts()` bridge helpers work on the output.
7. The function caches the download — HuggingFace `datasets` handles this automatically,
   but the function should not re-download on every call.

## Files to Create

- `src/datasets/__init__.py` — Package init, exports `load_hotpotqa`, `sample_hotpotqa`
- `src/datasets/hotpotqa.py` — The loader implementation

## Files to Read (for context, do NOT modify)

- `src/document.py` — `Document` dataclass, understand the fields
- `src/query.py` — `Query` dataclass, understand the fields (especially `reference_answer`, `metadata`)
- `src/experiment.py` — `Experiment.load_corpus()` to understand expected input format

## New Dependencies

None — `datasets` 4.7.0 is already installed.

## Edge Cases

- **Null or empty passages in context:** Skip any passage with empty text. Still create the
  Document from remaining passages.
- **Very long concatenated documents:** Some examples may produce documents >20K chars.
  This is fine — the chunker handles splitting. Do not truncate.
- **Duplicate passage titles across examples:** This is expected (same Wikipedia article
  appears in multiple examples). Each example gets its own Document — no deduplication
  across examples.
- **Empty answer field:** Skip the example entirely. Log a warning with the example ID.
- **sample_hotpotqa with n > available:** Return all available, same as `sample_corpus()`.

## Decisions Made

- **One Document per HotpotQA example, not per passage.** Why: The existing Experiment
  runner loops over documents and builds a per-document retriever. By concatenating all
  10 passages into one document, the chunker splits them into retrieval units and the
  strategy must find the relevant chunks among distractors. This matches how real RAG
  works (retrieve from a corpus, not from pre-selected passages) and works with the
  existing Experiment interface unchanged.

- **Use the "distractor" config, not "fullwiki".** Why: The distractor config provides
  10 curated passages per question (2 gold + 8 distractors). This is the standard
  evaluation setup for HotpotQA. The fullwiki config requires searching all of Wikipedia,
  which is out of scope.

- **Passage concatenation format:** Each passage gets a title header and separator:
  ```
  ## Radio City (Indian radio station)
  Radio City is India's first private FM radio station...

  ---

  ## Arthur's Magazine
  Arthur's Magazine (1844–1846) was an American literary periodical...
  ```
  Why: Clear separators help the chunker respect passage boundaries. Markdown headers
  give the retriever signal about passage identity.

- **Package name `src/datasets/`:** Why: follows the existing pattern (`src/chunkers/`,
  `src/embedders/`, `src/strategies/`). Note: this shadows the HuggingFace `datasets`
  package name at the `src` level, but since we import as `from src.datasets.hotpotqa import ...`
  and HuggingFace as `from datasets import load_dataset`, there's no conflict.

- **Stratified sampling:** `sample_hotpotqa(n, seed)` samples proportionally from each
  (type, level) combination. 6 strata: (bridge × easy/medium/hard) + (comparison × easy/medium/hard).
  Why: Ensures the sample isn't dominated by one difficulty level. Uses the same
  `random.Random(seed)` pattern as `sample_corpus()` for reproducibility.

## What NOT to Touch

- `src/experiment.py` — The experiment runner interface stays unchanged. HotpotQA data
  flows through the existing `load_corpus()` path.
- `src/document.py` / `src/query.py` — No changes to the dataclasses.
- Any existing test files.

## Testing Approach

Pre-written tests mock the HuggingFace `datasets.load_dataset` call to avoid network
dependency. Tests cover:
- Document creation from a sample HotpotQA example
- Query creation with reference_answer and metadata
- Passage concatenation format
- Stratified sampling distribution
- Edge cases (empty passages, empty answers)
- Output compatibility with `documents_to_dicts()` and `queries_to_dicts()`
