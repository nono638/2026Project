# task-015: SQuAD 2.0 Dataset Loader

## Summary
Add a SQuAD 2.0 dataset loader that converts the HuggingFace `squad_v2` dataset into
RAGBench's Document + Query format, following the exact same pattern as the HotpotQA
loader. This gives users an "easy baseline" gold-standard dataset alongside HotpotQA's
harder multi-hop questions — enabling calibration across difficulty levels.

## Requirements
1. `load_squad(split="train")` returns `tuple[list[Document], list[Query]]` — parallel
   lists, same length, same order.
2. Each SQuAD example with a non-empty answer becomes one Document + one Query.
3. Unanswerable questions (empty `answers["text"]`) are skipped with a log warning.
4. `sample_squad(documents, queries, n=200, seed=42)` returns a stratified subset,
   sampling proportionally by Wikipedia article title to ensure topic diversity.
5. Both functions are exported from `src/datasets/__init__.py`.
6. All HuggingFace API calls are mocked in tests — no network dependency.

## Files to Modify
- `src/datasets/squad.py` — **create**. The loader module. Contains `_build_document()`,
  `_build_query()`, `load_squad()`, `sample_squad()`. Follow `hotpotqa.py` structure exactly.
- `src/datasets/__init__.py` — **modify**. Add `from src.datasets.squad import load_squad, sample_squad`
  to imports and `__all__`.

## Files to Read for Context
- `src/datasets/hotpotqa.py` — the pattern to follow
- `src/document.py` — Document dataclass
- `src/query.py` — Query dataclass

## New Dependencies
None — `datasets` (HuggingFace) is already installed.

## Field Mapping

SQuAD 2.0 fields → RAGBench fields:

| SQuAD field | RAGBench field | Notes |
|---|---|---|
| `id` | `Document.title` = `"squad:{id}"` | Unique per question |
| `context` | `Document.text` | Single paragraph (no concatenation needed) |
| `title` | `Document.metadata["article_title"]` | Wikipedia article name |
| `question` | `Query.text` | |
| `answers["text"][0]` | `Query.reference_answer` | First answer span |
| (literal) | `Query.query_type` = `"factoid"` | All SQuAD questions are extractive factoid |
| (literal) | `Query.generator_name` = `"squad"` | |
| `id` | `Query.source_doc_title` = `"squad:{id}"` | Links back to Document |
| `title` | `Query.metadata["article_title"]` | Wikipedia article name |
| `len(answers["text"])` | `Query.metadata["num_answer_spans"]` | How many valid spans |

## Edge Cases
- **Unanswerable question** (`len(answers["text"]) == 0`): skip, log warning. Do not
  create a Document or Query.
- **Empty context string**: skip, log warning. (Shouldn't happen in practice but be safe.)
- **Multiple answer spans** (e.g., `answers["text"]` has 3 entries): use the first one
  as `reference_answer`. Store the count in metadata.
- **`sample_squad(n=0)`**: return empty lists.
- **`sample_squad(n > len(documents))`**: return all documents (copy, don't mutate).
- **Single-article dataset** (degenerate case): sampling should still work — just returns
  random subset from that one stratum.

## Decisions Made
- **Skip unanswerable questions**: no gold answer to evaluate against. Consistent with
  HotpotQA skipping empty answers. **Why:** RAG evaluation needs a reference to compare.
- **Use first answer span as reference_answer**: multiple spans are the same answer found
  at different character positions. **Why:** simpler, and the text content is identical.
- **query_type = "factoid" for all**: SQuAD questions are all extractive factoid Q&A.
  **Why:** unlike HotpotQA's bridge/comparison distinction, SQuAD has no type labels.
- **Stratify sampling by article title**: SQuAD has 442 unique Wikipedia articles. Stratifying
  by title ensures the sample isn't dominated by one topic. **Why:** HotpotQA stratifies
  by (type, difficulty) but SQuAD lacks those fields — article title is the natural axis.
- **1:1 Document:Query mapping**: each example gets its own Document even though multiple
  questions share the same context paragraph. **Why:** maintains the parallel-lists contract
  that the rest of the pipeline expects.
- **Document text = raw context paragraph**: no markdown headers or separators (unlike
  HotpotQA's multi-passage concatenation). **Why:** SQuAD contexts are single paragraphs.

## What NOT to Touch
- `src/datasets/hotpotqa.py` — leave unchanged
- `src/document.py` — no changes needed
- `src/query.py` — no changes needed
- Any experiment runner or scorer code

## Testing Approach
- Tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-015-squad-loader/tests/test_squad_loader.py`
- Mock `hf_load_dataset` to return fake SQuAD data matching the real schema
- Test classes mirror the HotpotQA test structure:
  - `TestLoadSquad` — returns correct types, parallel lists, reference answers present
  - `TestDocumentFormat` — text matches context, title format correct
  - `TestSampleSquad` — correct count, deterministic, handles n > available
  - `TestEdgeCases` — unanswerable skipped, empty context skipped
  - `TestCompatibility` — output works with `documents_to_dicts()`
- Run with: `pytest DaytimeNighttimeHandOff/WrittenByDaytime/task-015-squad-loader/tests/`
