# Task 021: Consolidate Nighttime Tests into Main Test Suite

## Goal

Move all test files from `DaytimeNighttimeHandOff/WrittenByNighttime/*/tests/` into
the main `tests/` directory so they run as part of the standard `pytest` invocation.
Rename duplicate class names to avoid pytest collection conflicts.

## Why

Nighttime tests for tasks 012-020 were written into `WrittenByNighttime/` but never
copied to `tests/`. They currently run during pytest discovery (since there's no
`testpaths` config), but duplicate class names across files cause silent collisions.
After this task, all tests live in one place and run cleanly.

## Files to Create/Modify

### Copy these files into `tests/` (new files):

| Source | Destination |
|---|---|
| `WrittenByNighttime/task-012-claude-scorer/tests/test_claude_scorer.py` | `tests/test_claude_scorer.py` |
| `WrittenByNighttime/task-013-hotpotqa-loader/tests/test_hotpotqa_loader.py` | `tests/test_hotpotqa_loader.py` |
| `WrittenByNighttime/task-014-experiment-timing/tests/test_experiment_timing.py` | `tests/test_experiment_timing.py` |
| `WrittenByNighttime/task-015-squad-loader/tests/test_squad_loader.py` | `tests/test_squad_loader.py` |
| `WrittenByNighttime/task-016-embedder-sdk-migration/tests/test_embedder_migration.py` | `tests/test_embedder_migration.py` |
| `WrittenByNighttime/task-017-llm-scorer/tests/test_llm_scorer.py` | `tests/test_llm_scorer.py` |
| `WrittenByNighttime/task-018-experiment-zero/tests/test_experiment_zero.py` | `tests/test_experiment_zero.py` |
| `WrittenByNighttime/task-019-hybrid-retrieval/tests/test_hybrid_retrieval.py` | `tests/test_hybrid_retrieval.py` |
| `WrittenByNighttime/task-020-llm-protocol/tests/test_llm_protocol.py` | `tests/test_llm_protocol.py` |

### Modify: rename duplicate test class names

After copying, rename these classes to avoid pytest collisions. Use the pattern
`Test<Feature><OriginalName>` to make them unique and descriptive:

**`tests/test_claude_scorer.py`:**
- `TestProtocolCompliance` → `TestClaudeScorerProtocolCompliance`
- `TestEdgeCases` → `TestClaudeScorerEdgeCases`

**`tests/test_hotpotqa_loader.py`:**
- `TestDocumentFormat` → `TestHotpotqaDocumentFormat`
- `TestEdgeCases` → `TestHotpotqaEdgeCases`
- `TestCompatibility` → `TestHotpotqaCompatibility`

**`tests/test_squad_loader.py`:**
- `TestDocumentFormat` → `TestSquadDocumentFormat`
- `TestEdgeCases` → `TestSquadEdgeCases`
- `TestCompatibility` → `TestSquadCompatibility`

**`tests/test_hybrid_retrieval.py`:**
- `TestEdgeCases` → `TestHybridRetrievalEdgeCases`

### Modify: `pytest.ini` or `pyproject.toml` (create if neither exists)

Add pytest config to restrict test discovery to `tests/` only:

```ini
[pytest]
testpaths = tests
```

This prevents pytest from discovering the original copies in `WrittenByNighttime/`.

### Note on task-012 (ClaudeScorer)

`test_claude_scorer.py` tests `ClaudeScorer` which was renamed to `LLMScorer` in
task-017. Check if the tests still import `ClaudeScorer` — if so, update imports to
use `LLMScorer` from `src.scorers.llm`. If the tests are fully redundant with
`test_llm_scorer.py`, skip copying `test_claude_scorer.py` entirely.

### Note on task-016 (Embedder Migration)

`test_embedder_migration.py` tests the google-genai SDK migration. Check if these
tests overlap significantly with `tests/test_google_embedders.py` (which already
exists in the main suite). If so, merge the unique tests into `test_google_embedders.py`
rather than creating a separate file.

## What NOT to touch

- Do NOT delete the original files in `WrittenByNighttime/` — they're historical records
- Do NOT modify any source code in `src/`
- Do NOT add new test logic — only copy, rename classes, and fix imports if needed

## Verification

1. Run `pytest tests/ -v` — all tests should pass with no collection warnings
2. Run `pytest tests/ --collect-only` — verify no duplicate test IDs
3. Count total tests before and after — the number should match or increase (if
   previously-hidden tests are now discovered)

## Edge Cases

- If a nighttime test imports from a relative path (e.g., `from ..src`), fix it to
  use absolute imports (`from src.whatever import ...`)
- If a test has an `__init__.py` that sets up paths, drop it — the main `tests/__init__.py`
  already exists
