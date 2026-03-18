# Task 022: Expose CLI Flags for Chunker, Embedder, Dataset, Retrieval Mode, and LLM Backend

## Goal

Add CLI arguments to `scripts/run_experiment.py` so users can select chunkers, embedders,
built-in datasets, retrieval mode, and LLM backend from the command line instead of
editing source code.

## Why

All the underlying components are already pluggable via Protocols — chunkers, embedders,
LLM backends, retrieval modes, and dataset loaders are all importable and wired up. But
`run_experiment.py` hardcodes most of these choices. Users (including us running experiments)
have to edit the script to change anything beyond models and strategies.

## File to Modify

**`scripts/run_experiment.py`** — this is the only file that changes.

## New CLI Arguments

Add these arguments to the existing `argparse.ArgumentParser`:

### `--chunkers`

- Type: comma-separated string
- Default: `"semantic,fixed,recursive"` (current hardcoded behavior)
- Valid values: `semantic`, `fixed`, `recursive`, `sentence`
- Maps to: `SemanticChunker`, `FixedSizeChunker`, `RecursiveChunker`, `SentenceChunker`
- Quick mode default: `"recursive"` (change from current `"semantic"` — recursive is the
  project's default baseline)

### `--embedder`

- Type: string (singular — one embedder per run)
- Default: `"ollama"`
- Valid values: `ollama`, `huggingface`, `google`
- Maps to: `OllamaEmbedder()`, `HuggingFaceEmbedder()`, `GoogleTextEmbedder()`
- `google` requires `GOOGLE_API_KEY` env var — raise clear error if missing

### `--dataset`

- Type: string
- Default: `None` (use `--corpus` CSV path, current behavior)
- Valid values: `hotpotqa`, `squad`
- When set, overrides `--corpus`. Loads via `sample_hotpotqa(n)` or `sample_squad(n)`.
  The `--sample` flag controls how many examples to load (default 50 for built-in
  datasets, 0=all for CSV).
- Built-in datasets produce `(documents, queries)` tuples. The experiment runner already
  accepts documents — but queries from built-in datasets should be saved to
  `{output_dir}/queries.json` for reproducibility.

### `--retrieval-mode`

- Type: string
- Default: `"hybrid"` (current default after task-019)
- Valid values: `hybrid`, `dense`, `sparse`
- Passed directly to `Experiment(retrieval_mode=...)`.

### `--llm-backend`

- Type: string
- Default: `"ollama"`
- Valid values: `ollama`, `openai-compat`
- `ollama` → `OllamaLLM()`
- `openai-compat` → `OpenAICompatibleLLM()` (uses default `http://localhost:1234`)
- Optionally accept `--llm-base-url` to override the OpenAI-compatible base URL.

## Implementation

### Argument parsing

Add all 5 arguments (+ `--llm-base-url`) to the existing `parse_args()` function.
Follow the same style as existing `--models` and `--strategies` arguments.

### `build_components()` changes

This function currently hardcodes chunkers, embedders, and LLM. Modify it to:

1. Accept the new parsed args
2. Build chunker list from `--chunkers` using a lookup dict:
   ```python
   CHUNKER_MAP = {
       "semantic": SemanticChunker,
       "fixed": FixedSizeChunker,
       "recursive": RecursiveChunker,
       "sentence": SentenceChunker,
   }
   ```
3. Build embedder from `--embedder` using a lookup dict (similar pattern)
4. Build LLM from `--llm-backend` (and `--llm-base-url` if openai-compat)
5. Load corpus from `--dataset` if set, otherwise from `--corpus` CSV path

### Dataset loading

When `--dataset` is used:
```python
if args.dataset == "hotpotqa":
    from src.datasets import sample_hotpotqa
    n = args.sample if args.sample > 0 else 50
    documents, queries = sample_hotpotqa(n=n, seed=42)
elif args.dataset == "squad":
    from src.datasets import sample_squad
    n = args.sample if args.sample > 0 else 50
    documents, queries = sample_squad(n=n, seed=42)
```

The existing CSV path remains the default when `--dataset` is not provided.

### Validation

- Unknown chunker/embedder/dataset/retrieval-mode/llm-backend names → clear error
  message listing valid options
- `--dataset` and `--corpus` are mutually exclusive — if both provided, error
- `--llm-base-url` without `--llm-backend openai-compat` → warning (ignored)
- Missing `GOOGLE_API_KEY` when `--embedder google` → error with instructions

## What NOT to touch

- Do NOT modify any source code in `src/` — all components already exist
- Do NOT change the `Experiment` class interface
- Do NOT change default behavior when no new flags are provided — existing
  `python scripts/run_experiment.py` should behave identically to before
- Do NOT add new dependencies

## Examples

```bash
# Current behavior (unchanged)
python scripts/run_experiment.py --models qwen3:4b --strategies naive

# New: use built-in dataset
python scripts/run_experiment.py --dataset hotpotqa --sample 100

# New: select chunkers and embedder
python scripts/run_experiment.py --chunkers recursive,sentence --embedder google

# New: use LM Studio instead of Ollama
python scripts/run_experiment.py --llm-backend openai-compat --llm-base-url http://localhost:1234

# New: dense-only retrieval
python scripts/run_experiment.py --retrieval-mode dense
```

## Verification

1. Run with no new flags — behavior identical to before
2. Run with `--chunkers fixed` — only FixedSizeChunker used
3. Run with `--dataset hotpotqa --sample 5` — loads 5 HotpotQA examples (will fail
   without Ollama, but should get past argument parsing and dataset loading)
4. Run with `--retrieval-mode dense` — passed through to Experiment
5. Run with invalid values — clear error messages
6. `--help` output shows all new arguments with descriptions
