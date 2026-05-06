# task-053: Quantization tracking — Ollama metadata, CSV column, explicit tag pinning

## Why

Experiment 0 v1/v2/v3 ran on `qwen3:4b` without recording quantization. Ollama silently
resolved the tag to its default (Q4_K_M), but no metadata captures this — `metadata.json`
only stores the tag string, and the CSV has no quantization column. The proposal documents
("Q4 quantization") establish intent but not evidence. Reproducibility hole: Ollama can
re-point a default tag without notice; pulls done weeks apart could yield different
artifacts and the project would have no record.

This task closes the hole before Experiments 1, 2, and 3 (the upcoming quantization-axis
experiment) generate their data on the new local 5090 machine, where users may also
have different default resolutions than the prior RunPod pod.

## Scope

Three things, all behind one task because they share the same hook (Ollama's `/api/show`):

1. Capture quantization (and adjacent provenance fields) at run time in `metadata.json`
   and on every row of `raw_answers.csv` / `raw_scores.csv`.
2. Pin explicit quantization tags in default model lists so future pulls are deterministic.
3. Backfill existing Exp 0 v1/v2/v3 metadata files retroactively with a one-shot script.

This task **does not** run any new experiments. It is infrastructure-only.

## Requirements

### 1. New helper: `get_ollama_model_details`

Add to `scripts/experiment_utils.py`:

```python
def get_ollama_model_details(model_tag: str, host: str | None = None) -> dict:
    """Query Ollama's /api/show for resolved model details.

    Returns a dict with these keys (all str | None — None on lookup failure):
      - tag: the input tag, echoed back (e.g., "qwen3:4b")
      - digest: the resolved manifest digest (sha256:...)
      - quantization_level: e.g., "Q4_K_M", "F16"
      - parameter_size: e.g., "4.0B"
      - family: e.g., "qwen3"
      - format: e.g., "gguf"
      - captured_at: ISO timestamp of this lookup

    On any HTTP error, missing model, or unreachable host: log a warning,
    return a dict with the same keys but values=None (except tag and
    captured_at which are always populated). Never raise.
    """
```

Implementation: POST to `{host or 'http://localhost:11434'}/api/show` with body
`{"model": model_tag, "verbose": false}`. The response has `details.quantization_level`,
`details.parameter_size`, `details.family`, `details.format`, plus a top-level `digest`.
Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information

Use the `requests` library (already a project dep). 10s timeout. Wrap in try/except for
`requests.RequestException` and any `KeyError`/`TypeError` from missing response fields.

### 2. Capture in `metadata.json`

Update `write_experiment_metadata()` signature in `scripts/experiment_utils.py`:

```python
def write_experiment_metadata(
    output_dir: Path,
    n_examples: int,
    config: dict,
    judges: list[dict],
    model_details: dict[str, dict] | None = None,  # NEW
    extra: dict | None = None,
) -> None:
```

`model_details` is a map from `model_tag` → the dict returned by
`get_ollama_model_details`. Written to `metadata.json` under a top-level
`"model_details"` field. Optional (defaults to None / omitted) so existing callers
do not break.

Each experiment script populates and passes it:

- `run_experiment_0.py` — single model. `model_details = {args.model: get_ollama_model_details(args.model, host=args.ollama_host)}`. Call after model is confirmed loaded and before generation begins.
- `run_experiment_1.py` and `run_experiment_2.py` — multiple models. Build the map by iterating `args.models` (or the script's MODELS constant) once at startup; cache results in a module-level dict so subsequent rows reuse them.

### 3. Per-row CSV column: `llm_quantization`

Add `llm_quantization` to the row schema written by `experiment_utils.generate_answer()`
(or wherever the row dict is assembled before being appended to the answers CSV — locate
this in run_experiment_0.py's generation loop and the equivalent spot in
experiment_utils.generate_answer used by Exp 1/2).

Value: the `quantization_level` string from the cached `model_details` lookup for that
row's model. If lookup failed (None), write the literal string `"unknown"`.

The column appears in `raw_answers.csv` and (since scoring joins on the same df)
`raw_scores.csv`.

**Do not** also add `llm_digest`, `llm_parameter_size`, etc. as per-row columns —
those are run-level facts and live in metadata.json. Keeping the row schema lean.

### 4. Explicit quant tags in default model lists

Update these files to use explicit Ollama quantization tags:

- `deploy/setup_pod.py:60` — change `DEFAULT_MODELS = ["mxbai-embed-large", "qwen3:4b"]`
  to `DEFAULT_MODELS = ["mxbai-embed-large", "qwen3:4b-q4_K_M"]`.
- `scripts/pull_models.py` (if it exists — verify with Glob) — same treatment for any
  hardcoded model list.
- `scripts/run_experiment_1.py` — find the MODELS constant (around line 520, models for the matrix). Replace each tag with its explicit-quant variant. **Verify each tag exists** by running `ollama show <tag>` against ollama.com manifests before committing — not all Ollama models have explicit-quant variants. If a variant does not exist (e.g., `gemma3:1b-q4_K_M` may not be a published tag), keep the bare tag and add a code comment: `# bare tag — explicit-quant variant not published as of 2026-05-06; default resolves to Q4_K_M`.
- `scripts/run_experiment_2.py` — same.
- `docs/running-experiments.md` and `docs/output-format.md` — update example commands to use explicit tags. Add a one-line note in output-format.md mentioning the new `llm_quantization` column.

**Verified during daytime on 2026-05-06** by querying `https://ollama.com/library/<model>/tags`:

| Tag | Explicit `-q4_K_M` variant exists? | Use this tag |
|---|---|---|
| `qwen3.5:0.8b` | ❌ no (smallest explicit is `q8_0`) | bare `qwen3.5:0.8b` |
| `qwen3.5:2b` | ✅ | `qwen3.5:2b-q4_K_M` |
| `qwen3.5:4b` | ✅ | `qwen3.5:4b-q4_K_M` |
| `qwen3.5:9b` | ✅ | `qwen3.5:9b-q4_K_M` |
| `qwen3.6:27b` | ✅ | `qwen3.6:27b-q4_K_M` |
| `qwen3.6:35b-a3b` | ✅ | `qwen3.6:35b-a3b-q4_K_M` |
| `gemma4:e2b` | ✅ but as `-it-q4_K_M` form | `gemma4:e2b-it-q4_K_M` |
| `gemma4:e4b` | ✅ as `-it-q4_K_M` | `gemma4:e4b-it-q4_K_M` |
| `gemma4:26b` | ✅ as `-a4b-it-q4_K_M` (note: `-a4b` is in the tag) | `gemma4:26b-a4b-it-q4_K_M` |
| `gemma4:31b` | ✅ as `-it-q4_K_M` | `gemma4:31b-it-q4_K_M` |
| `mxbai-embed-large` | bare tag only — embedders typically don't have quant variants | `mxbai-embed-large` |

**Special handling for `qwen3.5:0.8b`:** keep the bare tag and add this comment in
`setup_pod.py` and any model-list literals:
```python
"qwen3.5:0.8b",  # bare tag — no -q4_K_M variant published as of 2026-05-06; default resolves to bf16/q8 for this small a model
```

The `get_ollama_model_details` helper will still capture whatever quantization the
default tag resolves to in `metadata.json` and the `llm_quantization` column —
that's the whole point of task-053. So the user's experiment will record the
actual quant on every row regardless of whether the tag was explicit.

Implementer should still spot-check by running `ollama show qwen3.5:0.8b` against
a live Ollama after pulling, and paste the `details.quantization_level` into
`nighttime_comments`. Same for any other tag where the verification table left
ambiguity.

### 5. Backfill script: `scripts/backfill_quant_metadata.py`

One-shot utility (not part of the experiment runners). Takes an experiment directory,
reads its `metadata.json`, queries Ollama for the model(s) referenced in `config.model`
(or `extra.test_models`), and writes a `model_details` field plus a
`backfill_note` field saying:

> "Backfilled retroactively on YYYY-MM-DD. The values reflect what the tag resolves to
> at backfill time, not necessarily at original run time. Original Exp 0 runs predate
> this audit; quantization is most likely Q4_K_M (Ollama default) but cannot be
> confirmed from primary evidence."

Usage:
```
python scripts/backfill_quant_metadata.py results/experiment_0
python scripts/backfill_quant_metadata.py results/experiment_0_v2
python scripts/backfill_quant_metadata.py results/experiment_0_v3
```

Does **not** modify the CSVs (no row-level llm_quantization in legacy data — too late).
Does **not** overwrite `model_details` if it already exists (idempotent skip with log
message). The user will run this manually after the 5090 has Ollama running with the
default tags pulled.

### 6. Tests

In `tests/test_experiment_utils.py` (or a new `tests/test_quantization_tracking.py`):

- `test_get_ollama_model_details_success` — mock requests.post, return a realistic Ollama
  /api/show response, assert all fields populated correctly.
- `test_get_ollama_model_details_unreachable_host` — mock requests.post raising
  `ConnectionError`, assert returns dict with None values and tag/captured_at set.
- `test_get_ollama_model_details_404` — mock 404 response, assert graceful None dict.
- `test_get_ollama_model_details_missing_fields` — mock partial response (no `details`
  key), assert None for missing fields, populated for present ones.
- `test_write_experiment_metadata_with_model_details` — call with model_details, assert
  the JSON has the field at top level.
- `test_write_experiment_metadata_without_model_details` — call without (default None),
  assert no `model_details` key in output JSON (don't write empty dict).
- `test_backfill_skips_existing_model_details` — pre-populate metadata.json with
  model_details, run backfill script, assert unchanged (idempotent).
- `test_backfill_writes_note` — run on fresh metadata.json, assert backfill_note field
  added.
- `test_csv_has_llm_quantization_column` — generate one synthetic row through
  generate_answer (with a mocked Ollama /api/show), assert the column is present and
  the value matches.

### 7. Tag pinning verification (one-time)

Before merging: implementer manually runs `python -c "from scripts.experiment_utils
import get_ollama_model_details; print(get_ollama_model_details('qwen3:4b'))"` against
a running local Ollama on the 5090 (or any reachable instance). Pastes the output into
`nighttime_comments` for confirmation that the helper actually works against a live
Ollama, not just mocks.

## What NOT to touch

- `scripts/run_experiment.py` (the legacy single-experiment script — superseded by
  experiment_0/1/2; leave alone).
- The existing CSVs in `results/experiment_0/`, `results/experiment_0_v2/`,
  `results/experiment_0_v3/` — do not retroactively add `llm_quantization` columns.
- The `OllamaLLM` and `OllamaEmbedder` classes in `src/` — quantization tracking is
  experiment-script provenance, not LLM interface concern. Don't push it into `src/`.
- Cloud judges (Anthropic/Google/OpenAI). Quantization is an Ollama-only concept here.

## Rationale

- **Why a single CSV column instead of digest + quantization_level + family columns?**
  Quantization is the dimension that varies across runs the user cares about. Digest
  and family are run-level constants and belong in metadata.json. Keeps the row schema
  lean — Exp 1/2 already have ~30 columns.

- **Why not also pin embedder quant?** mxbai-embed-large is published as a single
  artifact; Ollama's default tag resolves to F16 with no alternative quants. Pinning
  is moot.

- **Why log warnings instead of raising on lookup failure?** A misconfigured Ollama
  host shouldn't kill an experiment that already generated answers. Better to write
  `"unknown"` and surface the problem in logs than lose 6 hours of generation.

## Acceptance

- [ ] `get_ollama_model_details` implemented and tested.
- [ ] `write_experiment_metadata` accepts and writes `model_details`.
- [ ] All three experiment scripts populate and pass `model_details` to the metadata writer.
- [ ] `llm_quantization` column appears in `raw_answers.csv` and `raw_scores.csv` going
      forward.
- [ ] Default model lists in `setup_pod.py` (and any other pull scripts) use explicit
      quant tags where published variants exist; bare tags are commented to explain why.
- [ ] `scripts/backfill_quant_metadata.py` created, tested, and documented in commit.
- [ ] All new tests pass; existing tests still pass.
- [ ] Verification output from a live Ollama call pasted into `nighttime_comments`.

## References

- Ollama /api/show docs: https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information
- Ollama explicit-quant tag examples: https://ollama.com/library/qwen3/tags
- Q4_K_M quantization scheme (k-quants): https://github.com/ggerganov/llama.cpp/pull/1684
