# Known Issues

> **Daytime only** — night instance does not read this file.
>
> Bugs and limitations documented for awareness.

---

## Experiment 0v2 scoring process silent deaths (2026-03-25)
Scoring process died silently twice before incremental checkpointing was added. Possible causes: Windows process timeout, network drop, or silent exception in LLMScorer. Checkpoint fix prevents data loss but root cause unknown. Incremental checkpointing now mitigates this.

## bert_score module not installed (2026-03-26)
13 pre-existing test failures due to missing bert_score package. Not blocking experiments but should be installed for full test suite green.

## Gemini 3.1 Pro Preview is paid-tier-only (2026-04-30)
`google:gemini-3.1-pro-preview` is in `JUDGE_CONFIGS` but produces zero scored rows on every run. Confirmed via direct API smoke test: Google AI Studio free tier has `limit: 0` for this model — it requires paid billing. Other Gemini models (Flash-Lite, Flash, 2.5 Pro) are unaffected. Left in configs intentionally (option C) so it auto-activates if billing is enabled later; runner already skips it gracefully on 429. Reference: `gemini-api-billing-setup.md` for paid-tier setup notes if ever needed.
