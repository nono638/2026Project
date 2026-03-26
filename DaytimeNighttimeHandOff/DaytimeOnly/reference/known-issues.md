# Known Issues

> **Daytime only** — night instance does not read this file.
>
> Bugs and limitations documented for awareness.

---

## Experiment 0v2 scoring process silent deaths (2026-03-25)
Scoring process died silently twice before incremental checkpointing was added. Possible causes: Windows process timeout, network drop, or silent exception in LLMScorer. Checkpoint fix prevents data loss but root cause unknown. Incremental checkpointing now mitigates this.

## bert_score module not installed (2026-03-26)
13 pre-existing test failures due to missing bert_score package. Not blocking experiments but should be installed for full test suite green.
