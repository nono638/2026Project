# Inbox

> **Managed by daytime Claude. Night instance does not read this file.**
>
> Zero-friction capture for anything that comes up mid-session but isn't being acted on
> right now. No judgment required to add — if it might matter, capture it with one sentence
> of context. Cleared completely at every session-open triage.

---

<!-- Add captures below with today's date header. Format:
## YYYY-MM-DD
- [one-line capture — what it is and why it came up]
-->

<!-- inbox cleared 2026-03-20 session -->

## 2026-03-25
- Re-run gemini-3.1-pro-preview scoring tomorrow (quota resets daily): `python scripts/run_experiment_0.py --skip-generation --judges gemini-3.1-pro-preview` — hit 250/day rate limit, only 11/150 scored. The merge logic will fold new scores into existing raw_scores.csv.
- Experiment 0v2 scoring process died silently twice before incremental checkpointing was added — investigate why (possible Windows process timeout, network drop, or silent exception). The checkpoint fix prevents data loss but doesn't explain root cause.
