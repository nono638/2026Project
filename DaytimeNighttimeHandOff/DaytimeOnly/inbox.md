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

## 2026-03-20
- Granular latency breakdown: split strategy_latency_ms into embedding/retrieval/reranking/generation — ready to spec for tonight
- top_k and chunk_overlap as experimental axes: engine supports both but CLI needs --chunk-overlap flag and experiment scripts need to loop over values — discuss when designing Exp 1 & 2
- Live demo guardrails: reject off-topic queries, block harmful outputs on the public-facing endpoint — spec when live demo work begins
