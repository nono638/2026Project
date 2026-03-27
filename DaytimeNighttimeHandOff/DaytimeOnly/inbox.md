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

<!-- inbox cleared 2026-03-26 session -->
<!-- Triaged: gemini-3.1-pro-preview retry → dropping (v3 drops this judge entirely per task-046). Silent scoring death → reference/known-issues.md. -->

## 2026-03-26
- CostGuard bug: when limit is exceeded, it logs an error per-call but does NOT abort the scoring loop. API calls continue (200 OK), credits are spent, but scores are discarded. Should short-circuit the loop instead. Discovered during Exp 0v3 — wasted ~200 questions × 6 judges of API calls. → Specced as task-048.
- CostGuard cost table missing Opus: `anthropic:claude-opus-4-20250514` is not in COST_PER_CALL, so it falls through to DEFAULT_COST_PER_CALL ($0.01) — same as Sonnet. Opus is ~$0.075/call (15x Sonnet). This caused the cost estimate to undercount, triggering the guard late. Quick fix: add the Opus entry to src/cost_guard.py.
