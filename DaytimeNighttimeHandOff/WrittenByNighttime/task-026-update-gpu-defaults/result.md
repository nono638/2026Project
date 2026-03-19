# Result: task-026 — Update GPU Defaults and Pricing
**Status:** done
**Completed:** 2026-03-19T15:17:46

## Commits
- `<pending>` — night: task-026 update GPU defaults and pricing

## Test Results
- Command run: `python -m pytest tests/test_runpod_manager.py -v`
- Outcome: 16 passed, 0 failed
- Failures: none

## Decisions Made
- Used "NVIDIA RTX 4000 Ada Generation" as the GPU type ID string, following the existing naming convention. Added a comment noting the exact ID may vary — user will verify at runtime.
- Recalculated all cost estimates at $0.27/hr (A5000) instead of $0.17/hr (A4000).
- Updated Experiment 0 cost estimate from ~$1.80 to ~$0.60 reflecting Gemini-only judges (much cheaper than Claude Opus).

## Flags for Morning Review
- "NVIDIA RTX 4000 Ada Generation" GPU type ID string is uncertain — verify with RunPod when running setup_pod.py. If it fails, try "NVIDIA RTX 4000 Ada" instead.
