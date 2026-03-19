# Plan: task-026 — Update GPU Defaults and Pricing

## Files to modify
- `deploy/runpod_manager.py` — update DEFAULT_GPU_TYPES and comments
- `DaytimeNighttimeHandOff/DaytimeOnly/reference/runpod-setup-guide.md` — update sections 2, 6, 12

## Approach
1. Update DEFAULT_GPU_TYPES with new pricing and ordering
2. Update setup guide pricing table, cost estimates, and hour calculations
3. No test changes needed per spec

## Ambiguities
- Exact RunPod GPU type ID strings for newer GPUs (RTX 4000 Ada, RTX 2000 Ada) are uncertain. Will use consistent naming convention and add comments noting uncertainty.
