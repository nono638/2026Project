# Task 026: Update GPU Defaults and Pricing

## What

Update the GPU fallback list in `deploy/runpod_manager.py` and pricing estimates
in the setup guide to reflect current RunPod availability and pricing (March 2026).

## Why

The original GPU defaults and pricing were based on late-2025 rates. Actual current
pricing (as seen by the user on 2026-03-19):
- RTX A5000: $0.27/hr (24GB) — cheapest the user found
- RTX 4000 Ada: $0.26/hr (available in "latest generation" section)
- RTX 2000 Ada: $0.24/hr (cheapest listed)
- L4: $0.39/hr
- RTX 3090: $0.46/hr
- RTX 4090: $0.59/hr
- A40: $0.40/hr (featured at $0.20/hr with savings plan)

The A4000 at $0.17/hr from the original guide may no longer be listed.

## Changes

### `deploy/runpod_manager.py`

1. **Update `DEFAULT_GPU_TYPES`** to reflect current cheapest-first ordering:

```python
DEFAULT_GPU_TYPES = [
    "NVIDIA RTX A5000",            # 24GB, ~$0.27/hr — cheapest 24GB
    "NVIDIA RTX 4000 Ada Generation",  # 20GB, ~$0.26/hr — good fallback
    "NVIDIA RTX A4000",            # 16GB — may still be available, cheapest if so
    "NVIDIA GeForce RTX 4090",     # 24GB, ~$0.59/hr — widely available last resort
]
```

NOTE: Check the exact GPU type ID strings that RunPod's API accepts. The current
code uses strings like `"NVIDIA RTX A4000"`. Keep the same naming convention.
The RunPod REST API `POST /pods` accepts `gpuTypeIds` as a list of strings.
If unsure of exact IDs, keep the existing format pattern.

2. **Update the comments** next to each GPU with current pricing.

### `DaytimeNighttimeHandOff/DaytimeOnly/reference/runpod-setup-guide.md`

1. **Section 12 (Understand the Costs) — GPU pricing table**: Update all prices
   to match current rates. Remove GPUs that are no longer available. Add RTX A5000
   and RTX 4000 Ada if not present.

2. **Section 12 — experiment cost estimates**: Recalculate based on ~$0.27/hr
   (cheapest realistic GPU) instead of $0.17/hr. Update the total estimate.

3. **Section 2 (Add Funds)**: The $25 recommendation and hour estimates should
   be recalculated at current prices.

4. **Section 6**: Update the expected output example to show realistic pricing
   ($0.27/hr instead of $0.17/hr).

### `tests/test_runpod_manager.py`

No test changes needed — the tests mock API responses and don't depend on the
default GPU list contents.

## What NOT to touch

- Do not modify the RunPodManager class methods or API logic
- Do not change `deploy/setup_pod.py` (it uses whatever defaults are in runpod_manager.py)
- Do not modify any src/ files

## Edge cases

- GPU type ID strings must match exactly what RunPod accepts. If the night instance
  is unsure about exact strings for newer GPUs (e.g., "NVIDIA RTX 4000 Ada Generation"
  vs "NVIDIA RTX 4000 Ada"), keep the safe option and add a comment noting the
  uncertainty. The user will verify when they actually run setup_pod.py.
