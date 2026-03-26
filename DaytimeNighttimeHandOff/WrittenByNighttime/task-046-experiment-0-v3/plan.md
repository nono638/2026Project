# Plan: task-046 — Experiment 0v3

## Files to Modify
- `scripts/run_experiment_0.py` — add `--generation-only` flag

## Files to Create
- `scripts/run_v3.py` — orchestrator script
- `tests/test_run_v3.py` — tests for wrapper script

## Approach

### Part 1: --generation-only flag
- Add argparse argument to `parse_args()`
- Add early return after raw_answers.csv save in `main()`

### Part 2: run_v3.py wrapper
- Follow spec's script flow exactly
- Import RunPodManager and setup_pod helpers
- Pod termination in finally block
- subprocess.run for experiment script calls

### Part 3: Run the experiment
- **Cannot execute**: requires network (RunPod API + cloud scoring APIs)
- Nighttime mode blocks network access
- Will mark this in result.md — user runs manually

## Ambiguities
- Spec says `manager.get_pod_url(pod_id)` — need to check if this method exists in RunPodManager.
  If not, will construct URL from pod data.
