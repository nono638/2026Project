# Plan: task-010 — Portable Setup and Bootstrap Scripts

## Approach

1. Install python-dotenv
2. Create `.env.example` — API key template
3. Create `ENVIRONMENT.md` — setup documentation
4. Create `scripts/pull_models.py` — Ollama model puller
5. Create `scripts/smoke_test.py` — end-to-end verification
6. Create `scripts/run_experiment.py` — experiment CLI entry point
7. Create `setup.py` — bootstrap script (creates venv, installs deps, runs smoke test)
8. Verify: run smoke_test.py, run setup.py in already-configured env

## Files to Create
- `setup.py` — bootstrap script (stdlib only)
- `scripts/pull_models.py` — Ollama model puller
- `scripts/smoke_test.py` — smoke test
- `scripts/run_experiment.py` — experiment CLI
- `.env.example` — environment variable template
- `ENVIRONMENT.md` — setup documentation

## Files to Modify
- `requirements.txt` — add python-dotenv pin

## Ambiguities
- Spec says smoke test should check Ollama and embedding, but we're running in a sandboxed env without Ollama. Smoke test will handle gracefully (warn, don't fail).
- Spec says setup.py should "ask user" about pulling models — but we're in unattended mode. The script will include the interactive prompt for normal use.
- task-007 is done, so run_experiment.py can be fully wired (not just a skeleton).
