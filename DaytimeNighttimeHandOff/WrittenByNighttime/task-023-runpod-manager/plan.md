# Plan: task-023 — RunPod Management Module

## Files to Create
- `deploy/__init__.py` — empty init
- `deploy/runpod_manager.py` — main module with RunPodManager class, RunPodError, constants

## Files to Copy (tests)
- Copy pre-written tests from WrittenByDaytime to `tests/test_runpod_manager.py`

## Approach
1. Create `deploy/` package with `__init__.py`
2. Implement `deploy/runpod_manager.py` with:
   - Constants: `RUNPOD_REST_BASE`, `RUNPOD_GRAPHQL_URL`, `DEFAULT_GPU_TYPES`, `DEFAULT_PORTS`
   - `RunPodError` exception class with `status_code` and `response_body`
   - `RunPodManager` class with all 8 methods per spec:
     - `create_pod()` — POST to REST API with GPU fallback list
     - `terminate_pod()` — DELETE, ignore 404
     - `get_pod()` — GET, return None on 404
     - `list_pods()` — GET all pods
     - `get_pod_url()` — build proxy URL string
     - `wait_for_ready()` — poll get_pod with timeout
     - `get_balance()` — GraphQL query
     - `get_spend_per_hour()` — GraphQL query
3. Use `requests` library for HTTP. Auth via Bearer token header (REST) and query param (GraphQL).
4. All methods get type hints and docstrings per code quality standards.
5. `env` parameter accepts `dict[str, str]`, converts to RunPod's `[{"key": k, "value": v}]` format.

## Ambiguities
- None significant. Spec is detailed and tests are clear.

## Dependencies
- `requests` (already installed)
- No new packages needed
