# Task 023: RunPod Management Module

## What to Build

A `deploy/runpod_manager.py` module that wraps RunPod's REST API for pod lifecycle
management. This is deployment infrastructure — it does NOT touch the experiment code
in `src/`. RAGBench users with their own GPU never interact with this module.

## Why

We're renting GPU compute from RunPod for experiments and a live demo. We need
programmatic control to:
- Create pods with GPU fallback (try A4000 first, fall back to RTX 3090, etc.)
- Terminate pods when done (not just stop — avoids the "resume gets zero GPUs" problem)
- Check pod readiness before sending requests
- Query remaining account balance for the budget-aware demo UI
- Auto-stop after idle timeout to save money

## Files to Create

### `deploy/__init__.py`
Empty init file.

### `deploy/runpod_manager.py`
The main module. All RunPod API interaction lives here.

### `tests/test_runpod_manager.py`
Tests with mocked HTTP responses. No real API calls.

## Files NOT to Touch

- Everything in `src/` — the experiment pipeline is provider-agnostic
- `scripts/run_experiment.py` — not yet; wiring comes in a future task
- `.env` or `.env.example` — don't add RUNPOD_API_KEY yet

## Detailed Specification

### Class: `RunPodManager`

```python
class RunPodManager:
    def __init__(self, api_key: str, default_gpu_types: list[str] | None = None):
        """
        Args:
            api_key: RunPod API key.
            default_gpu_types: Ordered GPU fallback list.
                Default: ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"]
        """
```

### Methods

#### `create_pod(name, image_name, ports, volume_gb, env, gpu_types=None) -> dict`
Create a new pod via REST API `POST https://rest.runpod.io/v1/pods`.

- Use `gpuTypeIds` (array) with `gpuTypePriority: "availability"` so RunPod picks
  whichever GPU is in stock from the fallback list.
- `gpu_types` param overrides `self.default_gpu_types` if provided.
- `ports` should default to `["11434/http", "8000/http"]` (Ollama + FastAPI).
- Set `cloudType: "ALL"` to search both secure and community cloud.
- Return the full pod dict from RunPod's response (includes `id`, `desiredStatus`, etc.).
- Raise `RunPodError` on non-2xx response.

REST API docs: https://docs.runpod.io/api-reference/pods/POST/pods

Example request body:
```json
{
  "name": "ragbench-gpu",
  "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "gpuTypeIds": ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "volumeInGb": 20,
  "ports": ["11434/http", "8000/http"],
  "cloudType": "ALL",
  "env": [{"key": "OLLAMA_HOST", "value": "0.0.0.0"}]
}
```

#### `terminate_pod(pod_id: str) -> None`
Terminate (permanently delete) a pod via `DELETE https://rest.runpod.io/v1/pods/{podId}`.

We use terminate+recreate instead of stop+resume because:
- Stopped pods can resume with zero GPUs if the machine was re-rented
- Terminate+recreate uses the GPU fallback list and always gets a GPU
- Data survives via network volumes (future enhancement)

Raise `RunPodError` on non-2xx response. Ignore 404 (pod already gone).

#### `get_pod(pod_id: str) -> dict | None`
Get pod status via `GET https://rest.runpod.io/v1/pods/{podId}`.

Return the pod dict, or `None` if 404. Raise `RunPodError` on other errors.

#### `list_pods() -> list[dict]`
List all pods via `GET https://rest.runpod.io/v1/pods`.

Return the list of pod dicts.

#### `get_pod_url(pod_id: str, port: int = 11434) -> str`
Build the stable HTTP proxy URL for a pod.

Format: `https://{pod_id}-{port}.proxy.runpod.net`

This URL is stable across pod restarts (it's based on pod ID, which doesn't change).
See: https://docs.runpod.io/pods/configuration/expose-ports

#### `wait_for_ready(pod_id: str, timeout_s: int = 300, poll_interval_s: int = 5) -> bool`
Poll `get_pod()` until `desiredStatus == "RUNNING"` and `runtime` is populated.

Return `True` if ready within timeout, `False` if timeout exceeded.
Use `time.sleep(poll_interval_s)` between polls.

#### `get_balance() -> float`
Query account balance via GraphQL (not available on REST API).

GraphQL endpoint: `https://api.runpod.io/graphql?api_key={api_key}`

Query:
```graphql
query { myself { currentSpendPerHr clientBalance } }
```

Return `clientBalance` as a float (USD).

See: https://docs.runpod.io/sdks/graphql/manage-pods

#### `get_spend_per_hour() -> float`
Query current spend rate via the same GraphQL `myself` query.

Return `currentSpendPerHr` as a float (USD/hr).

### Class: `RunPodError`
Simple exception subclass for API errors. Include the HTTP status code and response body.

```python
class RunPodError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
```

### Constants

```python
RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

DEFAULT_GPU_TYPES = [
    "NVIDIA RTX A4000",        # 16GB, ~$0.17/hr — cheapest viable
    "NVIDIA GeForce RTX 3090", # 24GB, ~$0.22/hr — good fallback
    "NVIDIA GeForce RTX 4090", # 24GB, ~$0.34/hr — widely available
]

DEFAULT_PORTS = ["11434/http", "8000/http"]  # Ollama + FastAPI
```

## Implementation Notes

- Use `requests` library for HTTP calls (already in requirements.txt).
- Auth header for REST: `{"Authorization": f"Bearer {api_key}"}`.
- Auth for GraphQL: pass as query param `?api_key={api_key}`.
- All methods should log actions via `logging.getLogger(__name__)`.
- Type hints on everything, `from __future__ import annotations`.
- The `env` parameter for `create_pod` should accept a `dict[str, str]` and convert
  to RunPod's format: `[{"key": k, "value": v} for k, v in env.items()]`.

## Edge Cases

- `create_pod` with all GPU types unavailable: RunPod returns an error. Surface it
  via `RunPodError` with a clear message like "No GPUs available from fallback list."
- `terminate_pod` on already-terminated pod: 404 is fine, don't raise.
- `wait_for_ready` timeout: return `False`, don't raise. Let the caller decide.
- `get_balance` with invalid API key: raise `RunPodError`.
- Network errors (connection refused, timeout): let `requests.RequestException`
  propagate — don't swallow them.

## What NOT to Build

- No auto-stop/idle detection yet (future task)
- No network volume management yet (future task)
- No Ollama setup/model pulling (future task — that's SSH/exec commands)
- No integration with `src/` experiment code
- No CLI commands
- No `.env` changes

## Tests

All tests use `unittest.mock.patch` to mock `requests.get`, `requests.post`,
`requests.delete`. No real API calls.

### Test cases:

1. **test_create_pod_success** — mock 201 response, verify request body has correct
   `gpuTypeIds`, `gpuTypePriority`, `ports`, and `env` format.

2. **test_create_pod_custom_gpu_types** — pass custom `gpu_types` to `create_pod`,
   verify it overrides defaults.

3. **test_create_pod_api_error** — mock 400 response, verify `RunPodError` raised
   with status code and body.

4. **test_terminate_pod_success** — mock 204 response, verify DELETE called with
   correct URL.

5. **test_terminate_pod_already_gone** — mock 404 response, verify no error raised.

6. **test_get_pod_success** — mock 200 response, verify pod dict returned.

7. **test_get_pod_not_found** — mock 404 response, verify `None` returned.

8. **test_list_pods** — mock 200 response with list, verify list returned.

9. **test_get_pod_url** — verify URL format: `https://{pod_id}-{port}.proxy.runpod.net`.

10. **test_wait_for_ready_success** — mock `get_pod` returning not-ready then ready,
    verify returns `True`.

11. **test_wait_for_ready_timeout** — mock `get_pod` always returning not-ready,
    verify returns `False` after timeout.

12. **test_get_balance** — mock GraphQL response with `clientBalance`, verify float returned.

13. **test_get_spend_per_hour** — mock GraphQL response with `currentSpendPerHr`,
    verify float returned.

14. **test_env_dict_converted** — verify `{"FOO": "bar"}` becomes
    `[{"key": "FOO", "value": "bar"}]` in the request body.

15. **test_default_gpu_types_used** — create pod without `gpu_types` param, verify
    `DEFAULT_GPU_TYPES` used in request.
