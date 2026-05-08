"""Pre-written tests for task-055: cost guard entries for Ollama judges.

Ollama runs locally (GPU time, no API spend) — both judges must have
explicit $0.0 entries so the default $0.01 fallback doesn't burn the
cost guard ceiling on long retroactive scoring runs.

Night instance: copy these tests into `tests/test_cost_guard.py` (or
keep them as a separate file).
"""

from __future__ import annotations


def test_cost_guard_has_ollama_gemma4_31b_zero():
    from src.cost_guard import COST_PER_CALL
    assert "ollama:gemma4:31b" in COST_PER_CALL
    assert COST_PER_CALL["ollama:gemma4:31b"] == 0.0


def test_cost_guard_has_ollama_qwen35_27b_zero():
    from src.cost_guard import COST_PER_CALL
    assert "ollama:qwen3.5:27b" in COST_PER_CALL
    assert COST_PER_CALL["ollama:qwen3.5:27b"] == 0.0


def test_cost_guard_500_row_run_does_not_trip_ceiling():
    """At $0.0 per Ollama call, 500 rows × 2 judges must stay under $5."""
    from src.cost_guard import CostGuard
    guard = CostGuard(max_cost_usd=5.0)
    for _ in range(500):
        guard.record_call("ollama", "gemma4:31b")
        guard.record_call("ollama", "qwen3.5:27b")
    # No exception means the ceiling was not tripped
