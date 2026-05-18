"""Ollama generation backend.

Wraps ollama.Client().chat() into the LLM protocol.
Default host: http://localhost:11434 (Ollama default).
"""

from __future__ import annotations

from ollama import Client

from src.monitoring import call_tracker


class OllamaLLM:
    """Ollama generation backend.

    Wraps ollama.Client().chat() into the LLM protocol.
    Default host: http://localhost:11434 (Ollama default).
    """

    # keep_alive='30m' keeps the model resident across all ~200 queries of a
    # single experiment config. Ollama's default of 5 minutes is fine for
    # interactive use but in long batch runs a slow generation (or a brief
    # judge-side stall) can let the timer expire mid-config, forcing a full
    # reload on the next call. 30 minutes safely spans worst-case config
    # duration without locking the GPU when the experiment moves to the next
    # model (Ollama evicts on model-switch regardless of keep_alive).
    DEFAULT_KEEP_ALIVE = "30m"

    # Hard ceiling on tokens generated per call. Without this, qwen3.5:9b
    # was observed generating runaway responses of 8-17+ minutes on single
    # queries (BSOD #3, 2026-05-18).
    #
    # Sized from analysis of completed Exp 1 rows (qwen3.5 0.8b/2b/4b ×
    # 5 strategies, n=200 each):
    #   - naive/multi_query/corrective: 99th-pct answer length ≈ 350 tokens
    #   - adaptive: 99th-pct ≈ 800 tokens (one strategy has multi-hop chain)
    #   - self_rag: 99th-pct ≈ 2400 tokens (critique + answer; longest tail)
    # 1024 covers >95% of non-degenerate answers across all 5 strategies
    # while still ensuring worst-case generation completes well under the
    # 180 s OllamaLLM timeout. Higher values (>2000) let the runaway-loop
    # outputs that drove BSOD #3 back in.
    DEFAULT_NUM_PREDICT = 1024

    # Per-call timeout (seconds) passed to the underlying httpx client.
    # Ollama's Python Client defaults to None (no timeout), which caused
    # the 8-17 minute hung chat() calls seen before BSOD #3. 180s is 3×
    # the expected worst-case for a 9B model on the 5090 Laptop GPU.
    DEFAULT_TIMEOUT = 180.0

    def __init__(
        self,
        host: str | None = None,
        keep_alive: str | int | None = None,
        timeout: float | None = None,
        num_predict: int | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            host: Ollama server URL. None uses the default localhost:11434.
            keep_alive: How long Ollama keeps this model resident after a
                call — a Go duration string (``"30m"``), an integer
                seconds value, or None to use ``DEFAULT_KEEP_ALIVE``.
            timeout: Per-request timeout in seconds. None uses
                ``DEFAULT_TIMEOUT`` (180 s). Pass 0 to disable.
            num_predict: Token generation cap passed as Ollama's
                ``num_predict`` option. None uses ``DEFAULT_NUM_PREDICT``
                (512). Pass -1 to disable the cap (not recommended for
                batch runs on the 5090).
        """
        _timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._client = Client(host=host, timeout=_timeout)
        self._keep_alive = (
            keep_alive if keep_alive is not None else self.DEFAULT_KEEP_ALIVE
        )
        self._num_predict = (
            num_predict if num_predict is not None else self.DEFAULT_NUM_PREDICT
        )

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "ollama"

    def generate(self, model: str, prompt: str) -> str:
        """Generate via Ollama chat API.

        Args:
            model: Ollama model name (e.g., 'qwen3:4b').
            prompt: The complete prompt text.

        Returns:
            The model's generated text response.
        """
        # think=False: thinking-capable models (Qwen 3.5 family) otherwise
        # consume the response budget on hidden CoT and emit empty content.
        # Same fix applied to the Ollama judge adapter on 2026-05-07.
        options: dict = {}
        if self._num_predict and self._num_predict > 0:
            options["num_predict"] = self._num_predict
        # Mark this call as in-flight so the runner's per-row heartbeat can
        # report what was happening if a BSOD strikes mid-call. record_end
        # is in a finally so an exception still clears the in_flight flag.
        call_tracker.record_start("chat", model)
        try:
            response = self._client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                keep_alive=self._keep_alive,
                options=options or None,
            )
        finally:
            call_tracker.record_end()
        # Ollama can return None content when the model emits no visible tokens
        # (e.g., thinking-only output even with think=False, or refusal).
        # Strategies call .strip() on the result, so a None would crash them.
        return response.message.content or ""
