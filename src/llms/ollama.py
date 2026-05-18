"""Ollama generation backend.

Wraps ollama.Client().chat() into the LLM protocol.
Default host: http://localhost:11434 (Ollama default).
"""

from __future__ import annotations

from ollama import Client


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

    def __init__(
        self,
        host: str | None = None,
        keep_alive: str | int | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            host: Ollama server URL. None uses the default localhost:11434.
            keep_alive: How long Ollama keeps this model resident after a
                call — a Go duration string (``"30m"``), an integer
                seconds value, or None to use ``DEFAULT_KEEP_ALIVE``.
        """
        self._client = Client(host=host) if host else Client()
        self._keep_alive = (
            keep_alive if keep_alive is not None else self.DEFAULT_KEEP_ALIVE
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
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            keep_alive=self._keep_alive,
        )
        return response.message.content
