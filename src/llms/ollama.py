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

    def __init__(self, host: str | None = None) -> None:
        """Initialize the Ollama client.

        Args:
            host: Ollama server URL. None uses the default localhost:11434.
        """
        self._client = Client(host=host) if host else Client()

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
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
