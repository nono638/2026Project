"""OpenAI-compatible generation backend.

Works with LM Studio, vLLM, llama.cpp server, and actual OpenAI.
All speak the same chat completions API format.
"""

from __future__ import annotations

from openai import OpenAI


class OpenAICompatibleLLM:
    """OpenAI-compatible generation backend.

    Works with LM Studio, vLLM, llama.cpp server, and actual OpenAI.
    All speak the same chat completions API format.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            base_url: API endpoint URL. Defaults to LM Studio's default.
                      For vLLM: "http://localhost:8000/v1"
                      For OpenAI: "https://api.openai.com/v1"
            api_key: API key. LM Studio/vLLM don't validate this.
                     For actual OpenAI, pass your real key.
        """
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._base_url = base_url

    @property
    def name(self) -> str:
        """Return backend identifier including base URL."""
        return f"openai-compat:{self._base_url}"

    def generate(self, model: str, prompt: str) -> str:
        """Generate via OpenAI chat completions API.

        Args:
            model: Model identifier (e.g., 'local-model' for LM Studio).
            prompt: The complete prompt text.

        Returns:
            The model's generated text response.
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
