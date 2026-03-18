"""LLM adapters for text generation backends."""

from src.llms.ollama import OllamaLLM
from src.llms.openai_compat import OpenAICompatibleLLM

__all__ = ["OllamaLLM", "OpenAICompatibleLLM"]
