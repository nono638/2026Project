"""Tests for LLM Protocol and adapters.

All API calls are mocked — no Ollama or LM Studio required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.protocols import LLM


# ---------------------------------------------------------------------------
# Mock LLM for strategy testing
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns a canned response.

    Satisfies the LLM protocol. Tracks calls for assertions.
    """

    def __init__(self, response: str = "Mock answer based on context"):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "mock:llm"

    def generate(self, model: str, prompt: str) -> str:
        self.calls.append((model, prompt))
        return self._response


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestLLMProtocol:
    """Verify LLM protocol is runtime-checkable."""

    def test_mock_llm_satisfies_protocol(self):
        """MockLLM should satisfy the LLM protocol."""
        assert isinstance(MockLLM(), LLM)

    def test_ollama_adapter_satisfies_protocol(self):
        """OllamaLLM should satisfy the LLM protocol."""
        from src.llms import OllamaLLM
        # Patch Client to avoid actual connection
        with patch("src.llms.ollama.Client"):
            llm = OllamaLLM()
            assert isinstance(llm, LLM)

    def test_openai_adapter_satisfies_protocol(self):
        """OpenAICompatibleLLM should satisfy the LLM protocol."""
        from src.llms import OpenAICompatibleLLM
        with patch("src.llms.openai_compat.OpenAI"):
            llm = OpenAICompatibleLLM()
            assert isinstance(llm, LLM)

    def test_non_llm_fails_check(self):
        """A plain object should not satisfy the LLM protocol."""
        assert not isinstance("not an llm", LLM)
        assert not isinstance(42, LLM)


# ---------------------------------------------------------------------------
# OllamaLLM adapter
# ---------------------------------------------------------------------------

class TestOllamaLLM:
    """Test OllamaLLM adapter."""

    def test_name(self):
        """OllamaLLM should have name 'ollama'."""
        from src.llms import OllamaLLM
        with patch("src.llms.ollama.Client"):
            llm = OllamaLLM()
            assert llm.name == "ollama"

    def test_generate_calls_chat(self):
        """generate() should call ollama Client.chat() with correct args."""
        from src.llms import OllamaLLM
        with patch("src.llms.ollama.Client") as MockClient:
            mock_client = MockClient.return_value
            mock_response = MagicMock()
            mock_response.message.content = "Generated text"
            mock_client.chat.return_value = mock_response

            llm = OllamaLLM()
            result = llm.generate("qwen3:4b", "What is Python?")

            assert result == "Generated text"
            mock_client.chat.assert_called_once_with(
                model="qwen3:4b",
                messages=[{"role": "user", "content": "What is Python?"}],
            )

    def test_custom_host(self):
        """OllamaLLM should accept a custom host."""
        from src.llms import OllamaLLM
        with patch("src.llms.ollama.Client") as MockClient:
            llm = OllamaLLM(host="http://gpu-server:11434")
            MockClient.assert_called_with(host="http://gpu-server:11434")


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM adapter
# ---------------------------------------------------------------------------

class TestOpenAICompatibleLLM:
    """Test OpenAI-compatible adapter (LM Studio, vLLM, etc.)."""

    def test_name_includes_base_url(self):
        """Name should include the base URL for identification."""
        from src.llms import OpenAICompatibleLLM
        with patch("src.llms.openai_compat.OpenAI"):
            llm = OpenAICompatibleLLM(base_url="http://localhost:1234/v1")
            assert "localhost:1234" in llm.name

    def test_generate_calls_completions(self):
        """generate() should call OpenAI chat.completions.create()."""
        from src.llms import OpenAICompatibleLLM
        with patch("src.llms.openai_compat.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "LM Studio answer"
            mock_client.chat.completions.create.return_value = mock_response

            llm = OpenAICompatibleLLM(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
            )
            result = llm.generate("local-model", "What is Python?")

            assert result == "LM Studio answer"
            mock_client.chat.completions.create.assert_called_once_with(
                model="local-model",
                messages=[{"role": "user", "content": "What is Python?"}],
            )

    def test_default_base_url(self):
        """Default base_url should be LM Studio's localhost:1234."""
        from src.llms import OpenAICompatibleLLM
        with patch("src.llms.openai_compat.OpenAI") as MockOpenAI:
            llm = OpenAICompatibleLLM()
            MockOpenAI.assert_called_once_with(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
            )


# ---------------------------------------------------------------------------
# Strategy integration — all 5 strategies accept LLM
# ---------------------------------------------------------------------------

class TestStrategyAcceptsLLM:
    """Verify all 5 strategies can be constructed with a mock LLM."""

    def test_naive_rag(self):
        """NaiveRAG should accept llm parameter."""
        from src.strategies import NaiveRAG
        llm = MockLLM()
        strategy = NaiveRAG(llm=llm)
        assert strategy.name == "naive"

    def test_multi_query_rag(self):
        """MultiQueryRAG should accept llm parameter."""
        from src.strategies import MultiQueryRAG
        llm = MockLLM()
        strategy = MultiQueryRAG(llm=llm)
        assert strategy.name == "multi_query"

    def test_corrective_rag(self):
        """CorrectiveRAG should accept llm parameter."""
        from src.strategies import CorrectiveRAG
        llm = MockLLM()
        strategy = CorrectiveRAG(llm=llm)
        assert strategy.name == "corrective"

    def test_self_rag(self):
        """SelfRAG should accept llm parameter."""
        from src.strategies import SelfRAG
        llm = MockLLM()
        strategy = SelfRAG(llm=llm)
        assert strategy.name == "self_rag"

    def test_adaptive_rag(self):
        """AdaptiveRAG should accept llm parameter."""
        from src.strategies import AdaptiveRAG
        llm = MockLLM()
        strategy = AdaptiveRAG(llm=llm)
        assert strategy.name == "adaptive"


class TestStrategyDelegatesToLLM:
    """Verify strategies call self._llm.generate() instead of ollama directly."""

    def _make_mock_retriever(self):
        """Create a mock retriever returning canned results."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Python is a programming language.", "score": 0.9, "index": 0},
            {"text": "Python was created by Guido van Rossum.", "score": 0.8, "index": 1},
        ]
        return retriever

    def test_naive_delegates_to_llm(self):
        """NaiveRAG.run() should call llm.generate() exactly once."""
        from src.strategies import NaiveRAG
        llm = MockLLM(response="Python is a language")
        strategy = NaiveRAG(llm=llm)
        retriever = self._make_mock_retriever()

        result = strategy.run("What is Python?", retriever, "qwen3:4b")

        assert result == "Python is a language"
        assert len(llm.calls) == 1
        model, prompt = llm.calls[0]
        assert model == "qwen3:4b"
        assert "What is Python?" in prompt

    def test_multi_query_delegates_to_llm(self):
        """MultiQueryRAG.run() should call llm.generate() at least twice."""
        from src.strategies import MultiQueryRAG
        llm = MockLLM(response="Generated text")
        strategy = MultiQueryRAG(llm=llm)
        retriever = self._make_mock_retriever()

        strategy.run("What is Python?", retriever, "qwen3:4b")

        # MultiQuery: at least 1 call for rephrasing + 1 for answer
        assert len(llm.calls) >= 2
        # All calls should use the same model
        for model, _ in llm.calls:
            assert model == "qwen3:4b"

    def test_no_ollama_import_in_strategies(self):
        """Strategy modules should not import ollama.Client directly."""
        import src.strategies.naive as naive_mod
        import src.strategies.multi_query as mq_mod
        import src.strategies.corrective as corr_mod
        import src.strategies.self_rag as sr_mod
        import src.strategies.adaptive as adapt_mod

        for mod in [naive_mod, mq_mod, corr_mod, sr_mod, adapt_mod]:
            source = open(mod.__file__).read()
            assert "from ollama import" not in source, \
                f"{mod.__name__} still imports ollama directly"
            assert "ollama.Client" not in source, \
                f"{mod.__name__} still references ollama.Client"


class TestRunExperimentScript:
    """Verify run_experiment.py creates strategies with an LLM."""

    def test_build_components_uses_llm(self):
        """build_components() should pass an LLM to strategies."""
        # This is a structural test — just verify the script imports LLM
        import importlib
        import scripts.run_experiment as script_mod
        importlib.reload(script_mod)
        source = open(script_mod.__file__).read()
        assert "OllamaLLM" in source, \
            "run_experiment.py should use OllamaLLM"
