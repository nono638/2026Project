"""Tests for strategy run() logic: AdaptiveRAG, CorrectiveRAG, SelfRAG.

All LLM and Retriever interactions are mocked — no external services needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.strategies.adaptive import AdaptiveRAG
from src.strategies.corrective import CorrectiveRAG
from src.strategies.self_rag import SelfRAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm() -> MagicMock:
    """Return a mock LLM with a generate method."""
    llm = MagicMock()
    llm.generate = MagicMock()
    return llm


def _mock_retriever(chunks: list[str] | None = None) -> MagicMock:
    """Return a mock Retriever that yields the given chunk texts."""
    if chunks is None:
        chunks = ["chunk-A", "chunk-B", "chunk-C"]
    retriever = MagicMock()
    retriever.retrieve.return_value = [{"text": c} for c in chunks]
    return retriever


# ===========================================================================
# AdaptiveRAG
# ===========================================================================


class TestAdaptiveClassify:
    """Tests for AdaptiveRAG._classify — query complexity routing."""

    def test_classify_simple(self) -> None:
        llm = _mock_llm()
        llm.generate.return_value = "simple"
        strategy = AdaptiveRAG(llm=llm)
        assert strategy._classify("What is Python?", "m") == "simple"

    def test_classify_moderate(self) -> None:
        llm = _mock_llm()
        llm.generate.return_value = "moderate"
        strategy = AdaptiveRAG(llm=llm)
        assert strategy._classify("Compare X and Y", "m") == "moderate"

    def test_classify_complex(self) -> None:
        llm = _mock_llm()
        llm.generate.return_value = "This is clearly complex"
        strategy = AdaptiveRAG(llm=llm)
        assert strategy._classify("Multi-step question", "m") == "complex"

    def test_classify_unparseable_defaults_to_moderate(self) -> None:
        llm = _mock_llm()
        llm.generate.return_value = "I don't know how to classify this"
        strategy = AdaptiveRAG(llm=llm)
        assert strategy._classify("Weird query", "m") == "moderate"


class TestAdaptiveRun:
    """Tests for AdaptiveRAG.run — end-to-end routing."""

    def test_simple_route_skips_retrieval(self) -> None:
        llm = _mock_llm()
        # First call: classify → simple; second call: generate answer
        llm.generate.side_effect = ["simple", "Direct answer"]
        retriever = _mock_retriever()
        strategy = AdaptiveRAG(llm=llm)

        result = strategy.run("What is 2+2?", retriever, "model-a")

        assert result == "Direct answer"
        retriever.retrieve.assert_not_called()

    def test_moderate_route_retrieves_once(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = ["moderate", "Answer from context"]
        retriever = _mock_retriever(["ctx-1", "ctx-2"])
        strategy = AdaptiveRAG(llm=llm)

        result = strategy.run("Synthesize A and B", retriever, "model-a")

        assert result == "Answer from context"
        retriever.retrieve.assert_called_once_with("Synthesize A and B")

    def test_complex_route_retrieves_twice(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = [
            "complex",            # classify
            "Preliminary answer", # first-pass generation
            "Follow-up query?",   # follow-up formulation
            "Final answer",       # final generation
        ]
        retriever = _mock_retriever(["c1", "c2"])
        strategy = AdaptiveRAG(llm=llm)

        result = strategy.run("Multi-hop question", retriever, "model-a")

        assert result == "Final answer"
        assert retriever.retrieve.call_count == 2


# ===========================================================================
# CorrectiveRAG
# ===========================================================================


class TestCorrectiveFilterRelevant:
    """Tests for CorrectiveRAG._filter_relevant."""

    def test_keeps_relevant_chunks(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = ["relevant", "partially relevant", "irrelevant"]
        strategy = CorrectiveRAG(llm=llm)
        retrieved = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

        result = strategy._filter_relevant("q", retrieved, "m")

        assert result == ["a", "b"]

    def test_keeps_all_when_none_irrelevant(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = ["relevant", "relevant"]
        strategy = CorrectiveRAG(llm=llm)
        retrieved = [{"text": "a"}, {"text": "b"}]

        result = strategy._filter_relevant("q", retrieved, "m")

        assert result == ["a", "b"]


class TestCorrectiveRun:
    """Tests for CorrectiveRAG.run — filter, reformulate, fallback."""

    def test_sufficient_relevant_chunks_no_reformulation(self) -> None:
        """When >= 2 chunks survive filtering, skip reformulation."""
        llm = _mock_llm()
        # 3 relevance ratings (all kept) + 1 final generation
        llm.generate.side_effect = ["relevant", "relevant", "relevant", "Good answer"]
        retriever = _mock_retriever(["a", "b", "c"])
        strategy = CorrectiveRAG(llm=llm)

        result = strategy.run("question", retriever, "m")

        assert result == "Good answer"
        retriever.retrieve.assert_called_once()

    def test_reformulation_triggered_when_few_survive(self) -> None:
        """When < 2 chunks survive, reformulate and retry."""
        llm = _mock_llm()
        llm.generate.side_effect = [
            "irrelevant",           # filter pass 1 chunk 1
            "irrelevant",           # filter pass 1 chunk 2
            "irrelevant",           # filter pass 1 chunk 3
            "Better question",      # reformulation
            "relevant",             # filter pass 2 chunk 1
            "relevant",             # filter pass 2 chunk 2
            "relevant",             # filter pass 2 chunk 3
            "Answer after reform",  # final generation
        ]
        retriever = _mock_retriever(["a", "b", "c"])
        strategy = CorrectiveRAG(llm=llm)

        result = strategy.run("vague question", retriever, "m")

        assert result == "Answer after reform"
        assert retriever.retrieve.call_count == 2

    def test_fallback_to_top_2_when_nothing_survives(self) -> None:
        """When nothing survives even after reformulation, use top 2 originals."""
        llm = _mock_llm()
        llm.generate.side_effect = [
            "irrelevant", "irrelevant",  # filter pass 1
            "Reformulated",              # reformulation
            "irrelevant", "irrelevant",  # filter pass 2
            "Fallback answer",           # final generation
        ]
        retriever = _mock_retriever(["orig-1", "orig-2"])
        strategy = CorrectiveRAG(llm=llm)

        result = strategy.run("bad question", retriever, "m")

        assert result == "Fallback answer"
        # Verify the generation prompt includes the original chunks as fallback
        final_call_prompt = llm.generate.call_args_list[-1][0][1]
        assert "orig-1" in final_call_prompt
        assert "orig-2" in final_call_prompt


# ===========================================================================
# SelfRAG
# ===========================================================================


class TestSelfRAGRun:
    """Tests for SelfRAG.run — decide, retrieve, evaluate, generate, critique."""

    def test_skips_retrieval_when_model_says_no(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = ["no", "Direct answer"]
        retriever = _mock_retriever()
        strategy = SelfRAG(llm=llm)

        result = strategy.run("What is 1+1?", retriever, "m")

        assert result == "Direct answer"
        retriever.retrieve.assert_not_called()

    def test_full_path_retrieve_evaluate_generate_critique(self) -> None:
        llm = _mock_llm()
        llm.generate.side_effect = [
            "yes",                 # retrieval decision
            "relevant",            # evaluate chunk 1
            "relevant",            # evaluate chunk 2
            "Draft answer",        # generate
            "Revised final answer",  # critique
        ]
        retriever = _mock_retriever(["ctx-1", "ctx-2"])
        strategy = SelfRAG(llm=llm)

        result = strategy.run("Complex question", retriever, "m")

        assert result == "Revised final answer"
        retriever.retrieve.assert_called_once()
        assert llm.generate.call_count == 5

    def test_fallback_when_all_chunks_irrelevant(self) -> None:
        """When all chunks are filtered out, use top 2 anyway."""
        llm = _mock_llm()
        llm.generate.side_effect = [
            "yes",              # retrieval decision
            "irrelevant",       # evaluate chunk 1
            "irrelevant",       # evaluate chunk 2
            "irrelevant",       # evaluate chunk 3
            "Draft answer",     # generate (with fallback chunks)
            "Final answer",     # critique
        ]
        retriever = _mock_retriever(["c1", "c2", "c3"])
        strategy = SelfRAG(llm=llm)

        result = strategy.run("Hard question", retriever, "m")

        assert result == "Final answer"
        # Verify fallback chunks are used in the generate prompt
        generate_call_prompt = llm.generate.call_args_list[4][0][1]
        assert "c1" in generate_call_prompt
        assert "c2" in generate_call_prompt

    def test_critique_can_return_unchanged(self) -> None:
        """Critique step may return the answer unchanged."""
        llm = _mock_llm()
        llm.generate.side_effect = [
            "yes",           # retrieval decision
            "relevant",      # evaluate chunk
            "Good answer",   # generate
            "Good answer",   # critique returns same
        ]
        retriever = _mock_retriever(["context"])
        strategy = SelfRAG(llm=llm)

        result = strategy.run("Question", retriever, "m")

        assert result == "Good answer"
