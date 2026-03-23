"""Tests for src/diagnostics.py — failure attribution logic.

These tests verify detect_failure_stage() correctly identifies which pipeline
stage caused a RAG failure by checking gold answer presence at each stage.
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics import detect_failure_stage


class TestDetectFailureStageNone:
    """Gold answer found in RAG answer → no failure."""

    def test_exact_match(self):
        result = detect_failure_stage(
            gold_answer="Paris",
            rag_answer="The capital of France is Paris.",
            all_chunks=["Paris is the capital of France."],
            retrieved_chunk_texts=["Paris is the capital of France."],
            context_sent_to_llm="Paris is the capital of France.",
        )
        assert result == "none"

    def test_case_insensitive_match(self):
        result = detect_failure_stage(
            gold_answer="paris",
            rag_answer="The answer is PARIS.",
            all_chunks=["Paris info"],
            retrieved_chunk_texts=["Paris info"],
            context_sent_to_llm="Paris info",
        )
        assert result == "none"


class TestDetectFailureStageChunker:
    """Gold answer not in any chunk → chunker lost the information."""

    def test_gold_not_in_any_chunk(self):
        result = detect_failure_stage(
            gold_answer="887",
            rag_answer="The area is approximately 500 acres.",
            all_chunks=[
                "Vilnius is a city in Lithuania.",
                "The old town has many churches.",
                "Tourism is a major industry.",
            ],
            retrieved_chunk_texts=["Vilnius is a city in Lithuania."],
            context_sent_to_llm="Vilnius is a city in Lithuania.",
        )
        assert result == "chunker"


class TestDetectFailureStageRetrieval:
    """Gold answer in chunks but not in retrieved chunks → retrieval failure."""

    def test_gold_in_chunks_not_retrieved(self):
        result = detect_failure_stage(
            gold_answer="887",
            rag_answer="The area is unknown.",
            all_chunks=[
                "Vilnius is a city.",
                "The old town covers 887 acres.",  # Gold is here
                "Tourism is important.",
                "Many churches exist.",
                "The river flows through.",
            ],
            retrieved_chunk_texts=[
                "Vilnius is a city.",
                "Tourism is important.",
                "Many churches exist.",
            ],
            context_sent_to_llm="Vilnius is a city.\n\nTourism is important.\n\nMany churches exist.",
        )
        assert result == "retrieval"


class TestDetectFailureStageFiltering:
    """Gold answer in retrieved chunks but not in context → strategy filtered it out."""

    def test_gold_filtered_by_strategy(self):
        result = detect_failure_stage(
            gold_answer="887",
            rag_answer="The area is not mentioned.",
            all_chunks=[
                "Vilnius is a city.",
                "The old town covers 887 acres.",
            ],
            retrieved_chunk_texts=[
                "Vilnius is a city.",
                "The old town covers 887 acres.",
            ],
            # Strategy filtered out the chunk with 887
            context_sent_to_llm="Vilnius is a city.",
        )
        assert result == "filtering"


class TestDetectFailureStageGeneration:
    """Gold answer in context but not in RAG answer → model failed to use it."""

    def test_gold_in_context_not_in_answer(self):
        result = detect_failure_stage(
            gold_answer="887",
            rag_answer="The old town is a historic area with many notable buildings.",
            all_chunks=["The old town covers 887 acres."],
            retrieved_chunk_texts=["The old town covers 887 acres."],
            context_sent_to_llm="The old town covers 887 acres.",
        )
        assert result == "generation"


class TestDetectFailureStageNoRetrieval:
    """Strategy skipped retrieval entirely."""

    def test_skipped_retrieval(self):
        result = detect_failure_stage(
            gold_answer="Paris",
            rag_answer="I think the capital is London.",
            all_chunks=["Paris is the capital of France."],
            retrieved_chunk_texts=[],
            context_sent_to_llm="",
            skipped_retrieval=True,
        )
        assert result == "no_retrieval"

    def test_skipped_retrieval_but_correct(self):
        """Even if retrieval was skipped, correct answer = 'none'."""
        result = detect_failure_stage(
            gold_answer="Paris",
            rag_answer="The capital of France is Paris.",
            all_chunks=[],
            retrieved_chunk_texts=[],
            context_sent_to_llm="",
            skipped_retrieval=True,
        )
        assert result == "none"


class TestDetectFailureStageUnknown:
    """No gold answer → can't determine failure stage."""

    def test_empty_gold(self):
        result = detect_failure_stage(
            gold_answer="",
            rag_answer="Some answer.",
            all_chunks=["Some chunk."],
            retrieved_chunk_texts=["Some chunk."],
            context_sent_to_llm="Some chunk.",
        )
        assert result == "unknown"

    def test_none_gold(self):
        result = detect_failure_stage(
            gold_answer=None,
            rag_answer="Some answer.",
            all_chunks=["Some chunk."],
            retrieved_chunk_texts=["Some chunk."],
            context_sent_to_llm="Some chunk.",
        )
        assert result == "unknown"


class TestDetectFailureStageCaseInsensitivity:
    """All comparisons should be case-insensitive."""

    def test_gold_uppercase_in_lowercase_chunks(self):
        result = detect_failure_stage(
            gold_answer="PARIS",
            rag_answer="Not found.",
            all_chunks=["paris is the capital."],
            retrieved_chunk_texts=["paris is the capital."],
            context_sent_to_llm="paris is the capital.",
        )
        assert result == "generation"

    def test_gold_mixed_case(self):
        result = detect_failure_stage(
            gold_answer="Albert Einstein",
            rag_answer="The physicist was albert einstein.",
            all_chunks=["ALBERT EINSTEIN was born in 1879."],
            retrieved_chunk_texts=["ALBERT EINSTEIN was born in 1879."],
            context_sent_to_llm="ALBERT EINSTEIN was born in 1879.",
        )
        assert result == "none"
