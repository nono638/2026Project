"""Tests for HotpotQA dataset loader.

Mocks HuggingFace datasets.load_dataset to avoid network dependency.
"""

from __future__ import annotations

import random
from unittest.mock import patch, MagicMock

import pytest

from src.document import Document, documents_to_dicts
from src.query import Query


# ---------------------------------------------------------------------------
# Mock data matching HotpotQA "distractor" schema
# ---------------------------------------------------------------------------

def _make_example(
    qid: str = "abc123",
    question: str = "Which magazine was started first?",
    answer: str = "Arthur's Magazine",
    qtype: str = "comparison",
    level: str = "medium",
    num_passages: int = 10,
) -> dict:
    """Build a mock HotpotQA example matching the real schema."""
    titles = [f"Passage {i}" for i in range(num_passages)]
    sentences = [
        [f"Sentence {j} of passage {i}." for j in range(3)]
        for i in range(num_passages)
    ]
    # Supporting facts: first two passages
    supporting_titles = titles[:2]
    supporting_sent_ids = [0, 0]

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": level,
        "supporting_facts": {
            "title": supporting_titles,
            "sent_id": supporting_sent_ids,
        },
        "context": {
            "title": titles,
            "sentences": sentences,
        },
    }


def _make_mock_dataset(examples: list[dict]) -> MagicMock:
    """Create a mock HuggingFace Dataset from a list of example dicts."""
    dataset = MagicMock()
    dataset.__len__ = lambda self: len(examples)
    dataset.__getitem__ = lambda self, idx: examples[idx]
    dataset.__iter__ = lambda self: iter(examples)

    # Support column access like dataset["question"]
    def column_access(key):
        return [ex[key] for ex in examples]
    dataset.__getitem__ = lambda self, key: (
        examples[key] if isinstance(key, int) else column_access(key)
    )

    return dataset


def _mock_load_dataset(name, config, **kwargs):
    """Mock datasets.load_dataset that returns a dict with a 'train' split."""
    examples = [
        _make_example("id1", "Question about bridges?", "Bridge Answer", "bridge", "easy"),
        _make_example("id2", "Compare X and Y?", "X", "comparison", "medium"),
        _make_example("id3", "Hard bridge question?", "Complex Answer", "bridge", "hard"),
        _make_example("id4", "Easy comparison?", "Simple", "comparison", "easy"),
        _make_example("id5", "Medium bridge?", "Middle", "bridge", "medium"),
        _make_example("id6", "Hard comparison?", "Difficult", "comparison", "hard"),
    ]
    mock_ds = _make_mock_dataset(examples)
    return {"train": mock_ds}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadHotpotqa:
    """Test the load_hotpotqa function."""

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_returns_documents_and_queries(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, queries = load_hotpotqa()
        assert len(docs) > 0
        assert len(queries) > 0
        assert len(docs) == len(queries)  # One doc per query

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_documents_are_document_objects(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        assert all(isinstance(d, Document) for d in docs)

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_are_query_objects(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        _, queries = load_hotpotqa()
        assert all(isinstance(q, Query) for q in queries)

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_have_reference_answers(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        _, queries = load_hotpotqa()
        for q in queries:
            assert q.reference_answer is not None
            assert len(q.reference_answer) > 0

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_query_metadata_has_required_fields(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        _, queries = load_hotpotqa()
        for q in queries:
            assert q.metadata is not None
            assert "difficulty" in q.metadata
            assert "question_type" in q.metadata
            assert "supporting_titles" in q.metadata
            assert q.metadata["difficulty"] in ("easy", "medium", "hard")
            assert q.metadata["question_type"] in ("bridge", "comparison")

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_metadata_has_required_fields(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        for d in docs:
            assert d.metadata is not None
            assert "passage_titles" in d.metadata
            assert "num_passages" in d.metadata
            assert isinstance(d.metadata["passage_titles"], list)


class TestHotpotqaDocumentFormat:
    """Test that concatenated documents have the right format."""

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_text_contains_passage_headers(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        # Each passage should have a markdown header
        assert "## Passage 0" in docs[0].text
        assert "## Passage 1" in docs[0].text

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_text_contains_separators(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        assert "---" in docs[0].text

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_text_contains_sentences(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        assert "Sentence 0 of passage 0." in docs[0].text


class TestSampleHotpotqa:
    """Test stratified sampling."""

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_returns_requested_count(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
        docs, queries = load_hotpotqa()
        s_docs, s_queries = sample_hotpotqa(docs, queries, n=4, seed=42)
        assert len(s_docs) == 4
        assert len(s_queries) == 4

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_is_deterministic(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
        docs, queries = load_hotpotqa()
        s1_docs, s1_queries = sample_hotpotqa(docs, queries, n=4, seed=42)
        s2_docs, s2_queries = sample_hotpotqa(docs, queries, n=4, seed=42)
        assert [d.title for d in s1_docs] == [d.title for d in s2_docs]

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_n_exceeds_available(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
        docs, queries = load_hotpotqa()
        s_docs, s_queries = sample_hotpotqa(docs, queries, n=10000, seed=42)
        assert len(s_docs) == len(docs)

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_includes_both_types(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa, sample_hotpotqa
        docs, queries = load_hotpotqa()
        s_docs, s_queries = sample_hotpotqa(docs, queries, n=4, seed=42)
        types = {q.metadata["question_type"] for q in s_queries}
        # With 6 examples (3 bridge + 3 comparison), a sample of 4 should include both
        assert "bridge" in types
        assert "comparison" in types


class TestHotpotqaCompatibility:
    """Test that output works with existing pipeline helpers."""

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_documents_to_dicts_works(self, mock_load):
        from src.datasets.hotpotqa import load_hotpotqa
        docs, _ = load_hotpotqa()
        dicts = documents_to_dicts(docs)
        assert all("title" in d and "text" in d for d in dicts)

    @patch("src.datasets.hotpotqa.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_have_expected_fields_for_experiment(self, mock_load):
        """Queries can be converted to dicts for Experiment.load_corpus."""
        from src.datasets.hotpotqa import load_hotpotqa
        _, queries = load_hotpotqa()
        for q in queries:
            # Experiment expects dicts with 'text' and 'type' keys
            assert hasattr(q, "text")
            assert hasattr(q, "query_type")


class TestHotpotqaEdgeCases:
    """Test edge case handling."""

    def test_empty_passage_skipped(self):
        """Passages with empty text should be skipped in concatenation."""
        example = _make_example()
        # Make one passage have empty sentences
        example["context"]["sentences"][3] = []

        examples = [example]
        mock_ds = _make_mock_dataset(examples)

        with patch("src.datasets.hotpotqa.hf_load_dataset",
                    return_value={"train": mock_ds}):
            from src.datasets.hotpotqa import load_hotpotqa
            docs, _ = load_hotpotqa()
            # Document should still be created, just without the empty passage
            assert len(docs) == 1
            assert "Passage 3" not in docs[0].text

    def test_empty_answer_skipped(self):
        """Examples with empty answers should be skipped entirely."""
        examples = [
            _make_example("good", answer="Real Answer"),
            _make_example("bad", answer=""),
        ]
        mock_ds = _make_mock_dataset(examples)

        with patch("src.datasets.hotpotqa.hf_load_dataset",
                    return_value={"train": mock_ds}):
            from src.datasets.hotpotqa import load_hotpotqa
            docs, queries = load_hotpotqa()
            assert len(docs) == 1
            assert queries[0].reference_answer == "Real Answer"
