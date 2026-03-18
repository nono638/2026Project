"""Tests for SQuAD 2.0 dataset loader.

Mocks HuggingFace datasets.load_dataset to avoid network dependency.
Follows the same structure as test_hotpotqa_loader.py.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.document import Document, documents_to_dicts
from src.query import Query


# ---------------------------------------------------------------------------
# Mock data matching SQuAD 2.0 schema
# ---------------------------------------------------------------------------

def _make_example(
    qid: str = "56be85543aeaaa14008c9063",
    title: str = "Beyonce",
    context: str = "Beyonce Giselle Knowles-Carter is an American singer.",
    question: str = "What is Beyonce's full name?",
    answer_text: list[str] | None = None,
    answer_start: list[int] | None = None,
) -> dict:
    """Build a mock SQuAD 2.0 example matching the real schema."""
    if answer_text is None:
        answer_text = ["Beyonce Giselle Knowles-Carter"]
    if answer_start is None:
        answer_start = [0]

    return {
        "id": qid,
        "title": title,
        "context": context,
        "question": question,
        "answers": {
            "text": answer_text,
            "answer_start": answer_start,
        },
    }


def _make_unanswerable(
    qid: str = "unanswerable_001",
    title: str = "Beyonce",
    context: str = "Beyonce Giselle Knowles-Carter is an American singer.",
    question: str = "When did Beyonce start acting in horror movies?",
) -> dict:
    """Build a mock unanswerable SQuAD 2.0 example (empty answers)."""
    return {
        "id": qid,
        "title": title,
        "context": context,
        "question": question,
        "answers": {
            "text": [],
            "answer_start": [],
        },
    }


def _make_mock_dataset(examples: list[dict]) -> MagicMock:
    """Create a mock HuggingFace Dataset from a list of example dicts."""
    dataset = MagicMock()
    dataset.__len__ = lambda self: len(examples)
    dataset.__iter__ = lambda self: iter(examples)

    def column_access(key):
        return [ex[key] for ex in examples]

    dataset.__getitem__ = lambda self, key: (
        examples[key] if isinstance(key, int) else column_access(key)
    )

    return dataset


# Six answerable examples across 3 articles for stratified sampling tests
_ANSWERABLE_EXAMPLES = [
    _make_example("id1", "Beyonce", "Beyonce is a singer.", "Who is Beyonce?", ["a singer"], [13]),
    _make_example("id2", "Beyonce", "She was born in Houston.", "Where was Beyonce born?", ["Houston"], [17]),
    _make_example("id3", "Oxygen", "Oxygen is a chemical element.", "What is oxygen?", ["a chemical element"], [10]),
    _make_example("id4", "Oxygen", "It has atomic number 8.", "What is oxygen's atomic number?", ["8"], [22]),
    _make_example("id5", "Geology", "Geology is the study of Earth.", "What is geology?", ["the study of Earth"], [12]),
    _make_example("id6", "Geology", "Rocks are classified by process.", "How are rocks classified?", ["by process"], [25]),
]


def _mock_load_dataset(name, **kwargs):
    """Mock datasets.load_dataset that returns a dict with train/validation splits."""
    mock_ds = _make_mock_dataset(_ANSWERABLE_EXAMPLES)
    return {"train": mock_ds, "validation": mock_ds}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadSquad:
    """Test the load_squad function."""

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_returns_documents_and_queries(self, mock_load):
        from src.datasets.squad import load_squad
        docs, queries = load_squad()
        assert len(docs) > 0
        assert len(queries) > 0
        assert len(docs) == len(queries)  # Parallel lists

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_documents_are_document_objects(self, mock_load):
        from src.datasets.squad import load_squad
        docs, _ = load_squad()
        assert all(isinstance(d, Document) for d in docs)

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_are_query_objects(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        assert all(isinstance(q, Query) for q in queries)

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_have_reference_answers(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        for q in queries:
            assert q.reference_answer is not None
            assert len(q.reference_answer) > 0

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_query_type_is_factoid(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        for q in queries:
            assert q.query_type == "factoid"

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_generator_name_is_squad(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        for q in queries:
            assert q.generator_name == "squad"

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_query_metadata_has_required_fields(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        for q in queries:
            assert q.metadata is not None
            assert "article_title" in q.metadata
            assert "num_answer_spans" in q.metadata
            assert isinstance(q.metadata["num_answer_spans"], int)

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_metadata_has_required_fields(self, mock_load):
        from src.datasets.squad import load_squad
        docs, _ = load_squad()
        for d in docs:
            assert d.metadata is not None
            assert "article_title" in d.metadata


class TestSquadDocumentFormat:
    """Test that documents have the right format."""

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_text_matches_context(self, mock_load):
        from src.datasets.squad import load_squad
        docs, _ = load_squad()
        # First example's context is "Beyonce is a singer."
        assert docs[0].text == "Beyonce is a singer."

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_document_title_format(self, mock_load):
        from src.datasets.squad import load_squad
        docs, _ = load_squad()
        assert docs[0].title == "squad:id1"
        assert docs[2].title == "squad:id3"

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_query_source_matches_document(self, mock_load):
        from src.datasets.squad import load_squad
        docs, queries = load_squad()
        for d, q in zip(docs, queries):
            assert q.source_doc_title == d.title


class TestSampleSquad:
    """Test stratified sampling by article title."""

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_returns_requested_count(self, mock_load):
        from src.datasets.squad import load_squad, sample_squad
        docs, queries = load_squad()
        s_docs, s_queries = sample_squad(docs, queries, n=4, seed=42)
        assert len(s_docs) == 4
        assert len(s_queries) == 4

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_is_deterministic(self, mock_load):
        from src.datasets.squad import load_squad, sample_squad
        docs, queries = load_squad()
        s1_docs, _ = sample_squad(docs, queries, n=4, seed=42)
        s2_docs, _ = sample_squad(docs, queries, n=4, seed=42)
        assert [d.title for d in s1_docs] == [d.title for d in s2_docs]

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_n_exceeds_available(self, mock_load):
        from src.datasets.squad import load_squad, sample_squad
        docs, queries = load_squad()
        s_docs, s_queries = sample_squad(docs, queries, n=10000, seed=42)
        assert len(s_docs) == len(docs)

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_includes_multiple_articles(self, mock_load):
        from src.datasets.squad import load_squad, sample_squad
        docs, queries = load_squad()
        # With 6 examples across 3 articles, sample of 4 should hit multiple articles
        s_docs, s_queries = sample_squad(docs, queries, n=4, seed=42)
        articles = {q.metadata["article_title"] for q in s_queries}
        assert len(articles) >= 2

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_sample_zero_returns_empty(self, mock_load):
        from src.datasets.squad import load_squad, sample_squad
        docs, queries = load_squad()
        s_docs, s_queries = sample_squad(docs, queries, n=0, seed=42)
        assert len(s_docs) == 0
        assert len(s_queries) == 0


class TestSquadEdgeCases:
    """Test edge case handling."""

    def test_unanswerable_skipped(self):
        """Unanswerable questions (empty answers) should be skipped."""
        examples = [
            _make_example("good", question="Answerable?", answer_text=["yes"], answer_start=[0]),
            _make_unanswerable("bad"),
        ]
        mock_ds = _make_mock_dataset(examples)

        with patch("src.datasets.squad.hf_load_dataset",
                    return_value={"train": mock_ds}):
            from src.datasets.squad import load_squad
            docs, queries = load_squad()
            assert len(docs) == 1
            assert queries[0].reference_answer == "yes"

    def test_empty_context_skipped(self):
        """Examples with empty context should be skipped."""
        examples = [
            _make_example("good", context="Real context.", answer_text=["Real"], answer_start=[0]),
            _make_example("bad", context="", answer_text=["something"], answer_start=[0]),
        ]
        mock_ds = _make_mock_dataset(examples)

        with patch("src.datasets.squad.hf_load_dataset",
                    return_value={"train": mock_ds}):
            from src.datasets.squad import load_squad
            docs, queries = load_squad()
            assert len(docs) == 1
            assert docs[0].text == "Real context."

    def test_multiple_answer_spans_uses_first(self):
        """When multiple answer spans exist, use the first one."""
        examples = [
            _make_example(
                "multi",
                context="The cat sat on the mat. The cat was happy.",
                question="What animal is mentioned?",
                answer_text=["cat", "cat"],
                answer_start=[4, 28],
            ),
        ]
        mock_ds = _make_mock_dataset(examples)

        with patch("src.datasets.squad.hf_load_dataset",
                    return_value={"train": mock_ds}):
            from src.datasets.squad import load_squad
            _, queries = load_squad()
            assert queries[0].reference_answer == "cat"
            assert queries[0].metadata["num_answer_spans"] == 2


class TestSquadCompatibility:
    """Test that output works with existing pipeline helpers."""

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_documents_to_dicts_works(self, mock_load):
        from src.datasets.squad import load_squad
        docs, _ = load_squad()
        dicts = documents_to_dicts(docs)
        assert all("title" in d and "text" in d for d in dicts)

    @patch("src.datasets.squad.hf_load_dataset", side_effect=_mock_load_dataset)
    def test_queries_have_expected_fields_for_experiment(self, mock_load):
        from src.datasets.squad import load_squad
        _, queries = load_squad()
        for q in queries:
            assert hasattr(q, "text")
            assert hasattr(q, "query_type")
