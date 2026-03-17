"""Tests for RagasQueryGenerator.

All tests mock RAGAS and OpenAI — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.document import Document
from src.query_generators.ragas import RagasQueryGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake OPENAI_API_KEY for all tests in this module."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")


def _make_mock_testset(rows: list[dict]) -> MagicMock:
    """Create a mock RAGAS Testset that converts to a DataFrame.

    Args:
        rows: List of dicts representing testset rows.

    Returns:
        Mock object with to_pandas() returning a DataFrame.
    """
    mock_testset = MagicMock()
    mock_testset.to_pandas.return_value = pd.DataFrame(rows)
    return mock_testset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRagasQueryGenerator:
    """Tests for the RagasQueryGenerator class."""

    def test_name_format(self) -> None:
        """Name uses the generator model name."""
        gen = RagasQueryGenerator()
        assert gen.name == "ragas:gpt-4o-mini"

    def test_name_custom_model(self) -> None:
        """Custom generator model appears in the name."""
        gen = RagasQueryGenerator(generator_model="gpt-4")
        assert gen.name == "ragas:gpt-4"

    @patch("src.query_generators.ragas.TestsetGenerator")
    @patch("src.query_generators.ragas.OpenAIEmbeddings")
    @patch("src.query_generators.ragas.ChatOpenAI")
    def test_generate_returns_query_objects(
        self,
        mock_chat: MagicMock,
        mock_embeddings: MagicMock,
        mock_generator_cls: MagicMock,
    ) -> None:
        """generate() returns Query objects from mocked RAGAS output."""
        # Set up mock chain
        mock_gen_instance = MagicMock()
        mock_generator_cls.from_langchain.return_value = mock_gen_instance
        mock_gen_instance.generate_with_langchain_docs.return_value = _make_mock_testset([
            {"question": "Q1?", "evolution_type": "simple", "ground_truth": "A1",
             "contexts": ["ctx1"], "metadata": {"title": "Doc1"}},
            {"question": "Q2?", "evolution_type": "reasoning", "ground_truth": "A2",
             "contexts": ["ctx2"], "metadata": {"title": "Doc1"}},
            {"question": "Q3?", "evolution_type": "multi_context", "ground_truth": "A3",
             "contexts": ["ctx3"], "metadata": {"title": "Doc1"}},
        ])

        gen = RagasQueryGenerator()
        docs = [Document(title="Doc1", text="Some content about things.")]
        queries = gen.generate(docs, queries_per_doc=3)

        assert len(queries) == 3
        assert all(q.text for q in queries)

    @patch("src.query_generators.ragas.TestsetGenerator")
    @patch("src.query_generators.ragas.OpenAIEmbeddings")
    @patch("src.query_generators.ragas.ChatOpenAI")
    def test_generate_maps_evolution_types(
        self,
        mock_chat: MagicMock,
        mock_embeddings: MagicMock,
        mock_generator_cls: MagicMock,
    ) -> None:
        """RAGAS evolution types are mapped to our query taxonomy."""
        mock_gen_instance = MagicMock()
        mock_generator_cls.from_langchain.return_value = mock_gen_instance
        mock_gen_instance.generate_with_langchain_docs.return_value = _make_mock_testset([
            {"question": "Q1?", "evolution_type": "simple", "ground_truth": "A",
             "contexts": [], "metadata": {}},
            {"question": "Q2?", "evolution_type": "reasoning", "ground_truth": "A",
             "contexts": [], "metadata": {}},
            {"question": "Q3?", "evolution_type": "multi_context", "ground_truth": "A",
             "contexts": [], "metadata": {}},
        ])

        gen = RagasQueryGenerator()
        queries = gen.generate([Document(title="D", text="T")], queries_per_doc=3)

        assert queries[0].query_type == "factoid"
        assert queries[1].query_type == "reasoning"
        assert queries[2].query_type == "multi_context"

    @patch("src.query_generators.ragas.TestsetGenerator")
    @patch("src.query_generators.ragas.OpenAIEmbeddings")
    @patch("src.query_generators.ragas.ChatOpenAI")
    def test_generate_sets_generator_name(
        self,
        mock_chat: MagicMock,
        mock_embeddings: MagicMock,
        mock_generator_cls: MagicMock,
    ) -> None:
        """All returned queries have the generator's name."""
        mock_gen_instance = MagicMock()
        mock_generator_cls.from_langchain.return_value = mock_gen_instance
        mock_gen_instance.generate_with_langchain_docs.return_value = _make_mock_testset([
            {"question": "Q?", "evolution_type": "simple", "ground_truth": "A",
             "contexts": [], "metadata": {}},
        ])

        gen = RagasQueryGenerator()
        queries = gen.generate([Document(title="D", text="T")])

        assert all(q.generator_name == "ragas:gpt-4o-mini" for q in queries)

    @patch("src.query_generators.ragas.TestsetGenerator")
    @patch("src.query_generators.ragas.OpenAIEmbeddings")
    @patch("src.query_generators.ragas.ChatOpenAI")
    def test_generate_sets_source_doc_title(
        self,
        mock_chat: MagicMock,
        mock_embeddings: MagicMock,
        mock_generator_cls: MagicMock,
    ) -> None:
        """Returned queries have source_doc_title from metadata."""
        mock_gen_instance = MagicMock()
        mock_generator_cls.from_langchain.return_value = mock_gen_instance
        mock_gen_instance.generate_with_langchain_docs.return_value = _make_mock_testset([
            {"question": "Q?", "evolution_type": "simple", "ground_truth": "A",
             "contexts": ["ctx"], "metadata": {"title": "MyDoc"}},
        ])

        gen = RagasQueryGenerator()
        queries = gen.generate([Document(title="MyDoc", text="Content")])

        assert queries[0].source_doc_title == "MyDoc"

    def test_missing_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor raises ValueError when OPENAI_API_KEY is not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            RagasQueryGenerator()

    @patch("src.query_generators.ragas.TestsetGenerator")
    @patch("src.query_generators.ragas.OpenAIEmbeddings")
    @patch("src.query_generators.ragas.ChatOpenAI")
    def test_protocol_compliance(
        self,
        mock_chat: MagicMock,
        mock_embeddings: MagicMock,
        mock_generator_cls: MagicMock,
    ) -> None:
        """RagasQueryGenerator satisfies the QueryGenerator protocol."""
        from src.protocols import QueryGenerator

        gen = RagasQueryGenerator()
        assert isinstance(gen, QueryGenerator)
