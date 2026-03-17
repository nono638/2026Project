"""Tests for additional query generators: HumanQuerySet, BEIRQuerySet, TemplateQueryGenerator.

Covers CSV/JSON loading, BEIR directory parsing, spaCy-based template generation,
protocol compliance, and edge cases.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.document import Document
from src.protocols import QueryGenerator
from src.query import Query
from src.query_generators.human import HumanQuerySet
from src.query_generators.beir import BEIRQuerySet
from src.query_generators.template import TemplateQueryGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_csv(path: Path, rows: list[dict]) -> Path:
    """Helper: write a CSV file from a list of dicts."""
    filepath = path / "queries.csv"
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def _create_json(path: Path, data: list[dict]) -> Path:
    """Helper: write a JSON file from a list of dicts."""
    filepath = path / "queries.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filepath


SAMPLE_QUERIES = [
    {
        "text": "What is the capital of France?",
        "query_type": "factoid",
        "source_doc_title": "Paris",
    },
    {
        "text": "How does German reunification compare?",
        "query_type": "reasoning",
        "source_doc_title": "Berlin",
    },
    {
        "text": "What is the population of Tokyo?",
        "query_type": "factoid",
        "source_doc_title": "Tokyo",
    },
]


SAMPLE_DOCUMENTS = [
    Document(title="Paris", text="Paris is the capital of France."),
    Document(title="Berlin", text="Berlin is the capital of Germany."),
    Document(title="Tokyo", text="Tokyo is the capital of Japan."),
]


def create_beir_fixture(tmp_path: Path) -> Path:
    """Create a minimal BEIR dataset directory for testing."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # corpus.jsonl
    with open(dataset_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.write('{"_id": "d1", "title": "Paris", "text": "Paris is the capital of France."}\n')
        f.write('{"_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany."}\n')

    # queries.jsonl
    with open(dataset_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        f.write('{"_id": "q1", "text": "What is the capital of France?"}\n')
        f.write('{"_id": "q2", "text": "What is the capital of Germany?"}\n')

    # qrels/test.tsv
    (dataset_dir / "qrels").mkdir()
    with open(dataset_dir / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\td1\t1\n")
        f.write("q2\td2\t1\n")

    return dataset_dir


# ===========================================================================
# HumanQuerySet Tests
# ===========================================================================


class TestHumanQuerySet:
    """Tests for HumanQuerySet — CSV/JSON query loader."""

    def test_load_from_csv(self, tmp_path: Path) -> None:
        """Load queries from a CSV file and verify correct count and fields."""
        csv_path = _create_csv(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(csv_path)
        result = qs.generate([])
        assert len(result) == 3
        assert result[0].text == "What is the capital of France?"
        assert result[0].query_type == "factoid"
        assert result[0].source_doc_title == "Paris"

    def test_load_from_json(self, tmp_path: Path) -> None:
        """Load queries from a JSON file (same format as save_queries)."""
        json_path = _create_json(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(json_path)
        result = qs.generate([])
        assert len(result) == 3
        assert result[1].text == "How does German reunification compare?"

    def test_name_format(self, tmp_path: Path) -> None:
        """Name is 'human:<filename_stem>'."""
        csv_path = _create_csv(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(csv_path)
        assert qs.name == "human:queries"

    def test_generate_filters_by_documents(self, tmp_path: Path) -> None:
        """When documents are provided, only return queries for those docs."""
        csv_path = _create_csv(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(csv_path)
        docs = [Document(title="Paris", text="Paris info")]
        result = qs.generate(docs)
        assert len(result) == 1
        assert result[0].source_doc_title == "Paris"

    def test_generate_empty_documents_returns_all(self, tmp_path: Path) -> None:
        """generate([]) returns all queries."""
        csv_path = _create_csv(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(csv_path)
        result = qs.generate([])
        assert len(result) == 3

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """CSV missing 'text' column raises ValueError."""
        filepath = tmp_path / "bad.csv"
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_type", "source_doc_title"])
            writer.writeheader()
            writer.writerow({"query_type": "factoid", "source_doc_title": "Doc"})
        with pytest.raises(ValueError, match="missing required columns"):
            HumanQuerySet(filepath)

    def test_protocol_compliance(self, tmp_path: Path) -> None:
        """HumanQuerySet satisfies the QueryGenerator protocol."""
        csv_path = _create_csv(tmp_path, SAMPLE_QUERIES)
        qs = HumanQuerySet(csv_path)
        assert isinstance(qs, QueryGenerator)


# ===========================================================================
# BEIRQuerySet Tests
# ===========================================================================


class TestBEIRQuerySet:
    """Tests for BEIRQuerySet — BEIR benchmark dataset loader."""

    def test_load_beir_dataset(self, tmp_path: Path) -> None:
        """Load a BEIR dataset and verify query count and source mapping."""
        dataset_dir = create_beir_fixture(tmp_path)
        qs = BEIRQuerySet(dataset_dir)
        result = qs.generate([])
        assert len(result) == 2
        # Verify source_doc_title mapping through qrels
        titles = {q.source_doc_title for q in result}
        assert titles == {"Paris", "Berlin"}

    def test_name_format(self, tmp_path: Path) -> None:
        """Name is 'beir:<dir_name>'."""
        dataset_dir = create_beir_fixture(tmp_path)
        qs = BEIRQuerySet(dataset_dir)
        assert qs.name == "beir:test_dataset"

    def test_generate_filters_by_documents(self, tmp_path: Path) -> None:
        """When documents are provided, only return matching queries."""
        dataset_dir = create_beir_fixture(tmp_path)
        qs = BEIRQuerySet(dataset_dir)
        docs = [Document(title="Paris", text="Paris is the capital of France.")]
        result = qs.generate(docs)
        assert len(result) == 1
        assert result[0].source_doc_title == "Paris"

    def test_load_corpus(self, tmp_path: Path) -> None:
        """load_corpus() returns Documents with correct titles and beir_id metadata."""
        dataset_dir = create_beir_fixture(tmp_path)
        qs = BEIRQuerySet(dataset_dir)
        corpus = qs.load_corpus()
        assert len(corpus) == 2
        assert corpus[0].title == "Paris"
        assert corpus[0].metadata["beir_id"] == "d1"
        assert corpus[1].title == "Berlin"

    def test_missing_files(self, tmp_path: Path) -> None:
        """Directory without queries.jsonl raises FileNotFoundError."""
        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            BEIRQuerySet(empty_dir)

    def test_protocol_compliance(self, tmp_path: Path) -> None:
        """BEIRQuerySet satisfies the QueryGenerator protocol."""
        dataset_dir = create_beir_fixture(tmp_path)
        qs = BEIRQuerySet(dataset_dir)
        assert isinstance(qs, QueryGenerator)


# ===========================================================================
# TemplateQueryGenerator Tests
# ===========================================================================


class TestTemplateQueryGenerator:
    """Tests for TemplateQueryGenerator — spaCy NER + template filling."""

    def test_generate_returns_queries(self) -> None:
        """Generate queries from a document with known entities."""
        doc = Document(
            title="Albert Einstein",
            text="Albert Einstein was born in Ulm, Germany. "
            "He developed the theory of relativity at the Swiss Patent Office.",
        )
        gen = TemplateQueryGenerator()
        result = gen.generate([doc], queries_per_doc=5)
        assert len(result) > 0
        # All queries should have correct metadata
        for q in result:
            assert q.source_doc_title == "Albert Einstein"
            assert q.generator_name == "template:en_core_web_sm"
            assert q.query_type in ("factoid", "reasoning", "multi_context", "conditional")

    def test_query_types_distributed(self) -> None:
        """With queries_per_doc=10, check approximate type distribution."""
        doc = Document(
            title="World History",
            text="The United Nations was founded in New York City in 1945. "
            "The European Union was established in 1993 by the Maastricht Treaty. "
            "NATO was formed in Washington, D.C. in 1949.",
        )
        gen = TemplateQueryGenerator()
        result = gen.generate([doc], queries_per_doc=10)

        type_counts: dict[str, int] = {}
        for q in result:
            type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1

        # Approximate distribution: factoid ~4, reasoning ~3, multi_context ~2, conditional ~1
        assert type_counts.get("factoid", 0) >= 3
        assert type_counts.get("reasoning", 0) >= 2

    def test_entity_extraction(self) -> None:
        """Document with 'New York' and 'United Nations' — both appear in queries."""
        doc = Document(
            title="International Organizations",
            text="The United Nations headquarters is located in New York City. "
            "The United Nations was established to maintain international peace.",
        )
        gen = TemplateQueryGenerator()
        result = gen.generate([doc], queries_per_doc=5)

        all_text = " ".join(q.text for q in result)
        # At least one entity should appear in the generated queries
        assert "United Nations" in all_text or "New York" in all_text

    def test_document_no_entities(self) -> None:
        """Document with no recognizable entities generates 0 queries."""
        doc = Document(
            title="Simple Text",
            text="The quick brown fox jumps over the lazy dog.",
        )
        gen = TemplateQueryGenerator()
        result = gen.generate([doc], queries_per_doc=5)
        assert len(result) == 0

    def test_single_entity_skips_multi_context(self) -> None:
        """Document with only one entity has no multi_context queries."""
        # Use a very short text with a clear single entity
        doc = Document(
            title="Single Entity",
            text="Einstein was a genius.",
        )
        gen = TemplateQueryGenerator()
        result = gen.generate([doc], queries_per_doc=5)

        multi_context = [q for q in result if q.query_type == "multi_context"]
        conditional = [q for q in result if q.query_type == "conditional"]
        # With only one entity, multi_context and conditional are skipped
        assert len(multi_context) == 0
        assert len(conditional) == 0

    def test_name_format(self) -> None:
        """Name is 'template:en_core_web_sm'."""
        gen = TemplateQueryGenerator()
        assert gen.name == "template:en_core_web_sm"

    def test_custom_templates(self) -> None:
        """Custom templates are used instead of defaults."""
        custom = {
            "factoid": ["Tell me about {entity} in {topic}."],
        }
        gen = TemplateQueryGenerator(templates=custom)
        doc = Document(
            title="Test Topic",
            text="Albert Einstein revolutionized physics.",
        )
        result = gen.generate([doc], queries_per_doc=3)
        # All queries should use the custom template pattern
        for q in result:
            assert "Tell me about" in q.text

    def test_protocol_compliance(self) -> None:
        """TemplateQueryGenerator satisfies the QueryGenerator protocol."""
        gen = TemplateQueryGenerator()
        assert isinstance(gen, QueryGenerator)
