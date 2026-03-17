"""Query representation for the RAG evaluation pipeline.

Separate module because queries carry metadata about their generation,
type classification, and source provenance. Will grow to support
multimodal queries (image-based questions) in the future.
See architecture-decisions.md: "every pipeline stage gets its own protocol."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Query:
    """An evaluation query generated from or associated with a document.

    Attributes:
        text: The question text.
        query_type: Category — one of: factoid, reasoning, multi_context, conditional.
        source_doc_title: Title of the document this query was generated from.
        reference_answer: Optional gold-standard answer for evaluation.
        generator_name: Name of the QueryGenerator that created this query.
        metadata: Optional extensible metadata.
    """

    text: str
    query_type: str
    source_doc_title: str
    reference_answer: str | None = None
    generator_name: str | None = None
    metadata: dict | None = field(default=None)


def save_queries(queries: list[Query], path: str | Path) -> None:
    """Save queries to a JSON file for reproducibility.

    Queries are generated once and frozen — this serializes them for reuse
    across experiment runs without re-generating.

    Args:
        queries: List of Query objects to serialize.
        path: Output file path (JSON).
    """
    data = [asdict(q) for q in queries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_queries(path: str | Path) -> list[Query]:
    """Load queries from a JSON file.

    Validates that required fields are present — raises ValueError if any
    query dict is missing text, query_type, or source_doc_title.

    Args:
        path: Path to a JSON file containing a list of query dicts.

    Returns:
        List of Query objects.

    Raises:
        ValueError: If a query dict is missing required fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = {"text", "query_type", "source_doc_title"}
    queries: list[Query] = []
    for i, item in enumerate(data):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"Query at index {i} is missing required fields: {missing}"
            )
        queries.append(
            Query(
                text=item["text"],
                query_type=item["query_type"],
                source_doc_title=item["source_doc_title"],
                reference_answer=item.get("reference_answer"),
                generator_name=item.get("generator_name"),
                metadata=item.get("metadata"),
            )
        )

    return queries


def queries_to_dicts(queries: list[Query]) -> list[dict]:
    """Convert Query objects to the dict format Experiment.load_corpus expects.

    Bridge helper so the new Query type works with the existing experiment
    runner without modifying experiment.py.

    Args:
        queries: List of Query objects.

    Returns:
        List of dicts with 'text' and 'type' keys.
    """
    return [{"text": q.text, "type": q.query_type} for q in queries]
