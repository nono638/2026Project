"""Human-curated query set loader.

Wraps a CSV or JSON file of hand-written queries in the QueryGenerator
interface. The 'generate' method is a misnomer here — it loads, not generates —
but conforming to the protocol lets human queries plug into the same pipeline
as synthetic generators.

Used as a validation anchor: run the same experiment with synthetic and human
queries, check if they rank RAG configurations the same way.
Ref: ARES (arxiv:2311.09476)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.document import Document
from src.query import Query


class HumanQuerySet:
    """Loads hand-curated queries from CSV or JSON files.

    Conforms to the QueryGenerator protocol so human queries can be used
    interchangeably with synthetic generators in the experiment pipeline.

    Args:
        path: Path to a CSV or JSON file containing queries.

    Raises:
        ValueError: If required fields (text, query_type, source_doc_title)
            are missing from the input file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._queries: list[Query] = self._load()

    @property
    def name(self) -> str:
        """Unique identifier based on filename stem (e.g., 'human:validation_queries')."""
        return f"human:{self._path.stem}"

    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
        """Return stored queries, optionally filtered to match provided documents.

        Args:
            documents: If non-empty, filter to queries whose source_doc_title
                matches a title in this list. If empty, return all queries.
            queries_per_doc: Ignored — the query set is fixed.

        Returns:
            List of Query objects with generator_name set to self.name.
        """
        # Set generator_name on all returned queries
        queries = [
            Query(
                text=q.text,
                query_type=q.query_type,
                source_doc_title=q.source_doc_title,
                reference_answer=q.reference_answer,
                generator_name=self.name,
                metadata=q.metadata,
            )
            for q in self._queries
        ]

        # Filter by provided documents if non-empty
        if documents:
            doc_titles = {d.title for d in documents}
            queries = [q for q in queries if q.source_doc_title in doc_titles]

        return queries

    def _load(self) -> list[Query]:
        """Load queries from CSV or JSON based on file extension.

        Returns:
            List of Query objects.

        Raises:
            ValueError: If file extension is unsupported or required fields missing.
        """
        ext = self._path.suffix.lower()
        if ext == ".csv":
            return self._load_csv()
        elif ext == ".json":
            return self._load_json()
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use .csv or .json."
            )

    def _load_csv(self) -> list[Query]:
        """Load queries from a CSV file.

        Expected columns: text, query_type, source_doc_title,
        and optionally reference_answer.
        """
        required = {"text", "query_type", "source_doc_title"}
        queries: list[Query] = []

        with open(self._path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {self._path} has no headers")

            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV file is missing required columns: {missing}"
                )

            for row in reader:
                queries.append(
                    Query(
                        text=row["text"],
                        query_type=row["query_type"],
                        source_doc_title=row["source_doc_title"],
                        reference_answer=row.get("reference_answer"),
                    )
                )

        return queries

    def _load_json(self) -> list[Query]:
        """Load queries from a JSON file (same format as save_queries output)."""
        required = {"text", "query_type", "source_doc_title"}

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: list[Query] = []
        for i, item in enumerate(data):
            missing = required - set(item.keys())
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
