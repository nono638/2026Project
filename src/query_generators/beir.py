"""BEIR benchmark query set loader.

Loads queries from BEIR-format datasets (https://github.com/beir-cellar/beir).
BEIR datasets use a standard directory structure:
  dataset_name/
    corpus.jsonl    — documents (id, title, text)
    queries.jsonl   — queries (id, text)
    qrels/
      test.tsv      — relevance judgments (query_id, corpus_id, score)

This loader reads the queries and maps them to corpus documents via qrels.
The corpus itself can be loaded separately via Document loaders.

Ref: BEIR benchmark (arxiv:2104.08663)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.document import Document
from src.query import Query


class BEIRQuerySet:
    """Loads queries from a BEIR-format benchmark dataset directory.

    Conforms to the QueryGenerator protocol so BEIR benchmark queries can be
    used interchangeably with synthetic generators in the experiment pipeline.

    Args:
        dataset_dir: Path to the BEIR dataset directory.
        split: Which qrels split to use (default "test").
        query_type: Default query type to assign. BEIR doesn't have query type
            metadata, so this is assigned uniformly. Use "factoid" for most
            datasets, "multi_context" for HotpotQA-style datasets.

    Raises:
        FileNotFoundError: If required files (queries.jsonl, corpus.jsonl,
            qrels/{split}.tsv) are missing.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "test",
        query_type: str = "factoid",
    ) -> None:
        self._dataset_dir = Path(dataset_dir)
        self._split = split
        self._query_type = query_type
        self._queries: list[Query] = self._load()

    @property
    def name(self) -> str:
        """Unique identifier based on dataset directory name (e.g., 'beir:nfcorpus')."""
        return f"beir:{self._dataset_dir.name}"

    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
        """Return loaded queries, optionally filtered to match provided documents.

        Args:
            documents: If non-empty, filter to queries whose source_doc_title
                matches a title in this list. If empty, return all queries.
            queries_per_doc: Ignored — the query set is fixed.

        Returns:
            List of Query objects with generator_name set to self.name.
        """
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

        if documents:
            doc_titles = {d.title for d in documents}
            queries = [q for q in queries if q.source_doc_title in doc_titles]

        return queries

    def load_corpus(self) -> list[Document]:
        """Load the BEIR corpus as Document objects.

        Convenience method so users can load both queries and documents from the
        same BEIR dataset directory.

        Returns:
            List of Document objects with beir_id in metadata.

        Raises:
            FileNotFoundError: If corpus.jsonl is missing.
        """
        corpus_path = self._dataset_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

        documents: list[Document] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                documents.append(
                    Document(
                        title=entry.get("title", ""),
                        text=entry.get("text", ""),
                        metadata={"beir_id": entry["_id"]},
                    )
                )

        return documents

    def _load(self) -> list[Query]:
        """Load queries, corpus titles, and qrels, then build Query objects.

        Maps each query to its highest-relevance corpus document via qrels.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        queries_path = self._dataset_dir / "queries.jsonl"
        corpus_path = self._dataset_dir / "corpus.jsonl"
        qrels_path = self._dataset_dir / "qrels" / f"{self._split}.tsv"

        for path, desc in [
            (queries_path, "queries.jsonl"),
            (corpus_path, "corpus.jsonl"),
            (qrels_path, f"qrels/{self._split}.tsv"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required BEIR file: {desc} at {path}")

        # Load corpus id→title mapping
        corpus_titles: dict[str, str] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                corpus_titles[entry["_id"]] = entry.get("title", "")

        # Load query id→text mapping
        query_texts: dict[str, str] = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                query_texts[entry["_id"]] = entry["text"]

        # Load qrels: map query_id → highest-relevance corpus_id
        # qrels format: query-id \t corpus-id \t score (with header line)
        query_to_doc: dict[str, str] = {}
        query_to_score: dict[str, int] = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or i == 0:
                    # Skip header line
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                qid, cid, score_str = parts[0], parts[1], parts[2]
                score = int(score_str)
                # Keep the highest-relevance doc for each query
                if qid not in query_to_doc or score > query_to_score[qid]:
                    query_to_doc[qid] = cid
                    query_to_score[qid] = score

        # Build Query objects
        queries: list[Query] = []
        for qid, text in query_texts.items():
            if qid not in query_to_doc:
                continue  # No relevance judgment for this query
            corpus_id = query_to_doc[qid]
            doc_title = corpus_titles.get(corpus_id, "")
            queries.append(
                Query(
                    text=text,
                    query_type=self._query_type,
                    source_doc_title=doc_title,
                )
            )

        return queries
