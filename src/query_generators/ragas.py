"""RAGAS-based query generation for RAG evaluation.

Uses the RAGAS TestsetGenerator to produce evaluation queries from documents,
with evolution for difficulty diversity. RAGAS is a widely cited framework
(arxiv:2309.15217) which gives the methodology academic credibility.

This is one implementation of the QueryGenerator protocol — the experiment
pipeline doesn't depend on RAGAS specifically.

Uses RAGAS 0.4.3 API: TestsetGenerator.from_langchain() with langchain
LLMs and embeddings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator

from src.query import Query

if TYPE_CHECKING:
    from src.document import Document

# RAGAS evolution type → our query taxonomy mapping.
# RAGAS uses different naming across versions; this mapping normalizes them.
_EVOLUTION_TYPE_MAP: dict[str, str] = {
    "simple": "factoid",
    "reasoning": "reasoning",
    "multi_context": "multi_context",
    "conditional": "conditional",
}


class RagasQueryGenerator:
    """Query generation via RAGAS TestsetGenerator (wraps OpenAI LLMs).

    Implements the QueryGenerator protocol from src.protocols. Uses RAGAS's
    TestsetGenerator which generates seed questions from document chunks and
    evolves them into difficulty tiers (factoid, reasoning, multi_context,
    conditional).

    Requires OPENAI_API_KEY environment variable — RAGAS uses OpenAI by default.
    """

    def __init__(
        self,
        generator_model: str = "gpt-4o-mini",
        critic_model: str = "gpt-4o-mini",
        distribution: dict[str, float] | None = None,
    ) -> None:
        """Initialize with OpenAI model names and query type distribution.

        Args:
            generator_model: OpenAI model for generating seed questions.
            critic_model: OpenAI model for evolution/critique.
            distribution: Proportions for each query type. Defaults to
                a balanced mix of simple, reasoning, multi_context, conditional.

        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Set OPENAI_API_KEY environment variable for RAGAS query "
                "generation. RAGAS uses OpenAI models by default."
            )

        self._generator_model = generator_model
        self._critic_model = critic_model
        self._distribution = distribution or {
            "simple": 0.3,
            "reasoning": 0.3,
            "multi_context": 0.25,
            "conditional": 0.15,
        }

    @property
    def name(self) -> str:
        """Return unique identifier including the generator model name."""
        return f"ragas:{self._generator_model}"

    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
        """Generate evaluation queries from documents using RAGAS.

        Converts Documents to langchain format, runs RAGAS TestsetGenerator,
        and maps results back to our Query dataclass.

        Args:
            documents: List of Document objects to generate queries from.
            queries_per_doc: Target queries per document (total = len(docs) * queries_per_doc).

        Returns:
            List of Query objects with type, source, and reference answer metadata.
        """
        from langchain_core.documents import Document as LCDocument

        # Convert our Documents to langchain format
        lc_docs = [
            LCDocument(
                page_content=doc.text,
                metadata={"title": doc.title},
            )
            for doc in documents
        ]

        # Build RAGAS generator using langchain LLMs
        # RAGAS 0.4.3 uses from_langchain(llm, embedding_model)
        generator_llm = ChatOpenAI(model=self._generator_model)
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            llm=generator_llm,
            embedding_model=embeddings,
        )

        test_size = len(documents) * queries_per_doc

        # Generate the testset
        testset = generator.generate_with_langchain_docs(
            documents=lc_docs,
            testset_size=test_size,
        )

        # Convert RAGAS testset to our Query objects
        df = testset.to_pandas()
        queries: list[Query] = []
        for _, row in df.iterrows():
            # Map RAGAS evolution type to our taxonomy
            evolution_type = str(row.get("evolution_type", "simple")).lower()
            query_type = _EVOLUTION_TYPE_MAP.get(evolution_type, "factoid")

            # Extract source document title from metadata
            source_title = ""
            source_contexts = row.get("contexts", [])
            if source_contexts and isinstance(source_contexts, list):
                # Try to find the source document title from metadata
                source_title = str(row.get("metadata", {}).get("title", ""))

            queries.append(
                Query(
                    text=str(row.get("question", row.get("user_input", ""))),
                    query_type=query_type,
                    source_doc_title=source_title,
                    reference_answer=str(row.get("ground_truth", row.get("reference", ""))) or None,
                    generator_name=self.name,
                )
            )

        return queries
