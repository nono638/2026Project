"""Template-based query generator using entity/fact extraction.

Extracts named entities and factual statements from documents using spaCy,
then slots them into query templates per type. Produces highly uniform queries
at zero LLM cost — useful for controlled experiments where query structure
should be held constant.

Weakness: limited to patterns the templates cover. Cannot generate genuinely
creative or nuanced questions. Best used alongside RAGAS or human queries
for comparison.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import spacy

from src.document import Document
from src.query import Query

# Default templates per query type. Slots: {entity}, {entity2}, {topic}.
DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "factoid": [
        "What is {entity}?",
        "When was {entity} established?",
        "Where is {entity} located?",
        "Who is {entity}?",
    ],
    "reasoning": [
        "Why is {entity} significant in the context of {topic}?",
        "How does {entity} relate to {entity2}?",
        "What is the significance of {entity} according to the document?",
    ],
    "multi_context": [
        "What do {entity} and {entity2} have in common?",
        "Compare {entity} and {entity2} based on the information provided.",
    ],
    "conditional": [
        "What would happen if {entity} were not involved?",
        "Under what conditions is {entity} relevant to {topic}?",
    ],
}

# Target distribution of query types (must sum to 1.0)
TYPE_DISTRIBUTION: dict[str, float] = {
    "factoid": 0.4,
    "reasoning": 0.3,
    "multi_context": 0.2,
    "conditional": 0.1,
}


class TemplateQueryGenerator:
    """Generates queries by extracting entities from documents and filling templates.

    Uses spaCy NER to find named entities (PERSON, ORG, GPE, EVENT, etc.),
    then slots them into predefined query templates for each query type.

    Args:
        spacy_model: spaCy model name for NER and sentence parsing.
            Defaults to 'en_core_web_sm' — small and fast.
        templates: Custom query templates per type. If None, use defaults.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        templates: dict[str, list[str]] | None = None,
    ) -> None:
        self._spacy_model = spacy_model
        self._templates = templates or DEFAULT_TEMPLATES
        self._nlp = spacy.load(spacy_model)

    @property
    def name(self) -> str:
        """Unique identifier (e.g., 'template:en_core_web_sm')."""
        return f"template:{self._spacy_model}"

    def generate(
        self,
        documents: list[Document],
        queries_per_doc: int = 5,
    ) -> list[Query]:
        """Generate template-based queries from documents using NER.

        Algorithm:
        1. For each document, extract entities via spaCy NER
        2. Deduplicate and sort entities by frequency (most common first)
        3. Distribute queries_per_doc across types proportionally
        4. Fill template slots with entities and document title

        Args:
            documents: Documents to generate queries from.
            queries_per_doc: Target number of queries per document.

        Returns:
            List of Query objects with generator_name set to self.name.
        """
        all_queries: list[Query] = []

        for doc in documents:
            doc_queries = self._generate_for_doc(doc, queries_per_doc)
            all_queries.extend(doc_queries)

        return all_queries

    def _generate_for_doc(
        self, doc: Document, queries_per_doc: int
    ) -> list[Query]:
        """Generate queries for a single document.

        Args:
            doc: The document to extract entities from.
            queries_per_doc: Target number of queries.

        Returns:
            List of Query objects. May be fewer than queries_per_doc if the
            document has few entities.
        """
        # Extract and deduplicate entities, sorted by frequency
        entities = self._extract_entities(doc.text)

        if not entities:
            # No extractable entities — skip this document
            print(
                f"WARNING: No entities found in document '{doc.title}', "
                f"skipping query generation",
                file=sys.stderr,
            )
            return []

        topic = doc.title
        has_two_entities = len(entities) >= 2
        entity1 = entities[0]
        entity2 = entities[1] if has_two_entities else None

        # Compute per-type allocations
        allocations = self._compute_allocations(
            queries_per_doc, has_two_entities
        )

        queries: list[Query] = []
        for query_type, count in allocations.items():
            type_templates = self._templates.get(query_type, [])
            if not type_templates:
                continue

            for i in range(count):
                # Cycle through templates for this type
                template = type_templates[i % len(type_templates)]

                # Fill slots — skip templates needing {entity2} if unavailable
                needs_entity2 = "{entity2}" in template
                if needs_entity2 and entity2 is None:
                    # Fallback to a factoid template
                    fallback_templates = self._templates.get("factoid", [])
                    if fallback_templates:
                        template = fallback_templates[i % len(fallback_templates)]
                    else:
                        continue

                text = template.format(
                    entity=entity1,
                    entity2=entity2 or entity1,
                    topic=topic,
                )

                queries.append(
                    Query(
                        text=text,
                        query_type=query_type,
                        source_doc_title=doc.title,
                        generator_name=self.name,
                    )
                )

        return queries

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text, deduplicated and sorted by frequency.

        Args:
            text: Document text to process.

        Returns:
            List of entity strings, most frequent first.
        """
        doc = self._nlp(text)
        # Count entity occurrences, keeping unique text forms
        entity_counts: Counter[str] = Counter()
        for ent in doc.ents:
            entity_counts[ent.text.strip()] += 1

        # Sort by frequency (most common first), then alphabetically for ties
        sorted_entities = sorted(
            entity_counts.keys(),
            key=lambda e: (-entity_counts[e], e),
        )

        return sorted_entities

    def _compute_allocations(
        self, queries_per_doc: int, has_two_entities: bool
    ) -> dict[str, int]:
        """Compute how many queries of each type to generate.

        Distributes queries_per_doc proportionally:
        factoid: 40%, reasoning: 30%, multi_context: 20%, conditional: 10%.

        If the document has fewer than 2 entities, skip multi_context and
        conditional (they need {entity2}), redistribute quota to factoid.

        Args:
            queries_per_doc: Total queries to distribute.
            has_two_entities: Whether the document has at least 2 entities.

        Returns:
            Dict mapping query type to count.
        """
        if has_two_entities:
            distribution = TYPE_DISTRIBUTION
        else:
            # Redistribute multi_context and conditional to factoid
            distribution = {
                "factoid": 0.7,  # 0.4 + 0.2 + 0.1
                "reasoning": 0.3,
            }

        allocations: dict[str, int] = {}
        total = 0
        for qtype, fraction in distribution.items():
            count = round(fraction * queries_per_doc)
            allocations[qtype] = count
            total += count

        # Adjust rounding to hit exact total
        diff = queries_per_doc - total
        if diff != 0:
            # Add/remove from the largest allocation
            largest_type = max(allocations, key=lambda t: allocations[t])
            allocations[largest_type] += diff

        return allocations
