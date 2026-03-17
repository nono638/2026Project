"""Distribution analysis for generated query sets.

Analyzes a query set as a whole for coverage, diversity, type balance,
and other set-level quality signals. Produces a report rather than
filtering individual queries.

Set-level analysis is under-used in RAG evaluation pipelines despite being
low-cost and highly informative (noted in Thakur et al., 2021, arxiv:2104.08663).
Catches problems that per-query filters miss: topic skew, corpus coverage gaps,
query type imbalance, and difficulty homogeneity.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.document import Document
    from src.protocols import Embedder
    from src.query import Query


class DistributionAnalyzer:
    """Analyzes query set distribution for quality signals.

    Produces a report dict with type distribution, length stats, coverage,
    diversity, and duplicate detection. Optionally uses an embedder for
    embedding-based diversity and corpus coverage analysis.

    Args:
        embedder: Optional embedder for embedding-based analyses. If None,
            skip those analyses (only run statistical checks).
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder

    @property
    def name(self) -> str:
        """Analyzer identifier."""
        return "distribution_analyzer"

    def analyze(
        self,
        queries: list[Query],
        documents: list[Document] | None = None,
    ) -> dict:
        """Analyze query set distribution and quality.

        Args:
            queries: List of queries to analyze.
            documents: Optional list of corpus documents for coverage analysis.

        Returns:
            Dict with analysis results including type_distribution, length_stats,
            coverage, diversity, duplicates, and warnings.
        """
        if not queries:
            return {
                "total_queries": 0,
                "type_distribution": {},
                "length_stats": {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0},
                "docs_with_queries": 0,
                "docs_without_queries": 0,
                "queries_per_doc": {"mean": 0, "min": 0, "max": 0},
                "lexical_diversity": 0,
                "duplicate_count": 0,
                "warnings": [],
            }

        warnings: list[str] = []
        result: dict = {"total_queries": len(queries)}

        # 1. Type distribution
        result["type_distribution"] = self._type_distribution(queries, warnings)

        # 2. Length stats
        result["length_stats"] = self._length_stats(queries, warnings)

        # 3. Document coverage
        doc_titles = {q.source_doc_title for q in queries}
        result["docs_with_queries"] = len(doc_titles)

        if documents is not None:
            all_titles = {d.title for d in documents}
            uncovered = all_titles - doc_titles
            result["docs_without_queries"] = len(uncovered)
            if documents and len(uncovered) / len(documents) > 0.2:
                coverage_pct = (1 - len(uncovered) / len(documents)) * 100
                warnings.append(
                    f"{len(uncovered)} documents have no queries — "
                    f"coverage is {coverage_pct:.1f}%, consider regenerating for these docs"
                )
        else:
            result["docs_without_queries"] = 0

        # 4. Queries per doc
        result["queries_per_doc"] = self._queries_per_doc(queries, warnings)

        # 5. Lexical diversity
        result["lexical_diversity"] = self._lexical_diversity(queries, warnings)

        # 6. Duplicates
        result["duplicate_count"] = self._duplicate_count(queries, warnings)

        # 7. Embedding-based analysis (if embedder provided)
        if self._embedder is not None:
            result["embedding_diversity"] = self._embedding_diversity(queries, warnings)
            if documents:
                result["corpus_coverage"] = self._corpus_coverage(queries, documents, warnings)

        result["warnings"] = warnings
        return result

    def print_report(self, analysis: dict) -> None:
        """Pretty-print the analysis dict.

        Args:
            analysis: Dict returned by analyze().
        """
        print()
        print("RAGBench Query Set Analysis")
        print("=" * 40)
        print(f"Total queries:          {analysis['total_queries']}")

        # Type distribution
        if analysis.get("type_distribution"):
            parts = []
            for qtype, info in analysis["type_distribution"].items():
                parts.append(f"{qtype}: {info['count']} ({info['fraction']:.0%})")
            print(f"Type distribution:      {', '.join(parts)}")

        # Length stats
        ls = analysis.get("length_stats", {})
        if ls.get("mean"):
            print(
                f"Length (words):         mean={ls['mean']:.1f}, "
                f"median={ls['median']}, range=[{ls['min']}, {ls['max']}]"
            )

        # Document coverage
        dwq = analysis.get("docs_with_queries", 0)
        dwoq = analysis.get("docs_without_queries", 0)
        total_docs = dwq + dwoq
        if total_docs > 0:
            pct = dwq / total_docs * 100
            print(f"Document coverage:      {dwq}/{total_docs} docs ({pct:.1f}%)")
        else:
            print(f"Document coverage:      {dwq} docs with queries")

        # Queries per doc
        qpd = analysis.get("queries_per_doc", {})
        if qpd.get("mean"):
            print(f"Queries per doc:        mean={qpd['mean']:.1f}, min={qpd['min']}, max={qpd['max']}")

        print(f"Lexical diversity:      {analysis.get('lexical_diversity', 0):.2f} (type-token ratio)")
        print(f"Duplicates:             {analysis.get('duplicate_count', 0)}")

        # Embedding analysis
        if "embedding_diversity" in analysis:
            ed = analysis["embedding_diversity"]
            print()
            print("Embedding analysis:")
            print(f"  Mean pairwise distance: {ed.get('mean_pairwise_distance', 0):.2f}")
            print(f"  Cluster count:          {ed.get('cluster_count', 0)}")
        if "corpus_coverage" in analysis:
            print(f"  Corpus coverage:        {analysis['corpus_coverage']:.2f}")

        # Warnings
        if analysis.get("warnings"):
            print()
            print("Warnings:")
            for w in analysis["warnings"]:
                print(f"  WARNING: {w}")

    def _type_distribution(
        self, queries: list[Query], warnings: list[str]
    ) -> dict:
        """Count queries by type and check for imbalance.

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Dict mapping query type to count and fraction.
        """
        known_types = {"factoid", "reasoning", "multi_context", "conditional"}
        counts: Counter[str] = Counter()
        for q in queries:
            if q.query_type in known_types:
                counts[q.query_type] += 1
            else:
                counts["other"] += 1

        total = len(queries)
        dist: dict = {}
        for qtype in list(known_types) + ["other"]:
            count = counts.get(qtype, 0)
            fraction = count / total if total > 0 else 0
            if count > 0 or qtype != "other":
                dist[qtype] = {"count": count, "fraction": fraction}

            # Warning if any type is <5% or >60%
            if count > 0 and (fraction < 0.05 or fraction > 0.60):
                warnings.append(
                    f"Type '{qtype}' is {fraction:.0%} of queries "
                    f"({count}/{total}) — potential imbalance"
                )

        return dist

    def _length_stats(
        self, queries: list[Query], warnings: list[str]
    ) -> dict:
        """Compute word-count statistics for queries.

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Dict with mean, median, min, max, std of word counts.
        """
        lengths = [len(q.text.split()) for q in queries]
        mean_len = statistics.mean(lengths)

        if mean_len < 7:
            warnings.append(f"Mean query length is {mean_len:.1f} words — queries may be too short")
        elif mean_len > 30:
            warnings.append(f"Mean query length is {mean_len:.1f} words — queries may be too long")

        return {
            "mean": mean_len,
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        }

    def _queries_per_doc(
        self, queries: list[Query], warnings: list[str]
    ) -> dict:
        """Compute queries-per-document statistics.

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Dict with mean, min, max of queries per document.
        """
        doc_counts: Counter[str] = Counter()
        for q in queries:
            doc_counts[q.source_doc_title] += 1

        counts = list(doc_counts.values())
        min_count = min(counts)
        max_count = max(counts)

        if min_count > 0 and max_count / min_count > 5:
            warnings.append(
                f"Queries per doc ratio is {max_count}/{min_count} = {max_count/min_count:.1f}x "
                f"— highly unbalanced"
            )

        return {
            "mean": statistics.mean(counts),
            "min": min_count,
            "max": max_count,
        }

    def _lexical_diversity(
        self, queries: list[Query], warnings: list[str]
    ) -> float:
        """Compute type-token ratio across all query texts.

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Type-token ratio (unique words / total words).
        """
        all_words: list[str] = []
        for q in queries:
            all_words.extend(q.text.lower().split())

        if not all_words:
            return 0.0

        ttr = len(set(all_words)) / len(all_words)

        if ttr < 0.3:
            warnings.append(
                f"Lexical diversity is {ttr:.2f} — queries are very repetitive"
            )

        return ttr

    def _duplicate_count(
        self, queries: list[Query], warnings: list[str]
    ) -> int:
        """Count exact string duplicates (case-insensitive, stripped).

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Number of duplicate query texts.
        """
        texts = [q.text.strip().lower() for q in queries]
        total = len(texts)
        unique = len(set(texts))
        duplicates = total - unique

        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate queries detected")

        return duplicates

    def _embedding_diversity(
        self, queries: list[Query], warnings: list[str]
    ) -> dict:
        """Compute embedding-based diversity metrics.

        Embeds all queries, computes mean pairwise cosine distance,
        and runs KMeans clustering.

        Args:
            queries: Queries to analyze.
            warnings: List to append warnings to.

        Returns:
            Dict with mean_pairwise_distance and cluster_count.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_distances

        texts = [q.text for q in queries]
        embeddings = self._embedder.embed(texts)

        # Mean pairwise cosine distance
        dist_matrix = cosine_distances(embeddings)
        # Upper triangle only (exclude diagonal)
        n = len(embeddings)
        if n < 2:
            mean_dist = 0.0
        else:
            upper_indices = np.triu_indices(n, k=1)
            mean_dist = float(np.mean(dist_matrix[upper_indices]))

        if mean_dist < 0.3:
            warnings.append(
                f"Mean pairwise embedding distance is {mean_dist:.2f} — "
                f"queries are too similar"
            )

        # KMeans clustering
        k = min(10, max(2, len(queries) // 5))
        if len(queries) >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            # Count non-empty clusters
            cluster_labels = kmeans.labels_
            cluster_count = len(set(cluster_labels))
        else:
            cluster_count = len(queries)

        return {
            "mean_pairwise_distance": mean_dist,
            "cluster_count": cluster_count,
        }

    def _corpus_coverage(
        self,
        queries: list[Query],
        documents: list[Document],
        warnings: list[str],
    ) -> float:
        """Compute fraction of documents covered by nearby query embeddings.

        Embeds document titles (or first 100 words) and checks if any query
        embedding is within cosine distance 0.5 of each document.

        Args:
            queries: Queries to check coverage for.
            documents: Documents to check coverage against.
            warnings: List to append warnings to.

        Returns:
            Coverage fraction (0-1).
        """
        from sklearn.metrics.pairwise import cosine_distances

        # Embed queries
        query_texts = [q.text for q in queries]
        query_embeddings = self._embedder.embed(query_texts)

        # Embed document summaries (title + first 100 words)
        doc_texts = []
        for doc in documents:
            words = doc.text.split()[:100]
            doc_texts.append(f"{doc.title} {' '.join(words)}")
        doc_embeddings = self._embedder.embed(doc_texts)

        # For each doc, check if any query is within cosine distance 0.5
        dist_matrix = cosine_distances(doc_embeddings, query_embeddings)
        covered = 0
        for i in range(len(documents)):
            if np.min(dist_matrix[i]) < 0.5:
                covered += 1

        coverage = covered / len(documents) if documents else 0.0

        if coverage < 0.7:
            warnings.append(
                f"Corpus coverage is {coverage:.0%} — "
                f"many documents lack nearby queries"
            )

        return coverage
