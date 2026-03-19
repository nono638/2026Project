"""Feature extraction for the meta-learner.

Extracts properties from the query and document before any model runs.
These become the feature matrix X for the XGBoost classifier.

Query-level features:
- query_length: token count (approximated by word count)
- query_type: lookup / synthesis / multi_hop (one-hot encoded at training time)
- num_named_entities: proxy for specificity

Document-level features (basic):
- doc_length: token count of source document
- doc_vocab_entropy: lexical diversity proxy

Document characterization features (content-aware):
- doc_ner_density: distinct named entities per 1000 tokens (spaCy NER).
  High = structured/factual content (legal, medical). Low = narrative/abstract.
- doc_ner_repetition: total entity mentions / distinct entities.
  High = document revisits same entities (good for RAG — retrieval finds anchors).
  Low = many one-off mentions (harder to retrieve coherently).
- doc_topic_count: number of semantic clusters in chunk embeddings (sklearn KMeans
  with silhouette-based k selection). Measures how many distinct subjects the document
  covers.
- doc_topic_density: topic_count / (doc_length / 1000). Topics per 1000 tokens.
- doc_semantic_coherence: mean cosine similarity between consecutive chunk embeddings.
  High = smooth narrative flow. Low = jumpy reference material.

Retrieval-level features:
- mean_retrieval_score: how confidently embedding matched query to doc
- var_retrieval_score: high variance = scattered context (Zhang et al., 2025)

Research context: No published work uses document-level features to predict optimal
RAG configuration. See reference/research.md "Document Characterization for RAG
Configuration Selection — 2026-03-19" for the literature gap analysis.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.retriever import Retriever

# Lazy-loaded spaCy model — loaded once on first use, kept in module scope.
# en_core_web_sm is installed as a pip package in the project venv
# (site-packages/en_core_web_sm/), so spacy.load() finds it via Python's
# import system, not any system-level spaCy data directory.
_spacy_nlp = None


def _get_spacy_nlp():
    """Lazy-load the spaCy English NER model from the project venv.

    Returns:
        A spaCy Language model with NER pipeline.

    Raises:
        OSError: If en_core_web_sm is not installed in the active venv.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        # Disable everything except NER for speed — we don't need
        # parsing, lemmatization, or text categorization here.
        _spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _spacy_nlp


def extract_features(query: str, document: str, retriever: Retriever) -> dict[str, float]:
    """Extract features for a (query, document) pair.

    Combines query-level, document-level, and retrieval-level features.
    Document characterization features (NER, topics, coherence) are derived
    from the document text and chunk embeddings in the retriever.

    Args:
        query: The question text.
        document: The full document text.
        retriever: A Retriever instance (used to get retrieval scores
                   and chunk embeddings for document characterization).

    Returns:
        Dict of feature name -> value.
    """
    retrieved = retriever.retrieve(query)
    scores = [r["score"] for r in retrieved]

    # Document characterization — computed from full doc text and chunk embeddings
    ner_density, ner_repetition = _ner_features(document)
    topic_count, semantic_coherence = _embedding_features(retriever)
    doc_length = len(document.split())
    topic_density = topic_count / (doc_length / 1000) if doc_length > 0 else 0.0

    return {
        # Query-level
        "query_length": len(query.split()),
        "num_named_entities": _count_entities(query),
        # Document-level (basic)
        "doc_length": doc_length,
        "doc_vocab_entropy": _vocab_entropy(document),
        # Document characterization (content-aware)
        "doc_ner_density": ner_density,
        "doc_ner_repetition": ner_repetition,
        "doc_topic_count": topic_count,
        "doc_topic_density": topic_density,
        "doc_semantic_coherence": semantic_coherence,
        # Retrieval-level
        "mean_retrieval_score": float(np.mean(scores)) if scores else 0.0,
        "var_retrieval_score": float(np.var(scores)) if scores else 0.0,
    }


def _ner_features(text: str) -> tuple[float, float]:
    """Extract NER-based document characterization features using spaCy.

    Runs spaCy's NER pipeline on the full document text to count entity
    mentions. Returns density (distinct entities per 1000 tokens) and
    repetition ratio (total mentions / distinct entities).

    Args:
        text: Full document text.

    Returns:
        Tuple of (ner_density, ner_repetition).
        ner_density: distinct entities per 1000 tokens.
        ner_repetition: total mentions / distinct count (>=1.0 always).
                        Returns 0.0 if no entities found.
    """
    nlp = _get_spacy_nlp()

    # spaCy's max_length defaults to 1M chars. For very long documents,
    # process in chunks to avoid memory issues. Most documents are under
    # 100K chars so this rarely triggers.
    if len(text) > nlp.max_length:
        text = text[:nlp.max_length]

    doc = nlp(text)

    # Collect all entity mentions — normalized to lowercase for deduplication.
    # Using .text (surface form) rather than .lemma_ because entity names
    # shouldn't be lemmatized ("United States" != "unite state").
    mentions: list[str] = [ent.text.lower().strip() for ent in doc.ents]
    total_mentions = len(mentions)
    distinct_entities = len(set(mentions))

    token_count = len(text.split())
    if token_count == 0:
        return 0.0, 0.0

    ner_density = distinct_entities / (token_count / 1000)
    ner_repetition = total_mentions / distinct_entities if distinct_entities > 0 else 0.0

    return ner_density, ner_repetition


def _embedding_features(retriever: Retriever) -> tuple[float, float]:
    """Extract embedding-based document characterization features.

    Uses chunk embeddings stored in the retriever's FAISS index to compute:
    1. Topic count — number of semantic clusters (KMeans with silhouette selection)
    2. Semantic coherence — mean cosine similarity between consecutive chunks

    All computation uses sklearn (pure Python, no external data downloads)
    and numpy. FAISS index.reconstruct() pulls vectors without re-embedding.

    Args:
        retriever: A Retriever with a populated FAISS index.

    Returns:
        Tuple of (topic_count, semantic_coherence).
        Returns (1, 1.0) for single-chunk documents (trivially one topic,
        perfect coherence).
    """
    n_chunks = retriever._index.ntotal
    if n_chunks <= 1:
        return 1, 1.0

    # Reconstruct all chunk embeddings from the FAISS index.
    # These are already L2-normalized (done during Retriever init),
    # so dot product = cosine similarity.
    embeddings = np.array(
        [retriever._index.reconstruct(i) for i in range(n_chunks)],
        dtype=np.float32,
    )

    # --- Semantic coherence ---
    # Average cosine similarity between consecutive chunk embeddings.
    # Normalized vectors → dot product = cosine similarity.
    coherence = _consecutive_cosine_mean(embeddings)

    # --- Topic count via KMeans + silhouette ---
    topic_count = _estimate_topic_count(embeddings)

    return float(topic_count), coherence


def _consecutive_cosine_mean(embeddings: np.ndarray) -> float:
    """Mean cosine similarity between consecutive embedding vectors.

    Since vectors are L2-normalized, dot product equals cosine similarity.

    Args:
        embeddings: (n, d) array of L2-normalized embeddings.

    Returns:
        Mean cosine similarity between adjacent pairs. Range [-1, 1],
        typically 0.3-0.9 for real documents.
    """
    if len(embeddings) < 2:
        return 1.0

    # Vectorized dot product between consecutive pairs
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return float(np.mean(sims))


def _estimate_topic_count(embeddings: np.ndarray, max_k: int = 10) -> int:
    """Estimate number of semantic topics via KMeans + silhouette score.

    Tries k=2..min(max_k, n_chunks-1) and picks the k with the highest
    silhouette score. Silhouette measures how well-separated clusters are
    (-1 to 1, higher = better). If no k improves on a single cluster,
    returns 1.

    Uses sklearn (installed in project venv, no external data downloads).

    Args:
        embeddings: (n, d) array of chunk embeddings.
        max_k: Maximum number of clusters to try. Capped at n_chunks - 1.

    Returns:
        Estimated number of topics (1 to max_k).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 3:
        # silhouette_score needs at least 2 clusters and 3 samples
        return 1

    upper = min(max_k, n - 1)
    if upper < 2:
        return 1

    best_k = 1
    best_score = -1.0

    for k in range(2, upper + 1):
        # n_init="auto" uses sklearn's default (10 for classic, 1 for elkan).
        # random_state for reproducibility across experiment runs.
        km = KMeans(n_clusters=k, n_init="auto", random_state=42, max_iter=100)
        labels = km.fit_predict(embeddings)

        # silhouette_score requires at least 2 distinct labels
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _count_entities(text: str) -> int:
    """Count capitalized multi-word sequences as a rough NER proxy.

    This is a fast heuristic used for query-level features where running
    full spaCy NER would be overkill. For document-level NER, use
    _ner_features() which runs proper spaCy NER.

    Args:
        text: Input text to scan for named entities.

    Returns:
        Count of likely named entity words.
    """
    words = text.split()
    count = 0
    for word in words:
        if not word:
            continue
        if word[0].isupper() and word not in ("What", "When", "Where", "Who",
                                                "Why", "How", "Is", "Are",
                                                "Does", "Do", "Can", "The", "A"):
            count += 1
    return count


def _vocab_entropy(text: str) -> float:
    """Calculate Shannon entropy of word frequency distribution.

    Args:
        text: Input text to analyze.

    Returns:
        Shannon entropy value (higher = more diverse vocabulary).
    """
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
