"""FastAPI endpoint for SmallModelBigStrategy.

User pastes a document and question → gets routed to the recommended
(chunker, embedder, strategy, model) configuration with a confidence score.

Updated from the original 2-axis (strategy, model) to 4-axis recommendation
to match the pluggable architecture introduced in task-001/task-002.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.chunkers import SemanticChunker
from src.embedders import OllamaEmbedder
from src.retriever import Retriever
from src.features import extract_features
from src.model.predict import predict


app = FastAPI(
    title="SmallModelBigStrategy",
    description="Predicts the optimal RAG configuration for your query.",
)


class QueryRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    document: str
    question: str
    query_type: str = "lookup"  # lookup, synthesis, or multi_hop


class RecommendationResponse(BaseModel):
    """Response body with all 4 axes of the recommended configuration."""

    chunker: str
    embedder: str
    strategy: str
    model: str
    confidence: float


# Lazy-initialized default components for inference-time feature extraction.
# These are used to chunk/embed the user's document so we can extract features
# for the meta-learner. The actual recommendation may suggest different components.
_default_chunker: SemanticChunker | None = None
_default_embedder: OllamaEmbedder | None = None


def _get_defaults() -> tuple[SemanticChunker, OllamaEmbedder]:
    """Return lazily-initialized default chunker and embedder.

    Returns:
        Tuple of (SemanticChunker, OllamaEmbedder).
    """
    global _default_chunker, _default_embedder
    if _default_chunker is None:
        _default_chunker = SemanticChunker()
        _default_embedder = OllamaEmbedder()
    return _default_chunker, _default_embedder


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: QueryRequest) -> RecommendationResponse:
    """Get the recommended 4-axis RAG configuration.

    Chunks and embeds the document using default components to extract features,
    then predicts the optimal (chunker, embedder, strategy, model) config.

    Args:
        req: QueryRequest with document, question, and optional query_type.

    Returns:
        RecommendationResponse with all 4 axes and a confidence score.
    """
    chunker, embedder = _get_defaults()
    chunks = chunker.chunk(req.document)
    retriever = Retriever(chunks, embedder)
    features = extract_features(req.question, req.document, retriever)
    features["query_type"] = req.query_type
    result = predict(features)
    return RecommendationResponse(**result)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dict with status 'ok'.
    """
    return {"status": "ok"}
