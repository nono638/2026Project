# Task 003: Implement Remaining Components

**Depends on:** task-002 (migrated components must exist first)

## What
Build out the remaining pluggable components that aren't in the MVP set:
- 3 additional RAG strategies: MultiQueryRAG, CorrectiveRAG, AdaptiveRAG
- 3 additional chunkers: FixedSizeChunker, RecursiveChunker, SentenceChunker
- 1 additional embedder: HuggingFaceEmbedder (sentence-transformers)

## Why
The full experiment described in the proposal tests 5 strategies. The tool also needs
multiple chunking and embedding options to be genuinely useful as a general-purpose
RAG research framework.

## Exact Files to Create

### `src/strategies/multi_query.py`

```python
class MultiQueryRAG:
    """Multi-Query RAG strategy.

    Instead of retrieving once with the original question, generates several
    different phrasings, retrieves for each, then merges results before generating.
    Compensates for weak query understanding in small models.
    """
```

**Implementation:**
1. Prompt the model to generate 3 alternative phrasings of the query
2. Retrieve top-k for each phrasing (original + 3 alternatives = 4 retrievals)
3. Merge results: union of all retrieved chunks, deduplicated, re-ranked by max score
4. Generate answer from merged context

**Phrasing prompt:**
```
Generate 3 alternative phrasings of this question. Each should ask the same thing
in a different way. Return only the 3 questions, one per line, no numbering.

Question: {query}
```

Parse the response by splitting on newlines and filtering empty lines.

**Edge case:** If the model fails to generate distinct phrasings (returns gibberish or
repeats), fall back to just the original query.

### `src/strategies/corrective.py`

```python
class CorrectiveRAG:
    """Corrective RAG strategy.

    After retrieval, scores each chunk's relevance and discards low-scoring ones.
    If too many are discarded, triggers a fresh retrieval with a reformulated query.
    Addresses small model hallucination from irrelevant context.

    Based on: Shi et al. (2024). Corrective RAG. arXiv:2401.15884.
    """
```

**Implementation:**
1. Retrieve top-k chunks normally
2. For each chunk, prompt the model to rate relevance: "relevant", "partially relevant", "irrelevant"
3. Keep only "relevant" and "partially relevant" chunks
4. If fewer than 2 chunks survive filtering:
   a. Prompt the model to reformulate the query
   b. Retrieve again with the reformulated query
   c. Apply the same relevance filter
5. Generate answer from the surviving chunks

**Relevance prompt** (same as in self_rag.py):
```
Rate how relevant this passage is to answering the question.
Question: {query}
Passage: {chunk}
Rate as "relevant", "partially relevant", or "irrelevant". Answer with just the rating.
```

**Reformulation prompt:**
```
The following question did not find good matches in the document.
Reformulate it to be more specific or use different terminology.
Original question: {query}
Reformulated question:
```

### `src/strategies/adaptive.py`

```python
class AdaptiveRAG:
    """Adaptive RAG strategy.

    Classifies query complexity first, then routes accordingly:
    - Simple (lookup): skip retrieval, answer from model weights
    - Moderate (synthesis): single retrieval pass
    - Complex (multi_hop): multiple retrieval passes with iterative refinement

    Based on: Jeong et al. (2024). Adaptive-RAG. NAACL.
    """
```

**Implementation:**
1. Classify query complexity by prompting the model:
```
Classify this question's complexity:
- "simple" if it asks for a single fact or definition
- "moderate" if it requires combining 2-3 pieces of information
- "complex" if it requires multi-step reasoning or comparing multiple concepts

Question: {query}
Answer with just: simple, moderate, or complex
```
2. Route based on classification:
   - **simple**: Generate directly without retrieval (just the question to the model)
   - **moderate**: Standard retrieve + generate (same as NaiveRAG)
   - **complex**: Two-pass retrieval:
     a. First retrieval pass → generate intermediate answer
     b. Use intermediate answer to formulate a follow-up query
     c. Second retrieval pass → generate final answer combining both contexts

**Edge case:** If classification fails to parse, default to "moderate" (safe middle ground).

### `src/chunkers/fixed.py`

```python
class FixedSizeChunker:
    """Fixed-size chunking by token count (approximated by words).

    The simplest chunking strategy — splits text into chunks of approximately
    equal size with optional overlap. No semantic awareness.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def name(self) -> str:
        return f"fixed:{self._chunk_size}:{self._overlap}"

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._overlap
        return chunks
```

### `src/chunkers/recursive.py`

```python
class RecursiveChunker:
    """Recursive character text splitting via LangChain.

    Splits on a hierarchy of separators (paragraphs → sentences → words)
    to keep chunks at a target size while respecting text structure.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    @property
    def name(self) -> str:
        return f"recursive:{self._chunk_size}:{self._overlap}"

    def chunk(self, text: str) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._overlap,
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]
```

### `src/chunkers/sentence.py`

```python
class SentenceChunker:
    """Sentence-level chunking.

    Splits text into individual sentences, then groups them into chunks
    of N sentences each. Simple and interpretable.
    """

    def __init__(self, sentences_per_chunk: int = 5):
        self._n = sentences_per_chunk

    @property
    def name(self) -> str:
        return f"sentence:{self._n}"

    def chunk(self, text: str) -> list[str]:
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self._n):
            chunks.append(" ".join(sentences[i:i + self._n]))
        return chunks
```

### `src/embedders/huggingface.py`

```python
import numpy as np
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    """Embedding via sentence-transformers models (runs locally).

    Useful for users who want to compare Ollama embeddings against
    HuggingFace models, or who don't have Ollama installed.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._model_name = model
        self._model = SentenceTransformer(model)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"hf:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
```

## Update `__init__.py` Re-exports

### `src/chunkers/__init__.py`
```python
from src.chunkers.semantic import SemanticChunker
from src.chunkers.fixed import FixedSizeChunker
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.sentence import SentenceChunker

__all__ = ["SemanticChunker", "FixedSizeChunker", "RecursiveChunker", "SentenceChunker"]
```

### `src/embedders/__init__.py`
```python
from src.embedders.ollama import OllamaEmbedder
from src.embedders.huggingface import HuggingFaceEmbedder

__all__ = ["OllamaEmbedder", "HuggingFaceEmbedder"]
```

### `src/strategies/__init__.py`
```python
from src.strategies.naive import NaiveRAG
from src.strategies.self_rag import SelfRAG
from src.strategies.multi_query import MultiQueryRAG
from src.strategies.corrective import CorrectiveRAG
from src.strategies.adaptive import AdaptiveRAG

__all__ = ["NaiveRAG", "SelfRAG", "MultiQueryRAG", "CorrectiveRAG", "AdaptiveRAG"]
```

## Tests to Add: `tests/test_components.py`

1. `test_fixed_chunker` — verify chunk count and overlap behavior
2. `test_sentence_chunker` — verify sentence grouping
3. `test_recursive_chunker` — verify it produces chunks under target size
4. `test_all_chunkers_protocol` — all chunkers pass `isinstance(c, Chunker)`
5. `test_all_strategies_protocol` — all strategies pass `isinstance(s, Strategy)`
6. `test_huggingface_embedder_protocol` — passes `isinstance(e, Embedder)`

For strategy tests, use the MockEmbedder/MockChunker from task-001's test_core.py to
build a real Retriever, then call each strategy with a small Ollama model. Mark these
as `@pytest.mark.slow` since they require Ollama to be running.

## What NOT to Touch
- `src/protocols.py`, `src/retriever.py`, `src/experiment.py`, `src/features.py`
- `src/model/`, `src/app.py`
- Existing files created in task-002 (only add new files)
