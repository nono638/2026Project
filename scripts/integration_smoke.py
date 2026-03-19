"""Minimal end-to-end integration test using real Ollama.

Tests the full pipeline: chunk -> embed -> index -> retrieve -> generate.
Uses qwen3:0.6b (smallest model) on a tiny synthetic document.
No scorer or API keys required.

Usage:
    python scripts/integration_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("\nRAGBench Integration Smoke Test")
    print("=" * 50)

    # --- Test document ---
    doc_text = (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was designed by Gustave Eiffel and built between 1887 and 1889. "
        "The tower is 330 metres tall and was the tallest man-made structure "
        "in the world until 1930. It receives about 7 million visitors per year. "
        "The tower has three levels for visitors, with restaurants on the first "
        "and second levels. The top level's upper platform is 276 metres above "
        "the ground. Tickets can be purchased to ascend by stairs or lift. "
        "The tower was initially criticized by some of France's leading artists "
        "and intellectuals for its design, but it has become a global cultural "
        "icon of France and one of the most recognizable structures in the world."
    )
    query = "How tall is the Eiffel Tower?"

    # --- 1. Chunker ---
    print("\n[1/5] Chunking...")
    from src.chunkers import RecursiveChunker
    chunker = RecursiveChunker()
    chunks = chunker.chunk(doc_text)
    print(f"  OK: {len(chunks)} chunks from RecursiveChunker")
    for i, c in enumerate(chunks[:3]):
        print(f"  chunk[{i}]: {c[:80]}...")

    # --- 2. Embedder ---
    print("\n[2/5] Embedding...")
    from src.embedders import OllamaEmbedder
    embedder = OllamaEmbedder()
    t0 = time.perf_counter()
    vectors = embedder.embed(chunks)
    t1 = time.perf_counter()
    print(f"  OK: {vectors.shape} in {(t1-t0)*1000:.0f}ms (dim={embedder.dimension})")

    # --- 3. Retriever ---
    print("\n[3/5] Building retriever index...")
    from src.retriever import Retriever

    # Test all 3 modes
    for mode in ("dense", "sparse", "hybrid"):
        retriever = Retriever(chunks, embedder, top_k=3, mode=mode)
        results = retriever.retrieve(query)
        top_text = results[0]["text"][:60] if results else "(empty)"
        print(f"  {mode:7s}: {len(results)} results, top: \"{top_text}...\"")

    # --- 4. Strategy (NaiveRAG with real Ollama) ---
    print("\n[4/5] Running NaiveRAG with qwen3:0.6b...")
    from src.strategies import NaiveRAG
    from src.llms import OllamaLLM

    llm = OllamaLLM()
    strategy = NaiveRAG(llm=llm)
    retriever = Retriever(chunks, embedder, top_k=3, mode="hybrid")

    t0 = time.perf_counter()
    answer = strategy.run(query, retriever, "qwen3:0.6b")
    t1 = time.perf_counter()
    print(f"  OK: answer in {(t1-t0)*1000:.0f}ms")
    print(f"  Answer: {answer[:200]}")

    # --- 5. Feature extraction ---
    print("\n[5/5] Feature extraction...")
    from src.features import extract_features
    features = extract_features(query, doc_text, retriever)
    print(f"  OK: {len(features)} features extracted")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("All pipeline stages passed!")
    print("The code is ready for real experiments.")


if __name__ == "__main__":
    main()
