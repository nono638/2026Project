# Research

> **Daytime only** — night instance does not read this file.
>
> External findings, tools evaluated, patterns worth remembering.

---

## RAG Evaluation Frameworks and Query Generation — 2026-03-16

### Frameworks evaluated

**RAGAS** — most popular synthetic query generation + evaluation for RAG.
Generates seed questions from chunks, evolves into types (simple, reasoning, multi-context, conditional). LLM-as-judge scoring.
- Paper: https://arxiv.org/abs/2309.15217
- Code: https://github.com/explodinggradients/ragas

**ARES (Stanford)** — statistical rigor approach. Synthetic generation + prediction-powered inference with 150–300 human annotations for confidence intervals.
- Paper: https://arxiv.org/abs/2311.09476
- Code: https://github.com/stanford-futuredata/ARES

**LlamaIndex eval** — practical/engineer-friendly. `RagDatasetGenerator` generates N questions per chunk. Less formalized taxonomy.
- Docs: https://docs.llamaindex.ai/en/stable/optimizing/evaluation/

### Benchmarks (curated, not generated)

**BEIR** — meta-benchmark aggregating 18 retrieval datasets across domains. Gold-standard human queries. Used by MTEB for retrieval tasks.
- Paper: https://arxiv.org/abs/2104.08663
- Code: https://github.com/beir-cellar/beir

**MTEB** — 56+ datasets across 8 task types for embedding evaluation. Retrieval subset = BEIR.
- Paper: https://arxiv.org/abs/2210.07316
- Code: https://github.com/embeddings-benchmark/mteb

**MultiHop-RAG** — specifically targets multi-hop reasoning. ~2,500 manually curated queries with evidence chains.
- Paper: https://arxiv.org/abs/2401.15391

### Query generation research

**InPars / InPars-v2** — LLM generates queries from passages. Key finding: quality filtering matters more than generation method.
- InPars: https://arxiv.org/abs/2202.05144
- InPars-v2: https://arxiv.org/abs/2301.01820

**Promptagator (Google Research)** — generates 8 queries per passage, filters via round-trip consistency (retrieve source passage from generated query, keep only if it works).
- Paper: https://arxiv.org/abs/2209.11755

### Key methodological findings

- LLM-generated queries are simpler than human queries (factoid-biased)
- Round-trip consistency filtering is the gold standard quality filter
- Multi-hop/reasoning queries require explicit evolution, not just generation
- Scoring multiple queries per document and averaging increases validity
- Different RAG systems rank differently depending on query type — always report per-type breakdowns

## Query Quality Validation Methods — 2026-03-16

### Computational filters (no LLM required)

**Round-trip consistency (Promptagator)** — generate query from passage, retrieve against full corpus, keep only if source passage appears in top-k. Gold standard for synthetic query filtering.
- Promptagator: https://arxiv.org/abs/2209.11755
- InPars-v2 variant uses cross-encoder instead of retriever: https://arxiv.org/abs/2301.01820

**Cross-encoder relevance scoring (InPars-v2)** — score each (query, passage) pair with a small cross-encoder model (~100M params, e.g. ms-marco-MiniLM-L-6-v2). More precise than round-trip. Purely computational after query generation.
- InPars-v2: https://arxiv.org/abs/2301.01820
- MonoT5 reranker architecture: https://arxiv.org/abs/2003.06713

**Query difficulty estimation** — pre-retrieval signals like IDF scoring, query clarity (KL divergence), query scope. 20+ years of IR research. Catches trivially easy and impossibly hard queries.
- Definitive survey: Carmel & Yom-Tov (2010). "Estimating the Query Difficulty for Information Retrieval." Synthesis Lectures on IR.
- Pre-retrieval predictors survey: Hauff et al. (2008). CIKM.
- Neural IR extension: Roy et al. (2019). arXiv.

**Heuristic pre-filters** — length bounds, question word detection, copy detection, near-duplicate removal. Universal first layer. Catches ~10-20% of generated queries.
- Standard practice across InPars, Promptagator, RAGAS pipelines.

**Distribution analysis** — corpus coverage via embeddings, query type balance, embedding clustering for diversity, length distribution. Catches set-level problems (skew, gaps, homogeneity). Under-used in practice.
- BEIR benchmark discusses query diversity importance: https://arxiv.org/abs/2104.08663
- Arabzadeh et al. (2022). "Shallow Pooling for Sparse Labels." https://arxiv.org/abs/2205.09446

### LLM-dependent methods

**Answerability detection** — classify if a question can be answered from the corpus. Extractive QA model (SQuAD 2.0-trained) or LLM-based check. Catches questions requiring external knowledge.
- SQuAD 2.0: Rajpurkar et al. (2018). https://arxiv.org/abs/1806.03822
- Asai & Choi (2021). https://arxiv.org/abs/2010.11915

**Faithfulness-based filtering** — generate answer from passage, check if answer is faithful. If not, question likely requires external knowledge.
- Honovich et al. (2022). "TRUE: Re-evaluating Factual Consistency Evaluation." https://arxiv.org/abs/2204.04991

**LLM-as-judge for query quality** — multi-dimensional scoring (clarity, specificity, groundedness, non-triviality, naturalness). Expensive but catches subtle issues.
- Zheng et al. (2023). "Judging LLM-as-a-Judge." https://arxiv.org/abs/2306.05685

### Recommended layered approach (cheapest first)

1. Heuristic pre-filters (free, instant)
2. Round-trip consistency (embedding cost only)
3. Cross-encoder scoring (small model inference)
4. Distribution analysis (free, batch-level)
5. Human spot-check of 50-100 queries (most expensive, ground truth)

## Chunking Strategies for RAG — 2026-03-16

### Key finding: chunking × model size interaction is unstudied

No peer-reviewed paper systematically studies how chunking strategy interacts with model size for small (<14B) models. All existing chunking comparisons use GPT-3.5/GPT-4. This is a genuine research gap.

### Existing comparisons

**Wang et al. (2024)** — "Searching for Best Practices in RAG." Tested chunk sizes 128/256/512 on large models. Found 256 optimal, modest overlap helps.
- Paper: https://arxiv.org/abs/2407.01219

**Chen et al. (2023)** — "Dense X Retrieval." Compared sentence/passage/proposition granularity. Propositions best for retrieval precision. Large models only.
- Paper: https://arxiv.org/abs/2312.06648

**Liu et al. (2023)** — "Lost in the Middle." Small models degrade more with irrelevant context. Indirect evidence that retrieval precision (driven by chunking) matters more for small models.
- Paper: https://arxiv.org/abs/2307.03172

**Jina AI (2024)** — "Late Chunking." Embeds full document first, then chunks embedding space. Preserves cross-chunk context.
- Paper: https://arxiv.org/abs/2409.04701

### Production defaults

- Recursive character splitting, 512 tokens, 50-token overlap is the de facto default (LangChain recommendation, most RAG tutorials)
- Semantic chunking is theoretically smarter but produces variable-size chunks and requires an embedding model for splitting
- No rigorous peer-reviewed comparison of fixed vs recursive vs semantic exists — only practitioner blog posts

### Defensible experimental design for chunking comparison

- Strategies to compare: fixed-size, recursive, semantic (3 strategies)
- Control for average chunk size across strategies
- Report both retrieval metrics AND downstream generation quality
- Vary model size as a second factor to test interaction effects
