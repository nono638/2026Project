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

## Reranking in RAG Systems — 2026-03-17

> **Summary:** Reranking is a well-studied parameter in RAG evaluation. Ablation studies
> consistently show reranking improves retrieval precision and downstream answer quality,
> but the magnitude varies (2-20 pp depending on baseline, dataset, and reranker). The
> interaction between retrieval method and reranking is real and non-trivial — smaller
> embeddings can outperform larger ones when paired with the right reranker. Relevance-only
> reranking can actually degrade answer quality in some settings.

### 1. Papers studying reranking impact on RAG quality

**RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs**
- Authors: Yue Yu (Georgia Tech), Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You (NVIDIA), Chao Zhang (Georgia Tech), Mohammad Shoeybi, Bryan Catanzaro (NVIDIA)
- Venue: NeurIPS 2024
- Key finding: Instruction-tuning a single LLM for both ranking and generation outperforms dedicated rerankers. Ablation w/o reranking: NQ EM dropped 50.6→48.0, TriviaQA EM 82.9→80.3, PopQA EM 57.6→49.3. Top-k tested: k∈{5,10,20}, saturation around k=10. Initial retrieval N=100 (8B) or N=30 (70B), reranked to k=5 for generation.
- Paper: https://arxiv.org/abs/2407.02485

**Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG**
- Authors: Gabriel de Souza P. Moreira, Ronay Ak, Benedikt Schifferer, Mengyao Xu, Radek Osmulski, Even Oldridge (NVIDIA)
- Year: 2024
- Key finding: Benchmarked 5 rerankers (ms-marco-MiniLM-L-12-v2 33M, jina-reranker-v2-base 278M, mxbai-rerank-large-v1 435M, bge-reranker-v2-m3 568M, NV-RerankQA-Mistral-4B-v3 4B) on NQ, HotpotQA, FiQA. Without reranker: avg NDCG@10=0.7173. Best reranker (NV-Mistral-4B): 0.7694 (+7.3%). Smaller rerankers (MiniLM-33M) actually hurt performance (0.5875). Model size ablation: 33M→0.6227, 435M→0.7277, 4B→0.7414 avg NDCG@10. Bidirectional attention +1.4% over causal. InfoNCE loss +2.5% over BCE.
- Paper: https://arxiv.org/abs/2409.07691

**ARAGOG: Advanced RAG Output Grading**
- Authors: Matous Eibich, Shivay Nagpal, Alexander Fred-Ojala (Predli / UC Berkeley)
- Year: 2024
- Key finding: Compared Naive RAG, HyDE, LLM Rerank, Cohere Rerank, MMR, Multi-query, Sentence Window on 107 QA pairs from 13 AI papers. LLM reranking and HyDE significantly enhanced retrieval precision. Cohere Rerank and MMR showed limited benefits (comparable to or below baseline). Statistical validation with ANOVA + Tukey HSD.
- Paper: https://arxiv.org/abs/2404.01037

**Relevance Isn't All You Need: Scaling RAG Systems With Inference-Time Compute Via Multi-Criteria Reranking (REBEL)**
- Authors: Will LeVine (Microsoft), Bijan Varjavand (Scale AI)
- Year: 2025
- Key finding: Optimizing solely for relevance can degrade answer quality. Cohere and LLM rerankers achieved higher retrieval precision while significantly degrading answer quality. REBEL adds secondary criteria (depth, diversity, clarity, authoritativeness, recency). Multi-criteria reranking improves both relevance AND answer quality simultaneously. Tested on 107 QA pairs, 10 runs per technique.
- Paper: https://arxiv.org/abs/2504.07104

**Enhancing RAG with Two-Stage Retrieval: FlashRank Reranking and Query Expansion**
- Author: Sherine George (Independent)
- Year: 2026
- Key finding: Ablation isolating reranking contribution. FlashRank improved NDCG@10 up to 5.4%, generation accuracy 6-8%, reduced context tokens 35%, response latency 22% improvement over cross-encoder baselines. Tested on BEIR, MS MARCO, FinanceBench.
- Paper: https://arxiv.org/abs/2601.03258

**How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models**
- Authors: Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt
- Venue: EMNLP Findings 2025
- Key finding: Most comprehensive study — 22 methods, 40 variants. LLM-based rerankers excel on familiar queries but generalization to novel queries varies. Lightweight models offer comparable efficiency. Introduced FutureQueryEval benchmark (148 queries) to test temporal novelty. Query novelty significantly impacts reranking effectiveness.
- Paper: https://arxiv.org/abs/2508.16757

**SciRerankBench: Benchmarking Rerankers Towards Scientific RAG-LLMs**
- Year: 2025
- Key finding: First benchmark for rerankers in scientific RAG. Evaluated 13 rerankers on 5 LLM families. Rerankers struggle to filter semantically similar but logically irrelevant passages. Final answer quality constrained by LLM reasoning limitations, not just reranker quality.
- Paper: https://arxiv.org/abs/2508.08742

### 2. Main reranking approaches in production RAG

**Cross-encoders** — Read query+document together, score relevance directly. Most accurate but slowest. Examples: ms-marco-MiniLM, bge-reranker, jina-reranker, mxbai-rerank.

**Late interaction (ColBERT)** — Encode query and document separately, score with token-level MaxSim. Faster than cross-encoders, more accurate than bi-encoders. Sweet spot for latency-sensitive production.

**LLM-based rerankers** — Use instruction-tuned LLMs (GPT-4, Llama, Mistral) to score or rank passages. Most capable but highest latency/cost. RankRAG unifies ranking + generation in one model.

**Lightweight/distilled** — FlashRank (~4MB nano model), runs on CPU, no torch/transformers dependency. Competitive for prototyping.

**API-based** — Cohere Rerank, Jina Reranker API, NVIDIA NeMo Retriever. Zero infrastructure but vendor lock-in.

### 3. Retrieval method × reranking interaction

**Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models**
- Authors: Arjun Rao, Hanieh Alipour, Nick Pendar
- Year: 2025
- Key finding: In tri-modal hybrid retrieval (dense + sparse + graph), MiniLM-v6 (small) outperformed BGE-Large (big) when paired with GPT-4o reranking. Critical result: BGE-Large nDCG@10 DECREASED after reranking (0.6608→0.6170) while MiniLM-v6 INCREASED (0.6505→0.6681) on SciFact. Embedding-reranker compatibility matters more than embedding size alone.
- Paper: https://arxiv.org/abs/2506.00049

**Blended RAG (IBM Research)**
- Authors: Kunal Sawarkar, Abhilasha Mangal, Shivam Raj Solanki
- Year: 2024
- Key finding: Three-way retrieval (BM25 + dense + sparse) outperforms two-way and one-way. On NQ: Blended RAG NDCG@10=0.67 (+5.8% over baseline 0.633). SQuAD F1: 68.4 (50% over fine-tuned baseline). Adding ColBERT reranker on top of three-way hybrid yields further improvement.
- Paper: https://arxiv.org/abs/2404.07220

**Hybrid Retrieval in RAG: A Comparison of Semantic, Lexical and Reranking Methods**
- Authors: Marcin Gabryel, M. Kocic, A. Gabryel
- Venue: ICAISC 2025 (Springer)
- Key finding: E-commerce domain. CQR (conversational query reformulation) + reranker combination proved most universal across precision/recall/F1.
- Paper: https://doi.org/10.1007/978-3-032-03711-4_8

**General pattern:** Hybrid retrieval (dense + sparse) improves NDCG by 26-31% over dense-only. Reranking on top of hybrid still helps but with diminishing returns. The recall-before-precision principle applies: rerankers can only reorder what was retrieved. If recall is already high from hybrid search, reranking adds precision. If recall is low, reranking cannot fix it.

### 4. Local reranker models (no API, ≤8GB VRAM)

| Model | Params | BEIR nDCG@10 | VRAM Notes | License |
|-------|--------|-------------|------------|---------|
| FlashRank nano | ~4MB | competitive | CPU only, no torch needed | Open |
| ms-marco-MiniLM-L-12-v2 | 33M | ~0.59 avg* | <1GB, CPU viable | Apache 2.0 |
| gte-reranker-modernbert-base | 149M | strong** | ~1-2GB FP16 | Open |
| bge-reranker-v2-m3 | 278M | 51.8 (BEIR) | ~1-2GB FP16, CPU viable for <100 pairs | Apache 2.0 |
| mxbai-rerank-large-v1 | 435M | ~0.61 avg* | ~2-3GB FP16 | Open |
| bge-reranker-large | 560M | ~53.8 (BEIR) | ~3-4GB FP16 | Apache 2.0 |
| jina-reranker-v3 | 560M | 61.94 (BEIR) | ~3-4GB FP16 | Apache 2.0 |
| jina-reranker-v2-base-multilingual | 278M | good | ~1-2GB, flash attention | Apache 2.0 |
| qwen3-reranker-0.6b | 600M | modest** | ~2-3GB FP16 | Open |

\* From NVIDIA benchmarking paper (NQ/HotpotQA/FiQA average)
\** From AIMultiple benchmark on Amazon reviews

**AIMultiple independent benchmark (March 2026, ~145k Amazon reviews):**
| Model | Hit@1 | nDCG@10 | Rerank Latency |
|-------|-------|---------|----------------|
| gte-reranker-modernbert-base (149M) | 83.00% | 0.8555 | 424ms |
| jina-reranker-v3 (560M) | 81.33% | 0.8426 | 167ms |
| bge-reranker-v2-m3 | 77.33% | 0.8159 | 527ms |
| bge-reranker-base | 74.33% | 0.7969 | 176ms |
| mxbai-rerank-xsmall (70M) | 64.67% | 0.7263 | 87ms |
| No reranker baseline | 62.67% | 0.6999 | — |

Key insight: gte-reranker-modernbert-base at 149M params matches nemotron-rerank-1b (1.2B params) in accuracy. Diminishing returns above ~500M params for cross-encoders.

**LlamaIndex evaluation (embedding × reranker interaction):**
- JinaAI-v2-base-en + bge-reranker-large: Hit Rate 0.938, MRR 0.869
- OpenAI + CohereRerank: Hit Rate 0.927, MRR 0.866
- bge-large + CohereRerank: Hit Rate 0.876, MRR 0.823
- Reranking consistently improves all embedding models by 3-8% Hit Rate

### 5. Reranking × top-k × model size interactions

**RankRAG (NeurIPS 2024):** Tested k∈{5,10,20}. Smaller k compromises recall, larger k introduces noise. Optimal around k=5-10 after reranking from N=100 candidates. 8B and 70B model sizes tested — both benefit from reranking but the gap narrows with larger generation models.

**NVIDIA benchmarking paper (2024):** Model size ablation: MiniLM 33M → DeBERTa 435M → Mistral 4B shows consistent improvement with scale. But DeBERTa-v3-large at 435M is "surprisingly accurate" — diminishing returns above this for strict latency requirements. Production recommendation: rerank top 20-50 documents down to 5-10 for the LLM.

**Rethinking Hybrid Retrieval (2025):** The interaction is non-trivial. BGE-Large (bigger embeddings) performed WORSE after GPT-4o reranking than MiniLM-v6 (smaller embeddings). This suggests embedding-reranker alignment matters, not just individual component quality. Implication: you cannot evaluate rerankers independently of the embedding model.

**REBEL (2025):** Relevance-only reranking can hurt answer quality even while improving retrieval metrics. Multi-criteria reranking (adding depth, diversity, clarity) resolves this tension. The interaction between "what the reranker optimizes for" and "what the generator needs" is a studied phenomenon.

**Key takeaway for experimental design:** Reranking, top-k, retrieval method, and generation model size all interact. Testing reranking as an isolated variable without controlling for the others produces misleading results. The field is moving toward joint evaluation (e.g., RankRAG, REBEL) rather than component-level benchmarking.
