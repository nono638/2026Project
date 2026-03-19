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

## Cloud GPU Rental Options for Ollama — 2026-03-18

> **Requirements:** Run Ollama serving multiple models (largest 8B, ~8GB VRAM needed).
> Budget-first. ~2 months (March-May 2026). Burst compute for experiments (10-20 hours
> total GPU time), then light inference for a web demo. Also need to host a simple FastAPI
> findings gallery (same or separate service).

### Comparison Table

| Provider | Cheapest Viable GPU | VRAM | $/hr | Billing | Ollama? | Web Endpoint? | Min Commit | Gotchas |
|----------|-------------------|------|------|---------|---------|---------------|------------|---------|
| **Vast.ai** | RTX 3090 | 24GB | ~$0.12 | per-second | Yes (Docker) | Yes (port mapping) | None | Marketplace prices fluctuate; storage costs vary by host; reliability varies; interruptible instances ~50% cheaper |
| **Vast.ai** | RTX A4000 | 16GB | ~$0.08 | per-second | Yes (Docker) | Yes (port mapping) | None | 16GB is tight but sufficient for 8B models |
| **Vast.ai** | RTX 3060 | 12GB | ~$0.04 | per-second | Yes (Docker) | Yes (port mapping) | None | Cheapest option; 12GB enough for 8B Q4 quantized |
| **RunPod** | A4000 | 16GB | $0.17 | per-second | Yes (Docker/template) | Yes (exposed HTTP ports) | None | Spot: $0.09/hr; storage $0.10/GB/mo; network volumes persist models across cold starts |
| **RunPod** | RTX 3090 | 24GB | $0.22 | per-second | Yes (Docker/template) | Yes (exposed HTTP ports) | None | Community Cloud; Spot: $0.11/hr |
| **RunPod** | RTX 4090 | 24GB | $0.34 | per-second | Yes (Docker/template) | Yes (exposed HTTP ports) | None | Spot: $0.20/hr |
| **Modal** | T4 | 16GB | $0.59 | per-second | Yes (SDK-based) | Yes (native web endpoints) | None | $30/mo free credits; cold starts 2-4s; idle containers billed; scales to zero; no raw Docker — must use Modal SDK |
| **Modal** | L4 | 24GB | $0.80 | per-second | Yes (SDK-based) | Yes (native web endpoints) | None | $30/mo free credits; better perf than T4; academic credits up to $10k |
| **Jarvis Labs** | L4 | 24GB | $0.44 | per-minute | Yes (bare-metal VM) | Yes (SSH/custom ports) | None | Managed JupyterLab; bare-metal VMs support Docker; no spot pricing |
| **Jarvis Labs** | A5000 | 24GB | $0.49 | per-minute | Yes (bare-metal VM) | Yes (SSH/custom ports) | None | Good middle ground |
| **Jarvis Labs** | A6000 | 48GB | $0.79 | per-minute | Yes (bare-metal VM) | Yes (SSH/custom ports) | None | Overkill VRAM but reasonable price |
| **Lambda Labs** | A6000 | 48GB | $0.80 | 1-min increments | Yes (bare-metal) | Yes (full VM) | None | No consumer GPUs offered; smallest GPU is A6000; zero egress fees |
| **Paperspace (DO)** | A4000 | 16GB | $0.76 | per-hour | Yes (Docker deployments) | Yes (container-as-service) | None | $39/mo Growth plan needed for high-end GPUs; hourly billing (not per-second); storage costs on top |
| **Google Colab Pro** | T4 | 15GB | ~$0.20/hr* | compute units | Hacky (no Docker) | No (notebook only) | $9.99/mo | *CU-based pricing; no persistent endpoint; sessions disconnect; no Docker; tunneling needed for web access |
| **Google Colab Pro+** | A100 | 40GB | ~$1.30/hr* | compute units | Hacky (no Docker) | No (notebook only) | $49.99/mo | *CU-based; same limitations as Pro but better GPU access |
| **Fly.io** | L40S | 48GB | $0.70 | per-second | Yes (Docker) | Yes (native) | None | **GPUs deprecated July 31, 2026** — do not use for new projects |
| **TensorDock** | RTX 3090 | 24GB | ~$0.15 | per-second | Yes (Docker on all VMs) | Yes (full VM) | None | Marketplace model; Docker included on all templates; pricing varies by host |
| **SaladCloud** | RTX 3060 | 12GB | ~$0.04 | per-second | Theoretically | Limited (IPv6 required) | None | **NOT recommended:** no persistent volumes; short-lived nodes; consumer hardware variability; no public IPv4; designed for batch workloads, not interactive serving |

\* Google Colab pricing is approximate — based on compute unit consumption rates, not direct hourly billing.

### Analysis and Recommendations

**For burst GPU experiments (10-20 hours):**

1. **Vast.ai (Best Budget)** — RTX 3060 at ~$0.04/hr or RTX A4000 at ~$0.08/hr. Total cost for 20 hours: $0.80-$1.60. Full Docker support, can run Ollama directly. Marketplace model means prices vary but are consistently the cheapest. Use on-demand (not interruptible) for experiment reliability. Expose ports for API access.

2. **RunPod (Best Experience)** — A4000 at $0.17/hr or spot at $0.09/hr. Total cost for 20 hours: $1.80-$3.40. Pre-built Ollama template, better UI than Vast.ai, network volumes persist models. Slightly more expensive but much smoother developer experience. FastAPI can be exposed via HTTP port configuration.

3. **Modal (Best for Serverless Pattern)** — T4 at $0.59/hr with $30/mo free credits. Total cost for 20 hours: $11.80 minus $30 credit = potentially free for first month. Native web endpoints, scales to zero. Downside: must use Modal's SDK (no raw Docker), cold starts of 2-4s, and Ollama integration requires their specific approach. Academic credits up to $10k available.

**For the web demo (light inference, needs to stay up):**

The web demo has different requirements — it needs a persistent public URL, not burst compute. Options:

- **Separate the concerns:** Host the FastAPI gallery on a free tier (Render, Fly.io non-GPU, Vercel) and call the GPU instance only when inference is needed. This avoids paying GPU rates for serving static HTML.

- **RunPod with network volume:** Keep a pod running at $0.17/hr (A4000) = ~$122/mo for always-on. Expensive for a demo. Better: use their serverless endpoint with network volume so models persist, pay only per-request.

- **Modal deployed endpoint:** `modal deploy` creates a persistent URL. Scales to zero when idle (no cost). Cold start of 2-4s on first request. With $30/mo free credits, light inference demo usage may be fully covered. This is the cleanest option for a demo that gets occasional traffic.

- **Vast.ai persistent instance:** Cheapest always-on option at ~$0.04-0.08/hr = $29-58/mo. But marketplace machines can go offline; not ideal for a demo that needs reliability.

**Recommended architecture:**

1. **Experiments phase (March-April):** Vast.ai or RunPod for burst GPU. Spin up, run experiments, spin down. Budget: $2-5 total.
2. **Demo phase (April-May):** Modal for the inference endpoint (scales to zero, free credits cover light usage) + Render free tier for the static gallery frontend. Budget: $0-30/mo depending on usage.

### Sources

- Vast.ai pricing: https://vast.ai/pricing — https://computeprices.com/providers/vast
- RunPod pricing: https://www.runpod.io/pricing — https://computeprices.com/providers/runpod
- Modal pricing: https://modal.com/pricing
- Lambda Labs pricing: https://lambda.ai/pricing
- Paperspace/DO pricing: https://docs.digitalocean.com/products/paperspace/pricing/
- Jarvis Labs pricing: https://jarvislabs.ai/pricing — https://computeprices.com/providers/jarvis
- TensorDock: https://www.tensordock.com/cloud-gpus.html
- SaladCloud: https://salad.com/pricing
- Fly.io GPU deprecation: https://community.fly.io/t/gpu-migration-fly-io-gpus-will-be-deprecated-as-of-july-31-2026/27110
- GPU comparison aggregator: https://getdeploying.com/gpus
- Cheapest providers overview: https://northflank.com/blog/cheapest-cloud-gpu-providers
- RunPod Ollama deployment: https://docs.runpod.io/tutorials/serverless/run-ollama-inference
- Modal Ollama guide: https://modal.com/blog/how_to_run_ollama_article
- Vast.ai networking/ports: https://docs.vast.ai/documentation/instances/connect/networking
- Google Colab Ollama: https://medium.com/data-science-collective/unleash-the-power-of-ai-host-your-own-ollama-models-for-free-with-google-colab-0aac5f237a9f

## RunPod API Deep Dive — 2026-03-18

> **Summary:** RunPod offers two APIs (REST and GraphQL) plus a Python SDK for pod lifecycle
> management. Pods have stable HTTP proxy URLs (format: `https://{podId}-{port}.proxy.runpod.net`).
> No built-in auto-stop/idle detection — you must build it yourself. Serverless is an
> alternative that scales to zero and bills per-second, but requires a custom handler (not
> raw Ollama). Cold start times for pods are undocumented; for serverless, they depend on
> image size and model loading (mitigated by FlashBoot and active workers).

### 1. Pod Lifecycle Management via API

RunPod provides **two APIs** for pod management:

**REST API** (newer, simpler):
- Base URL: `https://rest.runpod.io/v1`
- Auth: `Authorization: Bearer YOUR_API_KEY` header
- Endpoints:
  - `POST /pods` — create a pod (returns 201 with pod object)
  - `GET /pods` — list all pods (extensive query filters available)
  - `GET /pods/{podId}` — get specific pod
  - `POST /pods/{podId}/start` — start a stopped pod (no body needed, returns 200)
  - `POST /pods/{podId}/stop` — stop a running pod (no body needed, returns 200)
  - `DELETE /pods/{podId}` — terminate/delete a pod (returns 204, no body)
  - `PATCH /pods/{podId}` — update pod configuration
  - `POST /pods/{podId}/restart` — restart a pod
  - `POST /pods/{podId}/reset` — reset a pod

**GraphQL API** (older, more fields available):
- Endpoint: `https://api.runpod.io/graphql?api_key=${YOUR_API_KEY}`
- Auth: API key passed as query parameter (not header)
- Key mutations:
  ```graphql
  # Create on-demand pod
  mutation {
    podFindAndDeployOnDemand(input: {
      cloudType: ALL
      gpuCount: 1
      volumeInGb: 40
      containerDiskInGb: 40
      minVcpuCount: 2
      minMemoryInGb: 15
      gpuTypeId: "NVIDIA RTX A6000"
      name: "my-pod"
      imageName: "runpod/pytorch"
      ports: "8888/http"
      volumeMountPath: "/workspace"
      env: [{ key: "JUPYTER_PASSWORD", value: "password" }]
    }) {
      id imageName env machineId machine { podHostId }
    }
  }

  # Create spot/interruptible pod
  mutation {
    podRentInterruptable(input: {
      bidPerGpu: 0.2
      cloudType: SECURE
      gpuCount: 1
      volumeInGb: 40
      containerDiskInGb: 40
      gpuTypeId: "NVIDIA RTX A6000"
      name: "my-spot-pod"
      imageName: "runpod/pytorch"
      ports: "8888/http"
    }) {
      id imageName machineId
    }
  }

  # Stop pod
  mutation { podStop(input: {podId: "YOUR_POD_ID"}) { id desiredStatus } }

  # Resume on-demand pod
  mutation {
    podResume(input: {podId: "YOUR_POD_ID", gpuCount: 1}) {
      id desiredStatus imageName machineId
    }
  }

  # Resume spot pod
  mutation {
    podBidResume(input: {podId: "YOUR_POD_ID", bidPerGpu: 0.2, gpuCount: 1}) {
      id desiredStatus
    }
  }

  # Terminate pod (permanent delete)
  mutation { podTerminate(input: {podId: "YOUR_POD_ID"}) }

  # List all pods
  query {
    myself {
      pods {
        id name runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
          gpus { id gpuUtilPercent memoryUtilPercent }
        }
      }
    }
  }

  # Get specific pod
  query { pod(input: {podId: "YOUR_POD_ID"}) { id name runtime { uptimeInSeconds } } }
  ```

**REST API create pod example (curl):**
```bash
curl -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpu-pod",
    "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
    "gpuCount": 1,
    "containerDiskInGb": 50,
    "ports": ["8888/http", "22/tcp"]
  }'
```

**REST API create pod response (201):**
```json
{
  "id": "xedezhzb9la3ye",
  "name": "my pod",
  "desiredStatus": "RUNNING",
  "publicIp": "100.65.0.119",
  "ports": ["8888/http", "22/tcp"],
  "costPerHr": 0.74,
  "gpu": {},
  "machine": {}
}
```

### 2. Cold Start Times

**For pods:** RunPod docs do not publish specific cold start numbers. Anecdotally:
- Resuming a stopped pod: depends on GPU availability. The pod retains its volume disk
  but GPU must be re-allocated, which can take seconds to minutes depending on demand.
  If the same GPU type is unavailable, the resume can fail.
- Creating a new pod from scratch: image pull time + container startup + any init scripts.
  For a standard PyTorch image, expect 30-120 seconds. Custom images vary.
- The `runtime` field in the pod query response includes `uptimeInSeconds` which can be
  used to track when a pod is actually ready.

**For serverless:** Cold starts occur when no active workers exist. The platform must:
  1. Start the container
  2. Load models into GPU memory
  3. Initialize the runtime
  Mitigation strategies: FlashBoot (enabled by default, retains worker state), model caching,
  setting active workers > 0 (eliminates cold starts but costs 20-30% of full rate 24/7).
  No specific cold start durations are published — they depend on image size and model size.

### 3. Auto-Stop / Idle Detection

**No built-in auto-stop for pods.** RunPod does not offer native idle detection or
auto-stop after N minutes of inactivity.

**DIY workaround from their docs:**
```bash
# Schedule a stop after 2 hours (run inside the pod)
sleep 2h; runpodctl stop pod $RUNPOD_POD_ID &
```

**For real idle detection, you must build it yourself:**
- Run a cron job or background script inside the pod
- Monitor GPU utilization, API request count, or custom activity signals
- Call `runpodctl stop pod $RUNPOD_POD_ID` or hit the API when idle threshold is reached

**Serverless has built-in idle timeout:** Default 5 seconds. Workers automatically scale
down to zero when no requests arrive within the idle timeout window. This is the key
advantage of serverless over pods for cost management.

### 4. Serverless Endpoints

**How it works:**
- You write a handler function (Python), build a Docker image, deploy to an endpoint
- RunPod manages workers (container instances) that process requests
- Workers auto-scale: 0 to max_workers based on queue depth
- Two scaling modes: queue delay (default, adds workers when wait > 4s) and request count

**Pricing:** Per-second billing (not per-request). Charged from worker start to worker stop,
rounded up to nearest second. Example rates:
- A4000 (16GB): $0.00016/sec flex ($0.576/hr)
- RTX 4090 (24GB): available but pricing varies
- A100 (80GB): higher tiers
- Active workers get 20-30% discount but run 24/7

**Endpoint configuration options:**
- Active workers: default 0 (scales to zero), set > 0 to eliminate cold starts
- Max workers: default 3, caps scaling and cost
- Idle timeout: default 5 seconds
- Execution timeout: default 600s (10 min), range 5s to 7 days
- Job TTL: default 24 hours
- FlashBoot: enabled by default, retains worker state for faster restarts
- GPU selection: can specify multiple GPU types as fallback priority list

**Can you run Ollama on serverless?**
Not directly. Serverless requires a custom handler function that processes `{"input": {...}}`
requests. You cannot just run `ollama serve` as-is. You would need to:
1. Build a Docker image that starts Ollama inside the container
2. Write a handler function that translates RunPod requests to Ollama API calls
3. Or use vLLM instead (RunPod has native vLLM serverless support with pre-built workers)

vLLM on serverless is the recommended approach for LLM inference — it has pre-built
templates, supports the OpenAI API format, and handles most HuggingFace models.

**API for serverless endpoints:**
- Base URL: `https://api.runpod.ai/v2/ENDPOINT_ID/`
- Auth: `Authorization: Bearer RUNPOD_API_KEY`
- Operations:
  - `POST /runsync` — synchronous (wait for result, timeout ~30s typical)
  - `POST /run` — asynchronous (returns job ID, poll for result)
  - `GET /status/{jobId}` — check job status
  - `GET /stream/{jobId}` — stream results incrementally
  - `POST /cancel/{jobId}` — cancel a job
  - `GET /health` — endpoint health statistics
- Rate limits: `/runsync` 2,000 req/10s, `/run` 1,000 req/10s

### 5. Python SDK

**Package:** `runpod` (on PyPI)
**Install:** `pip install runpod`
**Requires:** Python 3.8+
**GitHub:** https://github.com/runpod/runpod-python

**Pod management example:**
```python
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")

# List all pods
pods = runpod.get_pods()

# Get specific pod
pod = runpod.get_pod("pod_id_here")

# Create pod with GPU
pod = runpod.create_pod(
    name="my-ollama-pod",
    image_name="runpod/pytorch",
    gpu_type_id="NVIDIA GeForce RTX 4090",
    gpu_count=1,
    volume_in_gb=40,
    container_disk_in_gb=50,
    ports="11434/http",
    volume_mount_path="/workspace",
    env={"OLLAMA_HOST": "0.0.0.0"},
    cloud_type="ALL",           # ALL, SECURE, or COMMUNITY
    support_public_ip=True,
    min_vcpu_count=2,
    min_memory_in_gb=15,
    network_volume_id="vol_xxx",  # optional, for persistent storage
)

# Stop pod (preserves volume, stops GPU billing)
runpod.stop_pod(pod["id"])

# Resume pod (re-allocates GPU)
runpod.resume_pod(pod["id"], gpu_count=1)

# Terminate pod (permanent delete)
runpod.terminate_pod(pod["id"])
```

**Serverless endpoint example:**
```python
import runpod

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("ENDPOINT_ID")

# Synchronous call (blocks until result)
result = endpoint.run_sync({"prompt": "Hello world"})

# Asynchronous call
run_request = endpoint.run({"prompt": "Hello world"})
print(run_request.status())   # "IN_QUEUE", "IN_PROGRESS", "COMPLETED"
print(run_request.output())   # blocks until result ready
```

### 6. Webhooks / Status Callbacks

**For serverless endpoints only (not pods):**
```json
{
  "input": {"prompt": "Hello"},
  "webhook": "https://your-server.com/webhook"
}
```
- RunPod POSTs to your webhook URL when the job completes
- Your webhook must return HTTP 200
- Retries: up to 2 additional attempts with 10-second delays on failure
- Webhook request body format is not fully documented

**For pods:** No webhook/callback mechanism. To know when a pod is ready, you must poll:
- REST API: `GET /v1/pods/{podId}` and check `desiredStatus`
- GraphQL: query the pod and check `runtime` fields
- Or use the Python SDK: `runpod.get_pod(pod_id)` in a loop

### 7. Port Exposure / HTTP Proxy

**HTTP proxy (recommended for web services like FastAPI):**
- Configure ports in pod creation: `ports: "8888/http,11434/http"`
- Access URL format: `https://{POD_ID}-{INTERNAL_PORT}.proxy.runpod.net`
- Example: pod `abc123xyz` with port 11434 -> `https://abc123xyz-11434.proxy.runpod.net`
- **URL is stable** — the pod ID doesn't change, so the URL persists across restarts
- All traffic uses HTTPS regardless of internal service config
- 100-second timeout enforced by Cloudflare (long-running requests need TCP instead)
- Route: User -> Cloudflare -> RunPod Load Balancer -> Pod
- Services must bind to `0.0.0.0`, not localhost

**TCP exposure (for non-HTTP protocols or long connections):**
- Configure in "Expose TCP Ports" field
- External port mappings change on pod restart (not stable)
- Symmetric port mapping available for ports > 70000
- Access assigned port via env var: `$RUNPOD_TCP_PORT_70000`

**Stability across restarts:**
- HTTP proxy URLs: **stable** (based on pod ID which doesn't change)
- TCP ports: **not stable** (external port mappings change on restart)
- Community Cloud pod IPs: change after migration/restart
- Secure Cloud pod IPs: stable public IPs, but TCP port mappings still change

**Key constraint:** Endpoints become publicly accessible — implement authentication yourself.

### Sources

- RunPod REST API reference: https://docs.runpod.io/api-reference/pods/POST/pods
- RunPod GraphQL API: https://docs.runpod.io/sdks/graphql/manage-pods
- GraphQL spec: https://graphql-spec.runpod.io
- Python SDK: https://github.com/runpod/runpod-python
- Pod management: https://docs.runpod.io/pods/manage-pods
- Port exposure: https://docs.runpod.io/pods/configuration/expose-ports
- Serverless overview: https://docs.runpod.io/serverless/overview
- Serverless pricing: https://docs.runpod.io/serverless/pricing
- Serverless endpoints: https://docs.runpod.io/serverless/endpoints/send-requests
- Serverless workers: https://docs.runpod.io/serverless/workers/overview
- Endpoint configuration: https://docs.runpod.io/serverless/endpoints/endpoint-configurations
- Ollama on RunPod pods: https://docs.runpod.io/tutorials/pods/run-ollama
- vLLM on serverless: https://docs.runpod.io/serverless/vllm/get-started

## RunPod GPU Type Selection, Fallback, and Availability — 2026-03-18

> **Summary:** RunPod's REST API supports specifying multiple GPU types with ordered fallback
> via `gpuTypeIds` (array) + `gpuTypePriority` ("availability" or "custom"). The GraphQL API
> is older and uses a single `gpuTypeId` string (no multi-type fallback). GPU availability
> can be queried via GraphQL `gpuTypes` query with `lowestPrice.stockStatus`. When resuming
> a stopped pod, the GPU is tied to the original machine — if that GPU is rented by someone
> else, the pod starts with zero GPUs. No built-in "cheapest GPU meeting spec X" feature
> exists, but you can build it by querying all GPU types and filtering client-side.

### 1. Can you specify multiple GPU types when creating a pod?

**REST API: Yes.** The `gpuTypeIds` parameter is an array of strings. You can pass multiple
GPU type IDs and they function as an ordered fallback list.

```json
{
  "gpuTypeIds": ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"],
  "gpuTypePriority": "custom",
  "gpuCount": 1
}
```

The `gpuTypePriority` parameter controls how the list is used:
- `"availability"` (default) — RunPod picks from your list based on current availability
  (may not follow your order)
- `"custom"` — RunPod tries GPU types strictly in the order you specified

**GraphQL API: No.** The `podFindAndDeployOnDemand` and `podRentInterruptable` mutations
use `gpuTypeId` (singular string, not array). You can only specify one GPU type per request.
To get fallback behavior via GraphQL, you'd need to query availability first, then pick
the best available type yourself.

**Python SDK: No multi-type support.** `runpod.create_pod()` accepts `gpu_type_id` (singular
string). The SDK wraps GraphQL, not the REST API.

### 2. What happens when you resume a pod and the GPU is unavailable?

Pods are tied to a specific physical machine. When you stop a pod, its GPU becomes available
to other users. If someone else rents that GPU while your pod is stopped:

- **The pod starts with zero GPUs.** It does not queue, fail with an error, or offer
  alternatives. It simply starts without GPU access (CPU-only).
- You get limited CPU resources — not suitable for compute tasks.
- **No automatic fallback** to a different GPU type or machine.

**Three options when this happens:**
1. Start with zero GPUs just to access your data/files
2. Wait and retry later (no guarantee of when GPU becomes free)
3. Terminate the pod and create a new one — the new pod will be scheduled on any machine
   with an available GPU of your chosen type

**Mitigation:** Use network volumes. They decouple data from the physical machine, so you
can terminate and redeploy to a new machine without losing data.

### 3. Can you query GPU availability via API?

**GraphQL: Yes.** The `gpuTypes` query returns availability information.

Query all GPU types (no filter):
```graphql
query {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    securePrice
    communityPrice
    lowestPrice(input: { gpuCount: 1 }) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      maxUnreservedGpuCount
      availableGpuCounts
    }
  }
}
```

Query a specific GPU type:
```graphql
query {
  gpuTypes(input: { id: "NVIDIA RTX A4000" }) {
    id
    displayName
    memoryInGb
    lowestPrice(input: {
      gpuCount: 1
      minMemoryInGb: 8
      minVcpuCount: 2
      secureCloud: true
    }) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      maxUnreservedGpuCount
      availableGpuCounts
    }
  }
}
```

Key fields:
- `stockStatus` — values like "High", "Medium", "Low", or "None" (availability indicator)
- `availableGpuCounts` — array of quantities in stock (e.g., `[1, 2, 4]`)
- `maxUnreservedGpuCount` — maximum GPUs of this type available right now
- `minimumBidPrice` — lowest spot price
- `uninterruptablePrice` — on-demand price

**REST API: No dedicated GPU query endpoint.** The REST API does not have a GET endpoint
for listing available GPU types. GPU availability querying is GraphQL-only.

### 4. Does RunPod have "cheapest GPU meeting spec X" feature?

**No built-in feature.** There is no API parameter like `minVramGb: 16` that returns the
cheapest available GPU meeting that spec.

**However, you can build it yourself:**
1. Query all `gpuTypes` via GraphQL (returns `memoryInGb` and `lowestPrice`)
2. Filter client-side by your VRAM requirement
3. Sort by price
4. Pick the cheapest one with `stockStatus` != "None"
5. Pass that GPU type ID to the pod creation API

The `lowestPrice` subquery accepts `minMemoryInGb` as a filter, but this filters the
*machines* available for a given GPU type (system RAM, not VRAM). VRAM is a property of
the GPU type itself (`memoryInGb` on the `GpuType` object).

### 5. Pod creation API parameter details

**REST API — POST /pods (full GPU-related parameters):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpuTypeIds` | `string[]` | — | Ordered list of acceptable GPU types |
| `gpuTypePriority` | `"availability" \| "custom"` | `"availability"` | How to use the GPU list |
| `gpuCount` | `int` | `1` | Number of GPUs (min: 1) |
| `minVCPUPerGPU` | `int` | `2` | Min vCPUs per GPU |
| `minRAMPerGPU` | `int` | `8` | Min system RAM (GB) per GPU |
| `allowedCudaVersions` | `string[]` | — | Filter by CUDA version (e.g., `["12.0", "12.1"]`) |
| `computeType` | `"GPU" \| "CPU"` | `"GPU"` | Pod type |
| `cloudType` | `"SECURE" \| "COMMUNITY"` | `"SECURE"` | Cloud tier |
| `dataCenterIds` | `string[]` | all 26 DCs | Restrict to specific data centers |
| `dataCenterPriority` | `"availability" \| "custom"` | `"availability"` | DC selection strategy |
| `countryCodes` | `string[]` | — | Restrict by country |
| `interruptible` | `bool` | `false` | Spot pricing (can be preempted) |

**GraphQL — podFindAndDeployOnDemand (GPU-related parameters):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpuTypeId` | `string` | Single GPU type (e.g., `"NVIDIA RTX A6000"`) |
| `gpuCount` | `int` | Number of GPUs |
| `minVcpuCount` | `int` | Min vCPUs |
| `minMemoryInGb` | `int` | Min system RAM (GB) |
| `cloudType` | `enum` | `ALL`, `SECURE`, or `COMMUNITY` |
| `allowedCudaVersions` | `string[]` | CUDA version filter |

**GPU type ID strings** are the full display names:
- `"NVIDIA GeForce RTX 4090"` (24GB)
- `"NVIDIA GeForce RTX 3090"` (24GB)
- `"NVIDIA RTX A4000"` (16GB)
- `"NVIDIA RTX A6000"` (48GB)
- `"NVIDIA A40"` (48GB)
- `"NVIDIA L40S"` (48GB)
- `"NVIDIA H100 80GB HBM3"` (80GB)
- `"NVIDIA A100 80GB PCIe"` (80GB)
- `"NVIDIA A100-SXM4-80GB"` (80GB)
- And ~30+ others (full list at https://docs.runpod.io/references/gpu-types)

**Serverless endpoints** have a separate GPU selection mechanism using pooled categories
like `AMPERE_16`, `AMPERE_24`, `ADA_24`, `HOPPER_141` — these group multiple GPU models
by architecture and VRAM tier. You can select multiple pools as fallback.

### 6. REST API resume endpoint

`POST /pods/{podId}/start` accepts **only the podId**. No request body. You cannot change
the GPU type when resuming — the pod retains its original configuration and is tied to its
original machine.

The GraphQL `podResume` mutation also cannot change the GPU type — it accepts `podId`,
`gpuCount`, and optionally `allowedCudaVersions`.

**Implication:** If you want GPU-type flexibility on resume, the correct pattern is:
terminate + create new pod (with network volume for data persistence).

### Sources

- REST API pod creation: https://docs.runpod.io/api-reference/pods/POST/pods
- GraphQL manage pods: https://docs.runpod.io/sdks/graphql/manage-pods
- GraphQL spec: https://graphql-spec.runpod.io
- GPU types reference: https://docs.runpod.io/references/gpu-types
- Zero GPU troubleshooting: https://docs.runpod.io/references/troubleshooting/zero-gpus
- REST API blog post: https://www.runpod.io/blog/runpod-rest-api-gpu-management
- Serverless endpoint config: https://docs.runpod.io/serverless/endpoints/endpoint-configurations
- runpodctl GPU issues: https://github.com/runpod/runpodctl/issues/189

## Document Characterization for RAG Configuration Selection — 2026-03-19

> **Summary:** The idea of analyzing document-level features (topic density, NER density,
> semantic clustering, etc.) BEFORE running a RAG pipeline to predict optimal configuration
> is a genuinely novel intersection. No single paper does exactly this. However, several
> closely related research threads exist that collectively validate the approach and provide
> building blocks. The strongest prior art is METIS (Microsoft, 2025) and RAGSmith (2025),
> which optimize RAG configuration but focus on query-side complexity, not document-side
> features. The gap — using document-level metrics to predict RAG configuration — is real
> and publishable.

### 1. RAG Configuration Adaptation Systems (closest prior art)

**METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation**
- Authors: Microsoft Research
- Venue: ACM SOSP 2025 (top systems conference)
- Year: 2025
- Key finding: First RAG system that jointly schedules queries and adapts RAG configurations
  (number of retrieved chunks, synthesis method) per-query. Uses "query profiles" to filter
  out undesirable configurations. Reduces latency 1.64-2.54x without quality loss.
- **Relevance:** METIS adapts based on QUERY complexity (simple yes/no vs. deep "why"
  questions), NOT document properties. The document corpus is treated as fixed. This is the
  gap — what if you also profiled the documents?
- Paper: https://arxiv.org/abs/2412.10543
- Published version: https://dl.acm.org/doi/10.1145/3731569.3764855

**RAGSmith: A Framework for Finding the Optimal Composition of RAG Methods Across Datasets**
- Year: 2025
- Key finding: Treats RAG design as architecture search over 9 technique families and
  46,080 feasible pipeline configurations. Uses evolutionary (genetic) search to find optimal
  RAG configuration per dataset. Finds configurations that outperform naive RAG by +3.8% avg
  (up to +12.5% retrieval, +7.5% generation). Key insight: "failure modes are domain-specific —
  informality/coreference in chat logs, long-tail jargon in biomed, morphology in Turkish,
  hierarchical structure in manuals."
- **Relevance:** RAGSmith optimizes per DATASET but through brute-force search, not by
  analyzing document features. The paper explicitly acknowledges that optimal configuration
  is corpus-dependent but doesn't predict it from document characteristics — it discovers it
  empirically. A document characterization approach could shortcut this search.
- Paper: https://arxiv.org/abs/2511.01386

**Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity**
- Authors: Jeong et al.
- Venue: NAACL 2024
- Key finding: Trains a query complexity classifier (small LM) to route queries to one of
  three strategies: no retrieval (simple), single-step RAG (moderate), multi-step iterative
  RAG (complex). Labels derived from actual model outcomes.
- **Relevance:** Purely query-side adaptation. The classifier looks at the question, not the
  documents. Document-side complexity is not considered.
- Paper: https://arxiv.org/abs/2403.14403
- Published version: https://aclanthology.org/2024.naacl-long.389/

### 2. Document Routing and Expert Selection (document-aware systems)

**MODE: Mixture of Document Experts for RAG**
- Author: Rahul Anand
- Year: 2025
- Key finding: Replaces fine-grained vector search with cluster-and-route mechanism.
  Documents are organized into semantically coherent clusters ("document experts") using
  hierarchical clustering on embeddings. Queries routed to the best cluster via centroid
  matching. Retrieval cost O(Md) where M = number of clusters, independent of corpus size.
  Matches or exceeds traditional RAG quality on HotpotQA and SQuAD.
- **Relevance:** This IS document-aware — it clusters documents by topic and routes queries
  to topic clusters. But it doesn't characterize documents for configuration selection; it
  characterizes them for retrieval routing. The clustering analysis (semantic coherence of
  clusters, cluster count) could be repurposed as document-level features.
- Paper: https://arxiv.org/abs/2509.00100
- Code: https://github.com/rahulanand1103/mode

**ExpertRAG: Efficient RAG with Mixture of Experts**
- Year: 2025
- Key finding: Integrates MoE architecture with RAG. Dynamic retrieval gating mechanism
  decides whether to retrieve or rely on parametric knowledge, per-query. Different experts
  handle different topical domains — a science question routes to a "science expert" and
  triggers retrieval from scientific literature, while commonsense queries use general expert
  without retrieval.
- **Relevance:** Expert routing is topic-aware but the routing is query-driven. The document
  corpus is pre-organized but not characterized for configuration prediction.
- Paper: https://arxiv.org/abs/2504.08744

**MixRAG: Mixture-of-Experts Retrieval-Augmented Generation**
- Year: 2025
- Key finding: Multiple specialized graph retrievers with a dynamic routing controller for
  diverse query intents in textual graph understanding.
- Paper: https://arxiv.org/abs/2509.21391

### 3. Entity Density as a Document Metric

**From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting**
- Authors: Griffin Adams et al. (Columbia, Salesforce, MIT)
- Year: 2023
- Key finding: Entity density (entities per token) used as a proxy for summary informativeness.
  Iteratively adds entities to summaries without increasing length. Human annotations show
  optimal density matches human-written summaries — too sparse is uninformative, too dense is
  hard to follow.
- **Relevance:** Directly validates entity density as a measurable, meaningful text metric.
  Could be applied to RAG source documents: high entity density documents may need different
  chunking (smaller chunks to avoid entity overflow) or different retrieval strategies
  (entity-aware retrieval). This paper operationalizes the metric.
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11419567/
- Code: https://github.com/richawo/chain-of-density

### 4. Topic Modeling for Document Characterization

**TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction**
- Authors: Bougouin, Boudin, Daille
- Venue: IJCNLP 2013
- Key finding: Clusters candidate keyphrases into topics using hierarchical agglomerative
  clustering, then ranks topics via graph-based methods. The number and distribution of
  topic clusters characterizes document structure.
- **Relevance:** TopicRank provides a method to count/characterize topics within a document.
  Topic count per N tokens = topic density. This is a direct building block for document
  characterization. TopicLPRank (2022) improves it with position information.
- Paper: https://aclanthology.org/I13-1062/

**Topic-Enriched Embeddings to Improve Retrieval Precision in RAG Systems**
- Year: 2025
- Key finding: Integrates TF-IDF, LSA (dimensionality-reduced semantic structure), and LDA
  (probabilistic topic mixtures) into sentence embeddings. Topic enrichment improves both
  clustering coherence and retrieval effectiveness (Precision@k, Recall@k, F1).
- **Relevance:** Directly connects topic analysis to RAG retrieval quality. Shows that topic
  information is useful signal for RAG, not just for document characterization.
- Paper: https://arxiv.org/abs/2601.00891

### 5. Embedding-Based Document Diversity Analysis

**Clustering for RAG Chunking**
- Multiple sources (Towards Data Science, Rohan Paul, etc.)
- Key approaches: K-means on chunk embeddings, spectral clustering using cosine similarity
  matrices, hierarchical agglomerative clustering. Cluster count and cluster spread measure
  semantic diversity within a document.
- **Relevance:** The techniques for clustering chunk embeddings already exist. The novel step
  is using cluster statistics (count, variance, inter-cluster distance) as document-level
  features that predict RAG configuration needs.

**RAGxplorer: Visualizing Document Chunks in the Embedding Space**
- Source: OpenAI Community
- Key idea: Visualization tool for chunk embeddings. Shows how document chunks distribute
  in embedding space — tight clusters vs. scattered distributions.
- **Relevance:** Visual validation of the concept. Documents with tight embedding clusters
  (homogeneous content) likely behave differently under RAG than documents with scattered
  embeddings (diverse content).

### 6. Query Performance Prediction (related IR tradition)

**Query Performance Prediction (QPP)**
- Extensive IR research tradition (20+ years)
- Key concepts: Pre-retrieval predictors (AvgIDF, MaxIDF, SCQ — analyze query + collection
  statistics before retrieval), post-retrieval predictors (WIG, NQC, UQC — analyze score
  distributions after retrieval). Both predict retrieval effectiveness without relevance
  judgments.
- **Relevance:** QPP is the closest methodological analog. It predicts performance from
  statistical features of queries and collections. The document characterization idea
  extends this: use statistical features of documents (not queries) to predict optimal
  configuration (not just performance). QPP could be adapted from "predict if this query
  will work" to "predict which RAG config works best for this document set."
- Key references:
  - Carmel & Yom-Tov (2010). "Estimating the Query Difficulty for Information Retrieval."
  - Hauff et al. (2008). Pre-retrieval predictors survey. CIKM.
  - Survey: https://www.sciencedirect.com/science/article/abs/pii/S0306437905000955

### 7. Self-RAG and Corrective RAG (adaptive retrieval decisions)

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
- Authors: Asai et al.
- Year: 2023
- Key finding: Trains LLM to decide when to retrieve (on-demand), generate, and self-critique
  using special "reflection tokens." Can retrieve multiple times or skip retrieval entirely.
- **Relevance:** Adaptation happens at inference time based on generation quality, not
  document properties. But the principle — that different situations need different retrieval
  strategies — is directly relevant.
- Paper: https://openreview.net/forum?id=hSyW5go0v8

**Corrective RAG (CRAG)**
- Key finding: Lightweight retrieval evaluator (fine-tuned T5-large) scores document relevance
  and triggers corrective actions: Correct, Incorrect, or Ambiguous. Falls back to web search
  when retrieval quality is low.
- **Relevance:** Document-level quality scoring exists but is applied post-retrieval, not
  pre-pipeline. The idea of scoring documents BEFORE choosing a RAG configuration is the gap.

### 8. Document Complexity and Text Difficulty Metrics (NLP/education research)

**Text Difficulty Prediction Research**
- Key finding from multiple papers: Document difficulty can be predicted using features
  including: non-narrativity, referential cohesion, situation model cohesion, syntactic
  complexity, word abstractness, academic vocabulary frequency, concreteness, word familiarity.
  These explain ~23-34% of variance in readability, with NLP features improving over
  traditional readability formulas.
- **Relevance:** These text difficulty features are well-validated and could serve as
  additional document characterization features. A document that is "difficult" by these
  metrics may need different RAG treatment (more context, bigger model, different chunking).
- Key reference: "Text-based Question Difficulty Prediction: A Systematic Review" (Springer,
  2023). https://link.springer.com/article/10.1007/s40593-023-00362-1

### 9. Chunking Adaptation Based on Document Properties

**Dynamic/Adaptive Chunking**
- Multiple industry sources (NVIDIA, Databricks, Microsoft, etc.)
- Key finding: Some systems dynamically allocate chunk sizes (200-800 tokens) based on
  content analysis. Benchmarks suggest up to 40% improvement in retrieval accuracy vs.
  fixed-size chunking. Document structure analysis identifies whether layout-aware chunking
  (structured docs) or semantic chunking (narrative content) works better.
- **Relevance:** This is the closest existing practice to the document characterization idea,
  but it operates at the chunk level, not the document level, and adapts only chunking
  strategy, not full RAG configuration.

**Chunk Twice, Embed Once (Chemistry-Aware RAG)**
- Year: 2025
- Key finding: Domain-specific chunking for chemistry documents. Systematic study of
  segmentation and representation trade-offs.
- **Relevance:** Domain-specific document properties drive different chunking strategies.
  Validates that document type matters for RAG configuration.
- Paper: https://arxiv.org/abs/2506.17277

### 10. RAPTOR: Document Structure as Retrieval Feature

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
- Authors: Sarthi et al. (Stanford)
- Year: 2024
- Key finding: Recursively clusters chunks, generates summaries, builds tree hierarchy.
  Retrieval at different tree levels handles different query types (detail vs. thematic).
  20% improvement on QuALITY benchmark with GPT-4.
- **Relevance:** RAPTOR implicitly characterizes document structure by building a tree.
  Documents with deep trees have hierarchical content; shallow trees have flat content.
  Tree depth/breadth could be document features.
- Paper: https://arxiv.org/abs/2401.18059

### 11. R2AG: Using Retrieval Information to Improve Generation

**R2AG: Incorporating Retrieval Information into Retrieval Augmented Generation**
- Venue: EMNLP 2024 Findings
- Key finding: Captures retrieval features (relevance scores, precedent similarity,
  neighbor similarity) and feeds them to the generator via a R2-Former module. Reduces
  irrelevant retrievals by 15% with only 0.8% inference time increase.
- **Relevance:** Uses retrieval-time document features to improve generation. Not
  pre-pipeline document characterization, but demonstrates that document-level signals
  (similarity patterns, relevance distributions) carry useful information for RAG.
- Paper: https://aclanthology.org/2024.findings-emnlp.678/
- Code: https://github.com/YeFD/RRAG

### Synthesis: The Research Gap

**What exists:**
- Query-side complexity classification for RAG strategy selection (Adaptive-RAG, METIS)
- Per-dataset RAG configuration search via brute force (RAGSmith)
- Document clustering for retrieval routing (MODE, ExpertRAG)
- Entity density as a text metric (Chain of Density)
- Topic modeling for keyphrase/topic extraction (TopicRank)
- Embedding clustering for chunk organization
- Query performance prediction from collection statistics (QPP tradition)
- Text difficulty metrics from education/readability research
- Adaptive chunking based on content analysis

**What does NOT exist (the gap):**
- A system that computes document-level features (topic density, NER density, embedding
  cluster count/variance, readability metrics) and uses those features to PREDICT which
  RAG configuration (chunk size, retrieval strategy, model size, reranking approach) will
  perform best — BEFORE running the pipeline.
- A predictive model that maps document characteristics to RAG configuration parameters.
- Empirical evidence for which document features are most predictive of RAG performance
  under different configurations.

**Why this is novel:**
- METIS and Adaptive-RAG adapt on the query side, not the document side.
- RAGSmith finds optimal configs empirically per dataset but doesn't predict from features.
- MODE clusters documents but for routing, not configuration selection.
- QPP predicts performance but doesn't recommend configuration changes.
- The gap is the PREDICTION step: document features → optimal RAG config.

**Candidate document-level features (from this research):**
1. Topic density (topics per N tokens, via TopicRank or LDA)
2. Entity density (entities per token, via NER, validated by Chain of Density)
3. Embedding cluster count/variance (via KNN/spectral clustering on chunk embeddings)
4. Readability metrics (Flesch-Kincaid, plus NLP-augmented features)
5. Vocabulary diversity (type-token ratio, hapax legomena ratio)
6. Structural complexity (heading depth, table count, list density)
7. Cross-reference density (citation/reference frequency)
8. Semantic coherence (average cosine similarity between consecutive chunks)
