# Production RAG System — Legal Domain: Final Report

## 1. Problem Statement

Build a production-style hybrid RAG system over Indian Supreme Court judgments
(2018–2022) with rigorous, multi-axis evaluation covering retrieval quality,
generation faithfulness, and hallucination rate — benchmarked against a
human-verified question-answering test set, and hardened to run as a service
rather than a notebook.

## 2. Architecture

- **Corpus**: 4,140 Supreme Court judgments, full text extracted from PDFs
  (pdfplumber, parallelised and checkpointed extraction)
- **Chunking**: three strategies compared — fixed-size, recursive
  (paragraph-aware), semantic (embedding-similarity merged)
- **Indexing**: hybrid — BM25 (sparse) + bge-large-en-v1.5 (dense), stored in
  Qdrant local embedded mode
- **Retrieval**: BM25 top-50 + dense top-50 → Reciprocal Rank Fusion (k=60)
- **Reranking**: bge-reranker-v2-m3 cross-encoder → top-5 to generation
- **Generation**: Ollama (qwen2.5:7b-instruct), citation-forcing prompt,
  score-based sufficiency guard
- **Evaluation**: retrieval metrics (Recall@k, MRR, nDCG) + RAGAS + a custom
  claim-level hallucination checker
- **Serving**: FastAPI, semantic cache, SQLite observability, /metrics endpoint

## 3. Eval Set Construction

Source: IndicLegalQA (10,000 QA pairs), matched against the corpus by normalised
fuzzy case-name matching (rapidfuzz).

A match threshold of ≥95 was required after discovering that the 85–95 band
contained real mismatches — e.g. "Avtar Singh vs UOI" matching a different case
with a similar name pattern. Final set: **250 QA pairs**, capped at 2 questions
per document (219 unique documents), manually spot-checked.

## 4. Ablation Axes 1 & 2 — Chunking and Reranking (n=250)

| Strategy | Reranker | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| fixed | bge-reranker-v2-m3 | 0.592 | 0.712 | 0.736 | 0.764 | 0.643 | 0.673 |
| fixed | none | 0.504 | 0.648 | 0.696 | 0.732 | 0.569 | 0.607 |
| recursive | bge-reranker-v2-m3 | 0.580 | 0.692 | 0.728 | 0.756 | 0.633 | 0.660 |
| recursive | none | 0.536 | 0.636 | 0.696 | 0.744 | 0.587 | 0.620 |
| **semantic** | **bge-reranker-v2-m3** | **0.612** | **0.712** | **0.740** | **0.772** | **0.657** | **0.684** |
| semantic | none | 0.532 | 0.640 | 0.684 | 0.744 | 0.587 | 0.613 |

**Winning configuration: semantic chunking + bge-reranker-v2-m3.** The reranker
gives a consistent +0.05 to +0.09 Recall@1 lift across every chunking strategy —
a reproducible signal, not noise.

## 5. Ablation Axis 3 — Fusion Weighting

| Mode | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| BM25 only | 0.524 | 0.652 | 0.700 | 0.583 |
| Dense only | 0.440 | 0.560 | 0.604 | 0.494 |
| Hybrid (RRF) | 0.532 | 0.640 | 0.684 | 0.586 |

**Negative result, reported as found:** hybrid only marginally outperforms pure
BM25 on this corpus, within noise on R@1 and MRR. Dense retrieval alone is
clearly weakest. Legal text's reliance on exact citations and section numbers
favours keyword matching more than typical semantic-search domains. This tempers
the common assumption that hybrid retrieval always clearly beats its components.

## 6. Ablation Axis 4 — Top-k Sensitivity

Comparing generation_top_k = 5 vs 10: context grew from ~2,777 to ~5,088 average
input tokens (+83%) and latency by +28%, while guardrail-declined questions fell
only from 8/20 to 7/20. Diminishing returns on additional context at this chunk
granularity; k=5 retained.

*Documented limitation:* the k=20 arm was capped by rerank_top_k=10 and therefore
did not test a genuinely distinct value.

## 7. Ablation Axis 5 — Prompt Variant (n=100 per variant)

Strict (citation-forcing) vs loose prompt, semantic + reranker, score guard left
ON in both arms so the prompt is the only variable.

| Metric | Strict | Loose | Delta |
|---|---|---|---|
| Guard declined (score-based) | 1% | 1% | 0 |
| **LLM self-declined (prompt-driven)** | **17%** | **0%** | **+17 pts** |
| Answered | 82 | 99 | −17 |
| **Citation rate among answered** | **100.0%** (82/82) | 94.9% (94/99) | +5.1 pts |
| Avg input tokens | 2,783 | 2,692 | +3.3% |

Three findings:

1. The prompt drives essentially **all** abstention behaviour (+17 points); the
   recalibrated score guard fires on only 1 in 100 real eval questions.
2. The strict prompt achieves **perfect citation compliance**. The loose prompt
   still cites 94.9% of the time unprompted, so citation is largely intrinsic to
   the model — the instruction closes the final 5 points.
3. Token cost is effectively identical between variants. **The guardrail is free
   in compute terms.**

## 8. Generation & Faithfulness Results (winning config)

| Metric | Score | n |
|---|---|---|
| Faithfulness (RAGAS) | 0.754 | 60 |
| Answer Relevancy (RAGAS) | 0.690 | 60 |
| Context Precision (RAGAS) | 0.614 | 60 |
| Context Recall (RAGAS) | 0.617 | 60 |
| Hallucination Rate (custom claim-checker) | 20.2% | 199 |

Two independent methods — RAGAS faithfulness and the custom claim-level checker —
converge on a similar picture (~75–80% of content well-supported), giving
confidence in both measurements rather than relying on a single tool.

## 9. The Guardrail: A Complete Investigation

This is the design choice the system is built around: knowing when *not* to
answer. It took three stages to establish that it actually works.

### 9.1 The guard was never firing

An audit found the coded sufficiency threshold (−5.0) never triggered:
bge-reranker-v2-m3 produces small positive scores in this setup, not the large
negative ones originally assumed. The apparent guardrail behaviour in early
testing was the **LLM self-declining per its system prompt** — a less reliable
mechanism than a deterministic score check.

Recalibrated to **0.10** using 5 known-good vs 5 known-bad questions (clean
separation: GOOD averaged 0.41–0.93, BAD averaged 0.004–0.023). Re-verified at
5/5 pass rate with the corrected deterministic threshold.

### 9.2 Quantifying which mechanism does the work

Axis 5 (§7) separated the two mechanisms directly: the score guard accounts for
1 percentage point of abstention, the prompt for 17. The §9.1 conclusion was
correct — the prompt was doing the guardrail's job all along, and is still doing
most of it.

### 9.3 Proving the abstention is *targeted*

Refusing questions is only valuable if the refused questions are the ones that
would have gone wrong. Tested directly on the **17 contested questions** where
strict declined but loose answered anyway:

| | Contested | Matched control | Earlier baseline |
|---|---|---|---|
| Questions | 17 | 17 | 199 |
| Claims | 141 | 117 | — |
| Unsupported | 43 | 19 | — |
| **Hallucination rate** | **30.5%** | **16.2%** | 20.2% |
| Per-question mean | 0.340 | 0.164 | 0.195 |
| **Per-question median** | **0.333** | **0.000** | 0.167 |

Mann-Whitney U, one-sided, on per-question rates:
- contested vs matched control (same run, same config, same judge): **p = 0.0084**
- contested vs n=199 baseline: **p = 0.0027**

Significance was tested on per-question rates rather than claim-level proportions
because claims within a single answer are not independent; a naive claim-level
interval would have overstated confidence.

**The median control answer contains zero unsupported claims. The median question
the strict prompt refused has a third of its claims unsupported.**

The guardrail does not decline indiscriminately — it declines precisely the
questions where answering produces unsupported content. This is the empirical
backing for the system's central claim to be grounded rather than a
hallucination machine.

## 10. Failure Analysis

- **Retrieval-miss failures**: vague, generically-phrased auto-generated
  questions ("What broader issue did the Court address...") retrieve
  plausible-but-wrong documents. The model then faithfully cites whatever it
  received — a retrieval failure, not a hallucination. Attributable to
  IndicLegalQA's question style rather than the pipeline. A follow-up
  quantification found only 0.8% of the eval set qualifies as vague, correcting
  an earlier theory that this effect was large.
- **Correctly-answered cases**: specific, well-formed questions retrieve the
  correct document and generate answers semantically aligned with gold answers,
  with correct citations.

See `results/failure_taxonomy.md` for categorised examples.

## 11. Production Hardening

### Caching
File-persisted semantic cache (query embedding + cosine similarity,
threshold 0.95), substituted for Redis due to no-root access on the shared HPC —
the same reasoning behind Qdrant local mode instead of Docker. Verified: an
identical repeated query dropped from **33.83s to 0.047s** on cache hit.

### Cost & Latency Observability
SQLite-backed per-query logging and a `/metrics` endpoint reporting p50/p95
latency, token totals and cache hit-rate. Generation runs on self-hosted Ollama,
so per-token dollar cost is $0; token counts are tracked as a compute-cost proxy
for comparing configurations.

### Reproducibility
`results/build_manifest.json` hashes the corpus, eval set and all three chunk
files together with model names/versions and the git commit, so any reported
result traces back to exactly what produced it. Dependencies are pinned in
`requirements.txt` (notably `ragas==0.3.9`).

### Deployment constraints
The original design specified `docker-compose up`. This was not achievable
without root on the shared HPC, driving three consistent substitutions: Qdrant
local embedded mode instead of Qdrant-in-Docker, a file-persisted semantic cache
instead of Redis, SQLite instead of Postgres. `start_api.sh` is the documented
entry point.

### Infrastructure finding: I/O contention
Qdrant local-mode cold start was measured varying from ~15–30s on a quiet day to
5+ minutes under load (24 concurrent users, `/home` at 90% capacity). The API
startup script was changed to poll for readiness rather than sleep a fixed
interval — a realistic demonstration of why production deployments need readiness
probes rather than fixed timeouts.

## 12. Engineering Process and Negative Results

- **Context-truncation bug**: both RAGAS and the custom hallucination checker
  were scoring against a 200-character `text_preview` instead of full chunk text
  (~1,300 chars). This alone made faithfulness read **0.198 instead of 0.754**,
  and the hallucination rate **77.1% instead of 20.2%**. Fixed by fetching full
  text from Qdrant by chunk_id at evaluation time. The strongest example in this
  project of why raw metric outputs must be sanity-checked against manual
  inspection before being trusted.
- **generation_top_k bug**: the pipeline fed all 10 reranked chunks into
  generation instead of the intended 5, roughly doubling prompt cost per query.
  Fixed; verified input tokens dropped ~50%.
- **Hybrid retrieval underperformed expectations** (§5) — reported as found
  rather than quietly dropped.
- **LLM provider pivot**: the Gemini free tier returned `limit: 0` (no quota
  grant on the available account). Pivoted to locally-hosted Ollama — zero cost,
  no external dependency, no rate limits.
- **Dependency conflicts**: ragas 0.4.x has a broken import chain
  (`langchain_community.chat_models.vertexai` removed upstream); resolved by
  pinning rather than chasing an unstable version matrix across three
  fast-moving libraries.
- **A metric failure that was not a bug**: a later probe returned `faithfulness:
  NaN` across all samples. Rather than assume a regression, the failure rate was
  compared across metrics — it tracked LLM calls per metric (answer_relevancy
  0/10 failed, context_recall 2/10, context_precision 8/10, faithfulness 10/10),
  the signature of per-call timeouts under load rather than a schema mismatch.
  The pinned environment and the intact n=60 run (0/60 nulls) confirmed this.
  No code change was made.

## 13. Limitations

- **RAGAS evaluated at n=60, not n=250.** A full-scale run was measured, not
  merely estimated: a 10-question probe took 168 minutes wall clock at load
  average 35 on 64 cores (**16.8 min/question**), extrapolating to ~70 hours for
  n=250. The run was declined as an unproductive use of contended shared
  infrastructure. Retrieval metrics remain uncompromised at the full n=250.
- **Generation-level metrics were only computed for the winning configuration.**
  fixed and recursive were evaluated for retrieval only.
- **Judge model is qwen2.5:7b-instruct**, a relatively small open model. Verdicts
  may differ from a GPT-4-class judge. No human-audit agreement rate was
  computed.
- **Axis 5 ran at n=100 per variant**, and the contested/control analysis at
  n=17 per group — small samples, though the effect cleared significance under a
  clustering-aware test in both comparisons.
- **Eval set quality is bounded by IndicLegalQA's question style**, and the
  corpus is Supreme Court only, 2018–2022.

## 14. Future Work

- Run RAGAS on the full n=250 with a dedicated, non-shared GPU allocation
- Human-audit agreement rate against the LLM judge (~50 samples)
- Query rewriting / HyDE — retrieval misses are the dominant error source, and
  Axis 3 showed BM25 carrying most of the retrieval weight
- Compare generator model size (7B / 14B / 32B, all available locally) on
  faithfulness — attempted, aborted under GPU contention
- Staleness and contradiction detection for conflicting precedents
- Expand beyond the Supreme Court to lower courts
