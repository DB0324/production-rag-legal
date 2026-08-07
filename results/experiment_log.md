# Experiment Log

## Eval Set Construction
- IndicLegalQA: 10,000 QA pairs matched against corpus (4,140 docs) via fuzzy case-name matching
- Match threshold >=95 required (85-95 band showed real mismatches, e.g. "Avtar Singh vs UOI" incorrectly matched to "Dalbir Singh vs UOI")
- Final eval set: 250 QA pairs, capped at 2 questions/doc, 219 unique documents represented
- Manually spot-checked 5 random samples — all coherent and correctly matched

## E01 - Baseline: Fixed chunking, hybrid retrieval (BM25+dense+RRF), no reranker
- Recall@1: 0.492, Recall@5: 0.660, Recall@10: 0.712, Recall@20: 0.748, MRR: 0.564
- 63/250 questions had no correct doc in top-20 (see results/retrieval_misses_fixed.json)
- Next: repeat for recursive and semantic chunking to compare

## Ablation Results — Chunking × Reranker (Axis 1 + Axis 2)
| Strategy  | Reranker           | R@1   | R@5   | R@10  | R@20  | MRR   | nDCG@10 |
|-----------|--------------------|-------|-------|-------|-------|-------|---------|
| fixed     | bge-reranker-v2-m3 | 0.592 | 0.712 | 0.736 | 0.764 | 0.643 | 0.673   |
| fixed     | none               | 0.504 | 0.648 | 0.696 | 0.732 | 0.569 | 0.607   |
| recursive | bge-reranker-v2-m3 | 0.580 | 0.692 | 0.728 | 0.756 | 0.633 | 0.660   |
| recursive | none               | 0.536 | 0.636 | 0.696 | 0.744 | 0.587 | 0.620   |
| semantic  | bge-reranker-v2-m3 | 0.612 | 0.712 | 0.740 | 0.772 | 0.657 | 0.684   |
| semantic  | none               | 0.532 | 0.640 | 0.684 | 0.744 | 0.587 | 0.613   |

**Winner: semantic chunking + bge-reranker-v2-m3.** Reranker gives a consistent
+0.05-0.09 Recall@1 lift across all three chunking strategies — clear signal,
not noise. Freezing this config for the generation stage.

## Bug Fix: generation_top_k not respected
- pipeline.py was passing all rerank_top_k=10 chunks to generation instead of
  the intended generation_top_k=5, roughly doubling prompt size/cost per query.
- Fixed: context_chunks = reranked_results[:generation_top_k]
- Verified: avg input tokens dropped from ~5,200 to ~2,750 per query on 5-question smoke test.
- Any full_eval runs generated before this fix should be considered invalid/re-run.

## Full Pipeline Run — Semantic + Reranker + Ollama (qwen2.5:7b-instruct)
- 250/250 questions completed, 0 errors
- Total tokens: 704,808 in / 49,346 out
- Total time: 88.4 min (avg 21.2s/query, after fixing the generation_top_k bug)
- Output: results/raw_outputs_semantic_reranked.json
- Note: mid-run, saw a temporary slowdown when another user's job consumed
  most GPU VRAM, causing Ollama to fall back to CPU inference for a period.
  This is a real shared-infrastructure constraint, noted as a limitation.

## Manual Spot-Check: Retrieval-miss vs Hallucination Distinction
- Q1/Q2 (vague, auto-generated eval questions like "What broader issue...")
  retrieve WRONG documents, but the model faithfully cites only what it
  retrieved -- this is a retrieval failure, not a generation hallucination.
- Q3 (specific, well-formed question) retrieves correctly and generates an
  answer semantically aligned with gold, using correct citations.
- Conclusion: vague/generic auto-generated questions in IndicLegalQA are a
  known limitation of the eval set itself, not necessarily a pipeline flaw.

## RAGAS Smoke Test (3 questions, includes the 2 known-vague questions)
- faithfulness: 0.159, answer_relevancy: 0.697, context_precision: 0.333, context_recall: 0.333
- Low scores here are consistent with manual spot-check: 2/3 of these questions
  had retrieval failures (vague auto-generated questions), which RAGAS correctly
  penalizes on faithfulness/context metrics. Not representative of full-set quality.
- Fixed RAGAS + langchain-community + ragas version conflicts (vertexai import
  stub, CPU-mode embeddings due to shared GPU contention, RunConfig timeout=600s
  max_workers=2 to avoid overwhelming contested GPU).

## Scope Decision: RAGAS evaluated on n=60 subsample, not full n=250
- Full-scale RAGAS (4 metrics x 250 = up to 1000 LLM judge calls) was
  attempted but projected 70+ hours due to shared-GPU contention with
  other users' jobs, with ~50% of calls timing out even at 600s timeout.
- Killed after 61/1000 jobs (2h16m). Switched to a random n=60 subsample
  (seed=42) for RAGAS specifically -- a standard practice given LLM-judge
  cost, and large enough for meaningful aggregate faithfulness/relevancy
  scores. Retrieval metrics (Recall@k, MRR) remain on the full n=250 set.
- Noted as a real-world constraint of shared HPC infrastructure in the
  final report's limitations section.

## RAGAS Full Results (n=60 subsample, semantic+reranker+Ollama qwen2.5:7b)
- faithfulness: 0.198
- answer_relevancy: 0.689
- context_precision: 0.369
- context_recall: 0.246
- n_samples: 60

Notable: faithfulness/context_recall are much lower than retrieval Recall@10
(0.740) would suggest. Needs failure analysis -- possible causes: (a) RAGAS's
own LLM-judge (qwen2.5:7b) may be a weak/inconsistent judge for legal text,
(b) context reconstruction from chunks_used truncates to text_preview (200
chars) which may be too short for RAGAS to properly assess, (c) genuine
generation quality issues beyond the known vague-question retrieval misses.

## RAGAS CORRECTED Results (n=60, full chunk text from Qdrant, not truncated preview)
- faithfulness: 0.754 (was 0.198 with truncated 200-char context -- measurement bug)
- answer_relevancy: 0.690
- context_precision: 0.614
- context_recall: 0.617 (was 0.246 with truncated context)
- n_samples: 60 (1 sample excluded due to LLM-judge output parsing error, expected
  occasionally with smaller open judge models)
- Total runtime: 6h12m on shared/contested GPU (240 LLM-judge calls)

CONCLUSION: The original low scores were an artifact of chunks_used only storing
a 200-char text_preview instead of full chunk text. After fetching full text from
Qdrant, scores are consistent with a well-functioning RAG pipeline.

## Hallucination Checker: Fixed same truncation bug as RAGAS
- Applied identical fix (fetch full chunk text from Qdrant by chunk_id)
- 5-question smoke test: hallucination rate corrected from 77.1% -> 11.4%
- Reused fetch_full_chunk_texts/detect_strategy_from_input from ragas_eval.py

## Hallucination Checker: FULL results (n=199, semantic+reranker+Ollama)
- Questions checked: 199 (51 excluded: "insufficient information" responses
  from the sufficiency guard, consistent with retrieval-miss rate)
- Total claims: 1441
- Unsupported claims: 291
- Overall hallucination rate: 20.2%
- Tokens used: 3,427,546

Cross-check: RAGAS faithfulness=0.754 (~24.6% unfaithful) and hallucination
checker's 20.2% unsupported-claims rate are in the same range -- two
independent methods converge, giving confidence in both measurements.
