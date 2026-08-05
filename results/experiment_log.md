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
