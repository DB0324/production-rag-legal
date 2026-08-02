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
