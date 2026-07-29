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
