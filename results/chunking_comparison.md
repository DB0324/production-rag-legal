# Chunking Strategy Comparison (Structural Stats)

| Strategy | Total Chunks | Avg Chunks/Doc | Avg Chars/Chunk | Min/Max Chars |
|---|---|---|---|---|
| Fixed | 107,056 | 25.9 | 2,189.1 | 203 / 4,077 |
| Recursive | 253,394 | 61.2 | 842.1 | 40 / 4,342 |
| Semantic | 163,884 | 39.6 | 1,302.6 | 40 / 4,342 |

Note: these are structural stats only (chunk count/size). Retrieval quality
(Recall@k, nDCG) and generation quality (faithfulness, hallucination rate)
per strategy will be measured in Week 5 evaluation, not assumed from these
numbers alone.
