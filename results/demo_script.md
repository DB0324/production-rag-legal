# Interview Demo Script — Legal RAG System
Target: ~5-6 minutes total. Rehearse each section until it's comfortable
without reading verbatim.

---

## 1. Architecture Overview (30-40s)
"I built a production-style hybrid RAG system over 4,140 Indian Supreme
Court judgments (2018-2022). The pipeline is: PDF extraction, three
chunking strategies I compared experimentally, hybrid retrieval (BM25 +
dense embeddings fused with Reciprocal Rank Fusion), cross-encoder
reranking, then generation with a citation-forcing prompt and a
calibrated guardrail that declines to answer when retrieval confidence
is too low."

[Optional: sketch the pipeline on a whiteboard/paper if in-person:
Query -> Cache -> Hybrid Retrieval -> Rerank -> Generate -> Response,
with an Evaluation Harness box on the side]

## 2. Live Query — Clean, Cited Answer (60-90s)
Run: a specific, well-formed question, e.g.
"What was the outcome of the appeal in the Supreme Court regarding
Ramji Singh?"

Point out while it runs:
- The answer is grounded, cites [Case Title, Doc ID] for every claim
- avg_rerank_score is high (~0.41+) -> confidence: "high"
- Latency breakdown: retrieval / rerank / generation timed separately

"This is my strongest correctness signal -- correct case identified,
correct citations, and the answer content matches the actual judgment
text, not just a plausible-sounding guess."

## 3. Live Query — Guardrail Triggering (45-60s)
Run: a genuinely out-of-corpus question, e.g.
"What is the boiling point of liquid nitrogen?"

"This is the moment that proves it's not just an LLM wrapper -- it
knows when it doesn't know. I calibrated this threshold empirically: I
found the original hardcoded value was actually dead code -- it assumed
a different reranker's score distribution -- so I measured real
good-vs-bad question scores and recalibrated it properly."

[This is your best "I actually debug things" story -- lean into it]

## 4. Caching Demonstration (30s)
Run the SAME first question again.

"Second identical query -- semantic cache hit. 33.8 seconds down to
0.047 seconds. I built this as a file-persisted cache instead of Redis
since I didn't have root access on the shared HPC server I was working
on -- same functional goal, adapted to the real constraint."

## 5. Metrics / Observability (30s)
Show /metrics output or the ablation table.

"Every query is logged -- latency per stage, token counts, cache hits --
so I can compute p50/p95 latency and track cost, not just eyeball single
responses."

## 6. Ablation Results + One Negative Result (60-90s)
Show the ablation table:
| Strategy | Reranker | Recall@1 | MRR |
Semantic + reranker won (Recall@1: 0.612, MRR: 0.657).

"The reranker gave a consistent 5-9 point Recall@1 lift across every
chunking strategy -- that's a real, reproducible signal, not noise.

One negative/surprising result: my first RAGAS run showed faithfulness
of 0.198, which looked alarming. I didn't just accept that number --
I investigated, and found the evaluation code was scoring against a
200-character text preview instead of the actual ~1,300-character
chunks the model saw. After fixing that, faithfulness corrected to
0.754. I think that's actually a stronger story than if everything had
just worked the first time -- it shows I don't trust a metric until
I've sanity-checked the measurement pipeline itself."

## 7. Close / Limitations (20-30s)
"Given more time, I'd want a dedicated GPU allocation to run RAGAS on
the full 250-question set instead of a 60-question subsample, and I'd
extend the corpus beyond Supreme Court judgments to lower courts."

---

## Backup: If the live API is down (infrastructure issue, e.g. today's
Qdrant cold-start slowness on the shared HPC server)
Have `results/final_report.md`, `results/ablation_table.csv`, and 2-3
screenshots of successful /query responses ready as a fallback. Explain
honestly: "The live demo depends on a shared university GPU server;
today it happened to be under heavy load. Here's a saved response from
earlier today showing the same behavior" -- this is a legitimate,
professional answer, not a weakness to hide.

## Key numbers to have memorized
- 4,140 documents, 250 eval questions, 219 unique cases
- Winning config: semantic chunking + bge-reranker-v2-m3
- Recall@1: 0.612, MRR: 0.657, nDCG@10: 0.684
- Faithfulness: 0.754, Hallucination rate: 20.2%
- Cache hit: 33.8s -> 0.047s
