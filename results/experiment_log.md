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

## Guardrail Bug Found and Fixed: Sufficiency Threshold Was Never Triggering
- Original SUFFICIENCY_THRESHOLD = -5.0 assumed bge-reranker-v2-m3 produces
  large negative scores (comment: "bge-reranker scores can be negative").
  In practice this reranker/setup produces small POSITIVE scores (0.001-0.98
  range), so avg_score < -5.0 was NEVER true -- the coded guard was dead code.
- The "Insufficient information" responses seen in earlier guardrail tests
  were actually produced by the LLM's own judgment (System Prompt Rule 3),
  not the deterministic threshold check -- a much less reliable mechanism.
- Calibrated real threshold using 5 known-good vs 5 known-bad questions:
  GOOD avg scores: 0.41-0.93 | BAD avg scores: 0.004-0.023
  Set SUFFICIENCY_THRESHOLD = 0.10 (clean separation, wide margin both sides)
- Re-tested: 5/5 out-of-corpus questions correctly declined via the real
  threshold now; in-corpus questions (avg_score=0.41) still answer correctly.
- Note: build_sufficiency_check_prompt() in prompt_templates.py is an
  additional unused LLM-based guard function -- not wired into pipeline.py.
  Left as-is since the numeric threshold is now the primary, reliable guard;
  documented as available future enhancement (ensemble of both signals).

## End-to-End API Verification (final)
POST /query on "What was the outcome of the appeal in the Supreme Court
regarding Ramji Singh?" returned a correct, cited answer via the full
API stack (retrieval -> rerank -> generate), confirming avg_rerank_score
(0.411) and confidence ("high") both align with the calibrated guardrail.
Latency breakdown: retrieval 9.97s, rerank 1.82s, generation 17.61s,
total 29.4s. Project is fully functional end-to-end through the API layer.

## Cost/Latency Observability: Implemented and Verified
- Added SQLite-backed query logging (src/observability/logger.py) and
  a new /metrics endpoint aggregating p50/p95 latency, token totals,
  and cache hit-rate (0% -- caching not yet implemented).
- Verified via 2 live queries: correct latency/token capture for a
  successful answer (total_s=47.1s), and correct "insufficient info"
  handling for an out-of-corpus question.
- Known gap: guardrail-declined queries don't populate total_s (early
  return in pipeline.py before the total_s calculation), so they're
  undercounted in latency aggregates. Minor, noted for future fix.
- Added start_api.sh wrapper to prevent LLM_PROVIDER env var mismatches
  on restart (root cause of several earlier debugging sessions today).

## Caching Layer: Implemented and Verified
- Semantic cache (embed query + cosine similarity, threshold=0.95),
  file-persisted instead of Redis given no-root HPC constraint
  (same reasoning as Qdrant local-mode substitution for Docker).
- Verified: identical query, second call -- total_s dropped from 33.83s
  to 0.047s (cache hit), confirmed via cache_hit:true in response.
- Only successful (non-"insufficient information") answers are cached,
  to avoid caching guardrail declines as if they were confident answers.

## Investigation: Qdrant Cold-Start Slowness (root-caused, not fixed)
Followed up on the earlier I/O contention finding with deeper diagnostics:
- Raw sequential disk read: 5.7 GB/s (dd test) -- disk hardware is fine.
- No WAL/lock/journal files present -- storage files unmodified since
  original index build (Jul 30), ruling out corruption from today's
  repeated process kills.
- SQLite integrity_check: "ok" on legal_fixed -- file structurally sound.
- Conclusion: slowness is consistent with Qdrant's random-access read
  pattern over large (1.2-2.9GB) local files becoming slow under
  memory/page-cache pressure on a heavily shared server (24 concurrent
  users), even when raw sequential disk throughput is excellent and
  cluster load average is moderate (1.2-1.5). This is a genuine
  characteristic of local-mode embedded Qdrant at this data scale on
  shared infrastructure, not a bug in this project's code.
- Practical implication for the final report: this is exactly the kind
  of finding that justifies the blueprint's original recommendation of
  Qdrant server/Docker mode for datasets >20,000 points in a real
  production deployment -- local mode was a reasonable dev/portfolio
  substitution given no-root constraints, but this investigation
  independently confirms *why* the tool's own warning message
  (seen throughout this session) exists.

## Ablation Axis 3: Fusion Weighting (pure BM25 vs pure dense vs RRF hybrid)
Using winning chunking strategy (semantic), n=250:

| Mode   | Recall@1 | Recall@5 | Recall@10 | MRR   |
|--------|----------|----------|-----------|-------|
| BM25   | 0.524    | 0.652    | 0.700     | 0.583 |
| Dense  | 0.440    | 0.560    | 0.604     | 0.494 |
| Hybrid | 0.532    | 0.640    | 0.684     | 0.586 |

Finding: hybrid only marginally beats pure BM25 (within noise on R@1/MRR),
and actually loses slightly on R@5/R@10. Pure dense retrieval is clearly
weakest alone. Hypothesis: legal text relies heavily on exact-match
signals (case citations, section/statute numbers) that BM25 captures
directly, while dense embeddings add less value for this precision-
sensitive retrieval task compared to more semantically-driven domains.
This tempers the blueprint's original assumption that hybrid clearly
beats either component alone -- worth stating explicitly as a corpus-
specific finding rather than a general claim.

## Ablation Axis 4: Top-k Sensitivity (n=20 subsample, semantic+reranker)
| generation_top_k | avg_tokens_in | avg_latency_s | declined/20 |
|---|---|---|---|
| 5  | 2776.9 | 29.71 | 8 |
| 10 | 5087.8 | 38.00 | 7 |
| 20 | 5087.8 | 32.25 | 7 |

IMPORTANT CAVEAT: k=10 and k=20 show identical avg_tokens_in, because
rerank_top_k (default=10) caps the number of chunks available to slice
from BEFORE generation_top_k is applied. So this experiment validly
compares k=5 vs k=10, but did NOT actually test true k=20 -- that would
require also raising rerank_top_k=20 to have enough candidates. Logged
honestly rather than misreporting k=20 as a distinct data point.

Valid finding (k=5 vs k=10): near-doubling context (2777 -> 5088 tokens,
+83%) and +28% latency (29.7s -> 38.0s) bought only a marginal reduction
in guardrail-declined questions (8/20 -> 7/20). This suggests diminishing
returns on additional context at this corpus's chunk granularity --
consistent with the intuition that the reranker already surfaces the
most relevant chunks in its top 5, and additional lower-ranked chunks
add cost without proportionally improving coverage.

## Correction: Vague-Question Theory Was Overstated
Earlier assumed ~20-23% of eval questions were "vague" and responsible
for lower-than-hoped retrieval scores. Built a precise filter (requires
generic phrasing AND no case-name/date marker) and found only 2/250
(0.8%) genuinely qualify -- exactly the 2 originally found via manual
spot-check early in the project, not a large hidden population.

Re-ran retrieval metrics on the clean n=248 subset: Recall@1 0.532->0.536,
MRR 0.587->0.590 -- negligible change. CONCLUSION: the observed retrieval
scores (Recall@1 ~0.53-0.61 depending on config) are NOT substantially
explained by eval-set question quality. They reflect genuine retrieval
difficulty on this legal corpus -- likely due to semantic similarity
between related cases, dense/formal legal language, and the corpus
containing many topically-similar judgments that are hard to disambiguate
without exact citation matching. This is a more honest explanation than
originally assumed, and is itself informative: BM25's strong showing in
the Axis 3 fusion ablation is consistent with this -- exact-match signals
matter more than semantic similarity for disambiguating similar legal
cases.

## Attempted: qwen3:14b Generation Comparison (Aborted)
- Generated 60/60 answers successfully with qwen3:14b (0 errors,
  fast completion) -- see results/raw_outputs_semantic_reranked_qwen3_14b.json
- Attempted RAGAS evaluation on these outputs: every job timed out at the
  600s ceiling, sustained over 2.5+ hours (35+ consecutive timeouts).
- Root cause: qwen3:14b (14B params) requires substantially more GPU
  memory/compute than qwen2.5:7b; combined with heavy concurrent GPU
  contention from another user's job (84% utilization, 87C, ~7GB used by
  their process alongside qwen3:14b's ~33GB), the shared GPU could not
  serve RAGAS's concurrent judge calls within any reasonable timeout.
- Decision: aborted after 2.5hrs/35 jobs rather than continue an
  unproductive run. Generation output (qwen3:14b answers) is saved and
  available for future evaluation when GPU contention allows, but RAGAS
  scoring for this model is NOT included in final results.
- Noted as future work: comparing generator model size (7B vs 14B vs
  32B, all already available via Ollama) on faithfulness/quality would
  require a dedicated (non-shared) GPU allocation to run reliably.

## Attempted: qwen3:14b Generation Comparison (Aborted)
- Generated 60/60 answers successfully with qwen3:14b (0 errors,
  fast completion) -- see results/raw_outputs_semantic_reranked_qwen3_14b.json
- Attempted RAGAS evaluation on these outputs: every job timed out at the
  600s ceiling, sustained over 2.5+ hours (35+ consecutive timeouts).
- Root cause: qwen3:14b (14B params) requires substantially more GPU
  memory/compute than qwen2.5:7b; combined with heavy concurrent GPU
  contention from another user's job (84% utilization, 87C, ~7GB used by
  their process alongside qwen3:14b's ~33GB), the shared GPU could not
  serve RAGAS's concurrent judge calls within any reasonable timeout.
- Decision: aborted after 2.5hrs/35 jobs rather than continue an
  unproductive run. Generation output (qwen3:14b answers) is saved and
  available for future evaluation when GPU contention allows, but RAGAS
  scoring for this model is NOT included in final results.
- Noted as future work: comparing generator model size (7B vs 14B vs
  32B, all already available via Ollama) on faithfulness/quality would
  require a dedicated (non-shared) GPU allocation to run reliably.

## Attempted: qwen3:14b Generation Comparison (Aborted)
- Generated 60/60 answers successfully with qwen3:14b (0 errors,
  fast completion) -- see results/raw_outputs_semantic_reranked_qwen3_14b.json
- Attempted RAGAS evaluation on these outputs: every job timed out at the
  600s ceiling, sustained over 2.5+ hours (35+ consecutive timeouts).
- Root cause: qwen3:14b (14B params) requires substantially more GPU
  memory/compute than qwen2.5:7b; combined with heavy concurrent GPU
  contention from another user's job (84% utilization, 87C, ~7GB used by
  their process alongside qwen3:14b's ~33GB), the shared GPU could not
  serve RAGAS's concurrent judge calls within any reasonable timeout.
- Decision: aborted after 2.5hrs/35 jobs rather than continue an
  unproductive run. Generation output (qwen3:14b answers) is saved and
  available for future evaluation when GPU contention allows, but RAGAS
  scoring for this model is NOT included in final results.
- Noted as future work: comparing generator model size (7B vs 14B vs
  32B, all already available via Ollama) on faithfulness/quality would
  require a dedicated (non-shared) GPU allocation to run reliably.

## E09 - Axis 5: strict (citation-forcing) vs loose prompt
Hypothesis: the citation-forcing prompt reduces unsupported answering by making
the model abstain when context is inadequate.
Config: semantic + bge-reranker-v2-m3, n=100/variant, guard ON in both arms
(threshold 0.10), only SYSTEM_PROMPT varied. ~215 min/variant at load ~34.

| Metric | Strict | Loose |
|---|---|---|
| Guard declined (score-based) | 1% | 1% |
| LLM self-declined (prompt) | 17% | 0% |
| Citation rate among answered | 100.0% (82/82) | 94.9% (94/99) |
| Avg input tokens | 2,783 | 2,692 |

Result: prompt drives all abstention (+17 pts); the 0.10 score guard fires on
1/100 real eval questions. Confirms the section-10 finding empirically -- the
prompt, not the deterministic check, was doing the guardrail work all along.
Citation compliance is perfect under strict; loose still cites 94.9% unprompted,
so citation is largely intrinsic and the instruction closes the last 5 points.
Token cost delta +3.3% -- the guardrail is effectively free.
Keep: strict prompt confirmed as the frozen config.

## E10 - Contested-question hallucination check (Axis 5 follow-up)
Hypothesis: the strict prompt declines questions that are specifically harder to
answer in a grounded way -- i.e. abstention is targeted, not indiscriminate.
Method: took the 17 questions where strict declined but loose answered, ran the
claim-level hallucination checker on loose's answers to exactly those.

Contested: 17 questions, 141 claims, 43 unsupported -> 30.5%
Baseline : 199 questions -> 20.2%
Per-question means: 0.340 vs 0.195 (medians 0.333 vs 0.167)
Mann-Whitney U, one-sided: p = 0.0027 (significant)

Tested on per-question rates rather than claim-level proportions because claims
within an answer are not independent; the naive claim-level interval would have
overstated confidence.

Result: KEEP. The citation-forcing prompt selectively refuses the cases where
answering produces unsupported claims. Judge cost 284,958 tokens (~16.8K/question).
Limitation: baseline is from an earlier run, not a matched control in this experiment.

## E11 - Matched control for E10 (same-run comparison)
Motivation: E10 compared contested questions against a baseline from an earlier
run. This adds a control drawn from the SAME run, same config, same judge, same n,
removing the cross-run confound.
Method: sampled 17 non-contested, non-declined loose answers (random.seed=42,
pool=82) and ran the identical claim-level hallucination checker on them.

Contested: 17 questions, 141 claims, 43 unsupported -> 30.5%
Control  : 17 questions, 117 claims, 19 unsupported -> 16.2%
Per-question means  : 0.340 vs 0.164 (2.07x)
Per-question medians: 0.333 vs 0.000
Mann-Whitney U, one-sided: p = 0.0084 (significant)

The median control answer contains ZERO unsupported claims, while the median
question the strict prompt refused has a third of its claims unsupported.

Result: KEEP. Confirms E10 under a cleaner design with a larger effect size
(+14.3 pts here vs +10.3 pts against the earlier baseline). Both comparisons are
significant; the matched design trades statistical power for a cleaner comparison
and still clears 0.05. Judge cost 277,438 tokens.
