# Failure Taxonomy — Legal RAG System

Based on manual review of retrieval misses, RAGAS low-scoring cases, and
hallucination checker flagged claims.

## Category 1: Retrieval-Miss on Vague Questions
**Pattern**: Generic, auto-generated questions lacking specific facts/parties
(e.g. "What broader issue did the Supreme Court address regarding the
administration of justice?") retrieve plausible-but-wrong documents.

**Example**:
- Question: "What broader issue did the Supreme Court address in its
  judgement regarding the administration of justice?"
- Expected: SOMESH CHAURASIA versus STATE OF M.P. & ANR (2021_6_692_722_EN)
- Retrieved instead: SWAPNIL TRIPATHI (open justice case) — topically
  plausible, wrong specific case.

**Root cause**: IndicLegalQA's auto-generated question style; not a
pipeline defect. The model faithfully cited only what it retrieved,
confirming this is a retrieval failure, not a hallucination.

**Frequency**: ~51/250 questions in the winning config triggered the
sufficiency guard for exactly this reason (Recall@20 = 0.772, meaning
~23% of questions never surface the correct doc even in the top 20).

## Category 2: Correct Retrieval, Correct Generation (majority case)
**Pattern**: Specific, well-formed questions retrieve the correct document
(often with rerank_score > 0.9) and generate answers semantically aligned
with gold answers, with correct citations.

**Example**:
- Question: "What was the Supreme Court's conclusion on the merit of the
  selection process conducted by the State Screening Committee?"
- Retrieved: BAIDYANATH YADAV versus ADITYA NARAYAN ROY & ORS, rank 1,
  correctly and consistently.
- Generated answer closely mirrors gold answer in substance.

**Frequency**: majority of the 199/250 questions that passed the
sufficiency guard.

## Category 3: Measurement Artifacts (caught and fixed, not real failures)
**Pattern**: Evaluation code (RAGAS, hallucination checker) initially
scored against a 200-char text_preview field instead of full chunk text
(~1,300 chars avg), producing artificially catastrophic scores.

**Impact before fix**: faithfulness 0.198, hallucination rate 77.1%
**Impact after fix**: faithfulness 0.754, hallucination rate 20.2%

**Lesson**: a metric that looks alarmingly bad relative to manual
spot-checks is a signal to audit the measurement pipeline itself before
concluding the system is broken.

## Category 4: Dead/Miscalibrated Guardrail (caught and fixed)
**Pattern**: The coded sufficiency threshold (-5.0) assumed a reranker
score distribution that didn't match bge-reranker-v2-m3's actual output
range (small positive scores, not large negatives). The threshold check
was silently dead code; "insufficient information" responses were
actually driven by the LLM's own prompt-instructed judgment, a less
reliable mechanism.

**Fix**: recalibrated using 5 known-good vs 5 known-bad questions
(GOOD avg 0.41-0.93, BAD avg 0.004-0.023), set threshold=0.10.

## Summary Table

| Category | Root Cause | Status |
|---|---|---|
| Vague-question retrieval miss | Eval set question style | Documented limitation |
| Correct end-to-end cases | N/A | Majority case, working as intended |
| Truncated-context measurement bug | Evaluation code bug | Fixed |
| Dead guardrail threshold | Miscalibrated constant | Fixed |
