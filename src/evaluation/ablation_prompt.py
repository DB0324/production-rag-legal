"""
Ablation Axis 5: strict (citation-forcing) vs loose prompt.

Varies the FULL guardrail bundle, not one rule: the loose variant drops the
only-from-context, mandatory-citation, insufficient-information and
no-speculation instructions. Documented as a bundle comparison.

The score-based sufficiency guard in pipeline.py is left ON for both variants
so the prompt is the only thing that changes. Declines are reported split into
guard-declined (deterministic score check) vs llm-declined (model self-declining
per prompt) -- the distinction that report section 10 had to untangle.

Usage:
    python -m src.evaluation.ablation_prompt --variant strict --limit 100
    python -m src.evaluation.ablation_prompt --variant loose  --limit 100
"""
import argparse, json, os, time

EVAL_PATH = "data/eval/indiclegalqa_filtered.json"

LOOSE_PROMPT = """You are a legal research assistant. Answer questions about Indian case law using the provided context passages.

Guidelines:
1. Use the context passages to inform your answer.
2. Provide a clear, helpful answer to the question.
3. Keep your answer concise and well-structured."""


def apply_variant(variant):
    """Patch SYSTEM_PROMPT for the loose variant; verify it actually took effect."""
    import src.generation.prompt_templates as pt
    if not hasattr(pt, "SYSTEM_PROMPT"):
        raise RuntimeError("prompt_templates.SYSTEM_PROMPT not found - script needs updating")
    if variant == "loose":
        pt.SYSTEM_PROMPT = LOOSE_PROMPT
        probe = pt.build_prompt("test", [{"text": "t", "chunk_id": "c",
                                          "doc_id": "d", "case_title": "ct"}])
        if "Guidelines:" not in probe:
            raise RuntimeError("Patch did not take effect - build_prompt does not read "
                               "SYSTEM_PROMPT at call time. Aborting before GPU spend.")
        print("  loose prompt patch VERIFIED")
    else:
        print("  strict prompt (repo default)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["strict", "loose"], required=True)
    p.add_argument("--strategy", default="semantic")
    p.add_argument("--limit", type=int, default=100)
    a = p.parse_args()

    apply_variant(a.variant)
    from src.generation.pipeline import query_pipeline

    out_path = f"results/raw_outputs_promptablation_{a.variant}_{a.strategy}.json"
    eval_data = json.load(open(EVAL_PATH))[:a.limit]

    results, done = [], 0
    if os.path.exists(out_path):                      # resume
        results = json.load(open(out_path))
        done = len(results)
        print(f"  resuming: {done} already complete")

    print(f"\n=== Axis 5: {a.variant} | {a.strategy} | n={len(eval_data)} ===\n")
    t0 = time.time()

    for i, item in enumerate(eval_data):
        if i < done:
            continue
        try:
            r = query_pipeline(question=item["question"], strategy=a.strategy,
                               use_reranker=True)
            results.append({
                "question": item["question"],
                "gold_doc_id": item["doc_id"],
                "gold_answer": item["answer"],
                "generated_answer": r["answer"],
                "citations": r["citations"],
                "chunks_used": r["chunks_used"],
                "latency": r["latency"],
                "tokens": r["tokens"],
                "confidence": r["confidence"],
                "avg_rerank_score": r.get("avg_rerank_score"),
            })
        except Exception as e:
            print(f"  ERROR q{i+1}: {e}")
            results.append({"question": item["question"], "error": str(e)})

        if (i + 1) % 10 == 0 or i + 1 == len(eval_data):
            json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
            el = time.time() - t0
            rate = (i + 1 - done) / el if el > 0 else 0
            eta = (len(eval_data) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i+1}/{len(eval_data)}  (~{eta:.0f} min left)")

    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    # ---- zero-LLM-cost metrics ----
    ok = [r for r in results if "error" not in r]
    guard = [r for r in ok if r.get("confidence") == "low"]
    llm_dec = [r for r in ok if r not in guard
               and "insufficient information" in r.get("generated_answer", "").lower()]
    cited = [r for r in ok if r.get("citations")]
    answered = [r for r in ok if r not in guard and r not in llm_dec]
    tin = [r["tokens"]["in"] for r in ok if r.get("tokens")]

    summary = {
        "variant": a.variant, "strategy": a.strategy, "n": len(ok),
        "guard_declined": len(guard),
        "guard_decline_rate": round(len(guard) / len(ok), 3) if ok else 0,
        "llm_self_declined": len(llm_dec),
        "llm_decline_rate": round(len(llm_dec) / len(ok), 3) if ok else 0,
        "answered": len(answered),
        "citation_presence_rate": round(len(cited) / len(answered), 3) if answered else 0,
        "avg_tokens_in": round(sum(tin) / len(tin), 1) if tin else 0,
        "errors": len(results) - len(ok),
    }
    json.dump(summary, open(f"results/ablation_prompt_{a.variant}.json", "w"), indent=2)

    print(f"\n=== {a.variant.upper()} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
