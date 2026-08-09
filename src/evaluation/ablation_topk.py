"""
Ablation Axis 4: Top-k sensitivity -- how many reranked chunks are passed
to generation (generation_top_k=5/10/20), using the winning config
(semantic chunking + reranker). Measures latency, token cost, and
answer characteristics on a small n=20 subsample (scoped given the
expense of full-scale generation eval demonstrated earlier today).
"""
import json
import time
from src.generation.pipeline import query_pipeline

SAMPLE_PATH = "data/eval/topk_ablation_sample20.json"
TOPK_VALUES = [5, 10, 20]


def main():
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)

    all_results = {}

    for k in TOPK_VALUES:
        print(f"\n=== Testing generation_top_k={k} ===")
        results = []
        start = time.time()

        for item in sample:
            result = query_pipeline(
                question=item["question"],
                strategy="semantic",
                use_reranker=True,
                generation_top_k=k,
            )
            results.append({
                "question": item["question"],
                "gold_doc_id": item["doc_id"],
                "answer": result["answer"],
                "tokens_in": result["tokens"]["in"],
                "tokens_out": result["tokens"]["out"],
                "total_s": result["latency"].get("total_s"),
                "confidence": result["confidence"],
            })

        elapsed = time.time() - start
        avg_tokens_in = sum(r["tokens_in"] for r in results) / len(results)
        avg_tokens_out = sum(r["tokens_out"] for r in results) / len(results)
        avg_latency = sum(r["total_s"] for r in results if r["total_s"]) / max(1, len([r for r in results if r["total_s"]]))
        n_declined = sum(1 for r in results if "Insufficient" in r["answer"])

        summary = {
            "generation_top_k": k,
            "avg_tokens_in": round(avg_tokens_in, 1),
            "avg_tokens_out": round(avg_tokens_out, 1),
            "avg_latency_s": round(avg_latency, 2),
            "n_declined": n_declined,
            "total_wall_time_s": round(elapsed, 1),
            "n_samples": len(results),
        }
        all_results[f"k={k}"] = {"summary": summary, "raw": results}
        print(f"  avg_tokens_in={summary['avg_tokens_in']}, avg_latency_s={summary['avg_latency_s']}, declined={n_declined}/{len(results)}")

    with open("results/ablation_topk.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Top-k Sensitivity Summary ===")
    for k in TOPK_VALUES:
        s = all_results[f"k={k}"]["summary"]
        print(f"k={k:2d} | avg_tokens_in={s['avg_tokens_in']:6.1f} | avg_latency_s={s['avg_latency_s']:5.2f} | declined={s['n_declined']}/{s['n_samples']}")

    print("\nSaved to results/ablation_topk.json")


if __name__ == "__main__":
    main()
