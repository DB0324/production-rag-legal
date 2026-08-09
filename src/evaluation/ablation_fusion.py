"""
Ablation Axis 3: Fusion weighting -- compares pure BM25, pure dense,
and RRF hybrid retrieval, using the winning chunking strategy (semantic).
Reuses existing retrieval functions, no new retrieval logic.
"""
import json
from src.retrieval.hybrid_retriever import bm25_search, dense_search, hybrid_search, load_bm25

COLLECTION_NAME = "legal_semantic"
BM25_PATH = "data/chunks/semantic_bm25.pkl"
EVAL_PATH = "data/eval/indiclegalqa_filtered.json"
TOP_K = 20


def evaluate_mode(mode: str, eval_data: list) -> dict:
    bm25_data = load_bm25(BM25_PATH) if mode in ("bm25", "hybrid") else None

    hits_at = {1: 0, 5: 0, 10: 0, 20: 0}
    reciprocal_ranks = []

    for item in eval_data:
        if mode == "bm25":
            results = bm25_search(bm25_data, item["question"], top_k=TOP_K)
        elif mode == "dense":
            results = dense_search(COLLECTION_NAME, item["question"], top_k=TOP_K)
        elif mode == "hybrid":
            results = hybrid_search(item["question"], COLLECTION_NAME, BM25_PATH, top_k=TOP_K)
        else:
            raise ValueError(mode)

        found_docs = [r["doc_id"] for r in results]
        if item["doc_id"] in found_docs:
            rank = found_docs.index(item["doc_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_data)
    return {
        "mode": mode,
        "recall_at_1": round(hits_at[1] / n, 3),
        "recall_at_5": round(hits_at[5] / n, 3),
        "recall_at_10": round(hits_at[10] / n, 3),
        "recall_at_20": round(hits_at[20] / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "n_eval": n,
    }


def main():
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    results = []
    for mode in ["bm25", "dense", "hybrid"]:
        print(f"Evaluating mode: {mode}...")
        r = evaluate_mode(mode, eval_data)
        results.append(r)
        print(f"  Recall@1={r['recall_at_1']}, Recall@10={r['recall_at_10']}, MRR={r['mrr']}")

    with open("results/ablation_fusion.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/ablation_fusion.json")

    print("\n=== Fusion Weighting Ablation Summary ===")
    for r in results:
        print(f"{r['mode']:8s} | R@1={r['recall_at_1']} R@5={r['recall_at_5']} R@10={r['recall_at_10']} MRR={r['mrr']}")


if __name__ == "__main__":
    main()
