"""
Compute Recall@k and MRR for hybrid retrieval against the full eval set.
"""
import json
from src.retrieval.hybrid_retriever import hybrid_search

COLLECTION_NAME = "legal_fixed"
BM25_PATH = "data/chunks/fixed_bm25.pkl"
EVAL_PATH = "data/eval/indiclegalqa_filtered.json"
TOP_K = 20


def main():
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    hits_at = {1: 0, 5: 0, 10: 0, 20: 0}
    reciprocal_ranks = []
    misses = []

    for i, item in enumerate(eval_data):
        results = hybrid_search(
            query=item["question"],
            collection_name=COLLECTION_NAME,
            bm25_path=BM25_PATH,
            top_k=TOP_K,
        )
        found_docs = [r["doc_id"] for r in results]

        if item["doc_id"] in found_docs:
            rank = found_docs.index(item["doc_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item)

        if (i + 1) % 50 == 0:
            print(f"  evaluated {i+1}/{len(eval_data)}")

    n = len(eval_data)
    print(f"\n=== Retrieval Metrics (n={n}) ===")
    for k in hits_at:
        print(f"Recall@{k}: {hits_at[k]/n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks)/n:.3f}")
    print(f"\nTotal misses (not found in top-{TOP_K}): {len(misses)}")

    with open("results/retrieval_misses_fixed.json", "w") as f:
        json.dump(misses, f, indent=2)
    print("Saved miss cases to results/retrieval_misses_fixed.json for failure analysis")


if __name__ == "__main__":
    main()
