"""
Parameterized retrieval evaluation: compute Recall@k, MRR, and nDCG@10
for any chunking strategy against the eval set.

Usage:
    python -m src.evaluation.run_retrieval_eval --strategy recursive
    python -m src.evaluation.run_retrieval_eval --strategy semantic
    python -m src.evaluation.run_retrieval_eval --all
"""
import argparse
import csv
import json
import math
import os
from src.retrieval.hybrid_retriever import hybrid_search

EVAL_PATH = "data/eval/indiclegalqa_filtered.json"
RESULTS_CSV = "results/ablation_table.csv"
TOP_K = 50

# Strategy → (Qdrant collection name, BM25 pickle path)
STRATEGY_CONFIG = {
    "fixed": ("legal_fixed", "data/chunks/fixed_bm25.pkl"),
    "recursive": ("legal_recursive", "data/chunks/recursive_bm25.pkl"),
    "semantic": ("legal_semantic", "data/chunks/semantic_bm25.pkl"),
}


def compute_ndcg_at_k(found_docs, gold_doc_id, k=10):
    """Compute nDCG@k for a single query (binary relevance: 1 if gold doc, else 0).

    Deduplicates found_docs to document-level ranking first, so multiple
    chunks from the same document don't inflate DCG beyond IDCG.
    """
    # Deduplicate to document-level ranking (keep first occurrence)
    seen = set()
    deduped = []
    for doc_id in found_docs:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append(doc_id)

    dcg = 0.0
    for i, doc_id in enumerate(deduped[:k]):
        rel = 1.0 if doc_id == gold_doc_id else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    # Ideal DCG: gold doc at rank 1
    idcg = 1.0 / math.log2(2)  # = 1.0
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_strategy(strategy: str):
    """Run retrieval evaluation for a single chunking strategy."""
    if strategy not in STRATEGY_CONFIG:
        print(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_CONFIG.keys())}")
        return

    collection_name, bm25_path = STRATEGY_CONFIG[strategy]

    # Verify BM25 index exists
    if not os.path.exists(bm25_path):
        print(f"ERROR: BM25 index not found at {bm25_path}. "
              f"Run indexing first: python -m src.ingestion.run_indexing_{strategy}")
        return

    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Evaluating strategy: {strategy}")
    print(f"  Collection: {collection_name}")
    print(f"  BM25 index: {bm25_path}")
    print(f"  Eval set: {len(eval_data)} questions")
    print(f"{'='*60}\n")

    hits_at = {1: 0, 5: 0, 10: 0, 20: 0}
    reciprocal_ranks = []
    ndcg_scores = []
    misses = []

    for i, item in enumerate(eval_data):
        results = hybrid_search(
            query=item["question"],
            collection_name=collection_name,
            bm25_path=bm25_path,
            top_k=TOP_K,
        )
        found_docs = [r["doc_id"] for r in results]

        # nDCG@10
        ndcg_scores.append(compute_ndcg_at_k(found_docs, item["doc_id"], k=10))

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
    recall_at = {k: hits_at[k] / n for k in hits_at}
    mrr = sum(reciprocal_ranks) / n
    ndcg_10 = sum(ndcg_scores) / n

    print(f"\n=== Retrieval Metrics [{strategy}] (n={n}) ===")
    for k in sorted(recall_at):
        print(f"  Recall@{k}: {recall_at[k]:.3f}")
    print(f"  MRR:       {mrr:.3f}")
    print(f"  nDCG@10:   {ndcg_10:.3f}")
    print(f"  Misses:    {len(misses)} (not in top-{TOP_K})")

    # Save miss cases
    miss_path = f"results/retrieval_misses_{strategy}.json"
    with open(miss_path, "w") as f:
        json.dump(misses, f, indent=2)
    print(f"  Saved miss cases → {miss_path}")

    # Append to ablation table
    _append_to_ablation_csv(strategy, recall_at, mrr, ndcg_10, n)

    return {
        "strategy": strategy,
        "recall_at": recall_at,
        "mrr": mrr,
        "ndcg_10": ndcg_10,
        "n_eval": n,
        "n_misses": len(misses),
    }


def _append_to_ablation_csv(strategy, recall_at, mrr, ndcg_10, n):
    """Append or update a row in the ablation CSV."""
    fieldnames = [
        "chunking_strategy", "reranker",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
        "mrr", "ndcg_at_10", "n_eval",
    ]

    # Read existing rows (if any)
    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # Remove existing row for this strategy + reranker=none (will replace it)
    rows = [r for r in rows if not (r.get("chunking_strategy") == strategy and r.get("reranker") == "none")]

    # Add new row
    new_row = {
        "chunking_strategy": strategy,
        "reranker": "none",
        "recall_at_1": f"{recall_at[1]:.3f}",
        "recall_at_5": f"{recall_at[5]:.3f}",
        "recall_at_10": f"{recall_at[10]:.3f}",
        "recall_at_20": f"{recall_at[20]:.3f}",
        "mrr": f"{mrr:.3f}",
        "ndcg_at_10": f"{ndcg_10:.3f}",
        "n_eval": str(n),
    }
    rows.append(new_row)

    # Sort: fixed first, then recursive, then semantic
    order = {"fixed": 0, "recursive": 1, "semantic": 2}
    rows.sort(key=lambda r: (order.get(r.get("chunking_strategy", ""), 99), r.get("reranker", "")))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Updated {RESULTS_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for chunking strategies")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"],
                        help="Which chunking strategy to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all three strategies sequentially")
    args = parser.parse_args()

    if args.all:
        for strategy in ["fixed", "recursive", "semantic"]:
            evaluate_strategy(strategy)
    elif args.strategy:
        evaluate_strategy(args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
