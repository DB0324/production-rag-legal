"""
Retrieval evaluation WITH cross-encoder reranking.
Runs hybrid retrieval → reranker → metrics for each chunking strategy.

Usage:
    python -m src.evaluation.run_retrieval_eval_reranked --strategy fixed
    python -m src.evaluation.run_retrieval_eval_reranked --all
"""
import argparse
import csv
import json
import math
import os
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank

EVAL_PATH = "data/eval/indiclegalqa_filtered.json"
RESULTS_CSV = "results/ablation_table.csv"
HYBRID_TOP_K = 50    # retrieve this many from hybrid search (widened from 20)
RERANK_TOP_K = 10    # keep this many after reranking (widened from 5)

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
    seen = set()
    deduped = []
    for doc_id in found_docs:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append(doc_id)

    dcg = 0.0
    for i, doc_id in enumerate(deduped[:k]):
        rel = 1.0 if doc_id == gold_doc_id else 0.0
        dcg += rel / math.log2(i + 2)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_strategy_reranked(strategy: str):
    """Run retrieval evaluation with reranking for a single strategy."""
    if strategy not in STRATEGY_CONFIG:
        print(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_CONFIG.keys())}")
        return

    collection_name, bm25_path = STRATEGY_CONFIG[strategy]

    if not os.path.exists(bm25_path):
        print(f"ERROR: BM25 index not found at {bm25_path}. Run indexing first.")
        return

    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Evaluating strategy: {strategy} + reranker")
    print(f"  Collection: {collection_name}")
    print(f"  BM25 index: {bm25_path}")
    print(f"  Hybrid top-k: {HYBRID_TOP_K} → rerank to top-{RERANK_TOP_K}")
    print(f"  Eval set: {len(eval_data)} questions")
    print(f"{'='*60}\n")

    # We measure recall/MRR on the *reranked* top-5, but also on wider cuts
    # For a fair comparison, measure recall@1 and recall@5 (reranked range)
    # Recall@10 and @20 are effectively capped by the hybrid top-20 input
    hits_at = {1: 0, 5: 0, 10: 0, 20: 0}
    reciprocal_ranks = []
    ndcg_scores = []
    misses = []
    avg_rerank_scores = []

    for i, item in enumerate(eval_data):
        # Step 1: Hybrid retrieval (top-20)
        hybrid_results = hybrid_search(
            query=item["question"],
            collection_name=collection_name,
            bm25_path=bm25_path,
            top_k=HYBRID_TOP_K,
        )

        # Step 2: Rerank the top-20 — but keep all 20 for metric computation
        reranked = rerank(item["question"], hybrid_results, top_k=HYBRID_TOP_K)
        found_docs = [r["doc_id"] for r in reranked]

        # Track average rerank score of top-5
        top5_scores = [r["rerank_score"] for r in reranked[:RERANK_TOP_K]]
        if top5_scores:
            avg_rerank_scores.append(sum(top5_scores) / len(top5_scores))

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

    print(f"\n=== Retrieval Metrics [{strategy} + reranker] (n={n}) ===")
    for k in sorted(recall_at):
        print(f"  Recall@{k}: {recall_at[k]:.3f}")
    print(f"  MRR:       {mrr:.3f}")
    print(f"  nDCG@10:   {ndcg_10:.3f}")
    print(f"  Misses:    {len(misses)} (not in top-{HYBRID_TOP_K})")
    if avg_rerank_scores:
        print(f"  Avg top-5 rerank score: {sum(avg_rerank_scores)/len(avg_rerank_scores):.4f}")

    # Save miss cases
    miss_path = f"results/retrieval_misses_{strategy}_reranked.json"
    with open(miss_path, "w") as f:
        json.dump(misses, f, indent=2)
    print(f"  Saved miss cases → {miss_path}")

    # Append to ablation table
    _append_to_ablation_csv(strategy, recall_at, mrr, ndcg_10, n)

    return {
        "strategy": strategy,
        "reranker": "bge-reranker-v2-m3",
        "recall_at": recall_at,
        "mrr": mrr,
        "ndcg_10": ndcg_10,
        "n_eval": n,
    }


def _append_to_ablation_csv(strategy, recall_at, mrr, ndcg_10, n):
    """Append or update a reranked row in the ablation CSV."""
    fieldnames = [
        "chunking_strategy", "reranker",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
        "mrr", "ndcg_at_10", "n_eval",
    ]
    reranker_name = "bge-reranker-v2-m3"

    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # Remove existing row for this strategy + reranker (will replace it)
    rows = [r for r in rows if not (r.get("chunking_strategy") == strategy
                                     and r.get("reranker") == reranker_name)]

    new_row = {
        "chunking_strategy": strategy,
        "reranker": reranker_name,
        "recall_at_1": f"{recall_at[1]:.3f}",
        "recall_at_5": f"{recall_at[5]:.3f}",
        "recall_at_10": f"{recall_at[10]:.3f}",
        "recall_at_20": f"{recall_at[20]:.3f}",
        "mrr": f"{mrr:.3f}",
        "ndcg_at_10": f"{ndcg_10:.3f}",
        "n_eval": str(n),
    }
    rows.append(new_row)

    # Sort: fixed → recursive → semantic, then none → reranker
    order = {"fixed": 0, "recursive": 1, "semantic": 2}
    rows.sort(key=lambda r: (order.get(r.get("chunking_strategy", ""), 99), r.get("reranker", "")))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Updated {RESULTS_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Run reranked retrieval evaluation")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"],
                        help="Which chunking strategy to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all three strategies with reranking")
    args = parser.parse_args()

    if args.all:
        for strategy in ["fixed", "recursive", "semantic"]:
            evaluate_strategy_reranked(strategy)
    elif args.strategy:
        evaluate_strategy_reranked(args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
