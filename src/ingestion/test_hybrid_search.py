"""
Test the full hybrid search pipeline (BM25 + dense + RRF) using real
questions from our eval set against the fully-indexed legal_fixed collection.
"""
import json
from src.retrieval.hybrid_retriever import hybrid_search

COLLECTION_NAME = "legal_fixed"
BM25_PATH = "data/chunks/fixed_bm25.pkl"
EVAL_PATH = "data/eval/indiclegalqa_filtered.json"


def main():
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    # Test on 3 real questions
    for item in eval_data[:3]:
        print(f"Question: {item['question']}")
        print(f"Expected case: {item['case_title_corpus']}")
        print(f"Expected doc_id: {item['doc_id']}")

        results = hybrid_search(
            query=item["question"],
            collection_name=COLLECTION_NAME,
            bm25_path=BM25_PATH,
            top_k=10,
        )

        found_docs = [r["doc_id"] for r in results]
        hit = item["doc_id"] in found_docs
        rank = found_docs.index(item["doc_id"]) + 1 if hit else None

        print(f"Correct doc found in top-10: {hit} (rank: {rank})")
        print("Top 3 retrieved:")
        for r in results[:3]:
            print(f"  - {r['case_title'][:60]} (doc_id: {r['doc_id']})")
        print()


if __name__ == "__main__":
    main()
