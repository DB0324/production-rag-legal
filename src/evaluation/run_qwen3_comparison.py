"""
Generation comparison: same n=60 questions, qwen3:14b instead of
qwen2.5:7b-instruct. Run overnight given current heavy server load.
"""
import json
import os
os.environ["LLM_MODEL"] = "qwen3:14b"  # force this model regardless of shell env

from src.generation.pipeline import query_pipeline

SAMPLE_PATH = "data/eval/qwen3_test_sample60.json"
OUTPUT_PATH = "results/raw_outputs_semantic_reranked_qwen3_14b.json"


def main():
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)

    results = []
    for i, item in enumerate(sample):
        try:
            result = query_pipeline(
                question=item["question"],
                strategy="semantic",
                use_reranker=True,
            )
            results.append({
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_doc_id": item["doc_id"],
                "gold_case_name": item.get("case_title_corpus", ""),
                "generated_answer": result["answer"],
                "citations": result["citations"],
                "chunks_used": result["chunks_used"],
                "latency": result["latency"],
                "tokens": result["tokens"],
                "confidence": result["confidence"],
                "avg_rerank_score": result.get("avg_rerank_score"),
                "model": result.get("model", "qwen3:14b"),
            })
        except Exception as e:
            print(f"  ERROR on question {i+1}: {e}")
            results.append({"question": item["question"], "gold_answer": item["answer"],
                           "gold_doc_id": item["doc_id"], "error": str(e)})

        if (i + 1) % 5 == 0:
            print(f"  processed {i+1}/{len(sample)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
