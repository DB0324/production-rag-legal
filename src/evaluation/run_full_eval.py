"""
Week 4 checkpoint: run all eval-set questions through the full
query → retrieve → rerank → generate pipeline.

Saves raw outputs to results/raw_outputs_{strategy}.json for
downstream evaluation (RAGAS, hallucination checker, etc.).

Usage:
    python -m src.evaluation.run_full_eval --strategy fixed
    python -m src.evaluation.run_full_eval --strategy fixed --no-reranker
    python -m src.evaluation.run_full_eval --all
"""
import argparse
import json
import os
import time
from src.generation.pipeline import query_pipeline

EVAL_PATH = "data/eval/indiclegalqa_filtered.json"
RESULTS_DIR = "results"


def run_full_eval(strategy: str, use_reranker: bool = True, limit: int = None):
    """Run all eval questions through the pipeline and save raw outputs."""
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)
    if limit:
        eval_data = eval_data[:limit]

    reranker_tag = "reranked" if use_reranker else "no_reranker"
    output_path = os.path.join(RESULTS_DIR, f"raw_outputs_{strategy}_{reranker_tag}.json")

    print(f"\n{'='*60}")
    print(f"Full pipeline evaluation: {strategy} ({reranker_tag})")
    print(f"  Eval set: {len(eval_data)} questions")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    results = []
    total_tokens_in = 0
    total_tokens_out = 0
    errors = 0

    start_time = time.time()

    for i, item in enumerate(eval_data):
        try:
            result = query_pipeline(
                question=item["question"],
                strategy=strategy,
                use_reranker=use_reranker,
            )

            output = {
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_doc_id": item["doc_id"],
                "gold_case_name": item.get("case_title_corpus", item.get("case_name_qa", "")),
                "generated_answer": result["answer"],
                "citations": result["citations"],
                "chunks_used": result["chunks_used"],
                "latency": result["latency"],
                "tokens": result["tokens"],
                "confidence": result["confidence"],
                "avg_rerank_score": result.get("avg_rerank_score"),
                "model": result.get("model", ""),
            }
            results.append(output)

            total_tokens_in += result["tokens"]["in"]
            total_tokens_out += result["tokens"]["out"]

        except Exception as e:
            print(f"  ERROR on question {i+1}: {e}")
            results.append({
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_doc_id": item["doc_id"],
                "error": str(e),
            })
            errors += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(eval_data) - i - 1) / rate / 60
            print(f"  processed {i+1}/{len(eval_data)} "
                  f"({errors} errors, ~{eta:.1f} min remaining)")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"  Questions: {len(eval_data)}")
    print(f"  Errors: {errors}")
    print(f"  Total tokens: {total_tokens_in} in, {total_tokens_out} out")
    print(f"  Total time: {elapsed_total/60:.1f} min")
    print(f"  Avg time/query: {elapsed_total/len(eval_data):.1f}s")
    print(f"  Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline eval over the eval set")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"],
                        help="Chunking strategy to evaluate")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip reranking (pass hybrid results straight to generation)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of eval questions (for smoke testing)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all strategies (with reranker)")
    args = parser.parse_args()

    if args.all:
        for strategy in ["fixed", "recursive", "semantic"]:
            run_full_eval(strategy, use_reranker=True)
    elif args.strategy:
        run_full_eval(args.strategy, use_reranker=not args.no_reranker, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
