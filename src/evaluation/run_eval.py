"""
Master evaluation orchestrator — runs all evaluation modules
and produces a complete results row for the ablation table.

Combines:
  - Retrieval metrics (Recall@k, MRR, nDCG@10)
  - RAGAS scores (faithfulness, answer relevancy, context precision/recall)
  - Hallucination rate (claim-level decomposition + verification)

Usage:
    python -m src.evaluation.run_eval --strategy fixed
    python -m src.evaluation.run_eval --strategy fixed --no-reranker
    python -m src.evaluation.run_eval --all
    python -m src.evaluation.run_eval --strategy fixed --skip-generation
        (skip generation + RAGAS + hallucination, only run retrieval metrics)
"""
import argparse
import json
import os
import csv
from src.evaluation.run_retrieval_eval import evaluate_strategy
from src.evaluation.run_retrieval_eval_reranked import evaluate_strategy_reranked


RESULTS_CSV = "results/ablation_table.csv"


def run_full_evaluation(strategy: str, use_reranker: bool = True, skip_generation: bool = False):
    """
    Run the complete evaluation pipeline for one strategy+reranker configuration.

    Steps:
      1. Retrieval metrics (always runs)
      2. Full pipeline eval (run_full_eval) → raw outputs
      3. RAGAS eval on raw outputs
      4. Hallucination checker on raw outputs
      5. Consolidate into one ablation table row
    """
    print(f"\n{'#'*60}")
    print(f"# FULL EVALUATION: {strategy} {'+ reranker' if use_reranker else '(no reranker)'}")
    print(f"{'#'*60}\n")

    # ── Step 1: Retrieval metrics ──
    print("\n── Step 1: Retrieval Metrics ──")
    if use_reranker:
        retrieval_result = evaluate_strategy_reranked(strategy)
    else:
        retrieval_result = evaluate_strategy(strategy)

    if skip_generation:
        print("\n  --skip-generation flag set. Stopping after retrieval metrics.")
        return retrieval_result

    # ── Step 2: Full pipeline eval ──
    print("\n── Step 2: Full Pipeline Eval (generation) ──")
    from src.evaluation.run_full_eval import run_full_eval
    run_full_eval(strategy, use_reranker=use_reranker)

    reranker_tag = "reranked" if use_reranker else "no_reranker"
    raw_outputs_path = f"results/raw_outputs_{strategy}_{reranker_tag}.json"

    # ── Step 3: RAGAS eval ──
    print("\n── Step 3: RAGAS Evaluation ──")
    try:
        from src.evaluation.ragas_eval import run_ragas_evaluation
        ragas_result = run_ragas_evaluation(raw_outputs_path)
        ragas_scores = ragas_result["aggregate"]
    except Exception as e:
        print(f"  WARNING: RAGAS evaluation failed: {e}")
        print("  Continuing without RAGAS scores...")
        ragas_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    # ── Step 4: Hallucination checker ──
    print("\n── Step 4: Hallucination Checker ──")
    try:
        from src.evaluation.hallucination_checker import check_hallucinations
        hall_result = check_hallucinations(raw_outputs_path)
        hall_rate = hall_result["aggregate"]["overall_hallucination_rate"]
    except Exception as e:
        print(f"  WARNING: Hallucination checker failed: {e}")
        print("  Continuing without hallucination rate...")
        hall_rate = None

    # ── Step 5: Consolidate ──
    print("\n── Step 5: Consolidating Results ──")
    _update_full_ablation_csv(
        strategy=strategy,
        use_reranker=use_reranker,
        retrieval_result=retrieval_result,
        ragas_scores=ragas_scores,
        hallucination_rate=hall_rate,
    )

    print(f"\n{'#'*60}")
    print(f"# EVALUATION COMPLETE: {strategy}")
    print(f"{'#'*60}")


def _update_full_ablation_csv(strategy, use_reranker, retrieval_result, ragas_scores, hallucination_rate):
    """Update the ablation table with all metrics."""
    fieldnames = [
        "chunking_strategy", "reranker",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
        "mrr", "ndcg_at_10",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "hallucination_rate",
        "n_eval",
    ]

    reranker_name = "bge-reranker-v2-m3" if use_reranker else "none"

    # Read existing rows
    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # Remove existing row for this config
    rows = [r for r in rows if not (r.get("chunking_strategy") == strategy
                                     and r.get("reranker") == reranker_name)]

    # Build new row
    recall_at = retrieval_result.get("recall_at", {})
    new_row = {
        "chunking_strategy": strategy,
        "reranker": reranker_name,
        "recall_at_1": f"{recall_at.get(1, 0):.3f}",
        "recall_at_5": f"{recall_at.get(5, 0):.3f}",
        "recall_at_10": f"{recall_at.get(10, 0):.3f}",
        "recall_at_20": f"{recall_at.get(20, 0):.3f}",
        "mrr": f"{retrieval_result.get('mrr', 0):.3f}",
        "ndcg_at_10": f"{retrieval_result.get('ndcg_10', 0):.3f}",
        "faithfulness": f"{ragas_scores['faithfulness']:.3f}" if ragas_scores.get("faithfulness") is not None else "",
        "answer_relevancy": f"{ragas_scores['answer_relevancy']:.3f}" if ragas_scores.get("answer_relevancy") is not None else "",
        "context_precision": f"{ragas_scores['context_precision']:.3f}" if ragas_scores.get("context_precision") is not None else "",
        "context_recall": f"{ragas_scores['context_recall']:.3f}" if ragas_scores.get("context_recall") is not None else "",
        "hallucination_rate": f"{hallucination_rate:.3f}" if hallucination_rate is not None else "",
        "n_eval": str(retrieval_result.get("n_eval", 250)),
    }
    rows.append(new_row)

    # Sort
    order = {"fixed": 0, "recursive": 1, "semantic": 2}
    rows.sort(key=lambda r: (order.get(r.get("chunking_strategy", ""), 99), r.get("reranker", "")))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Updated {RESULTS_CSV} with full metrics row")


def main():
    parser = argparse.ArgumentParser(description="Master evaluation orchestrator")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Only run retrieval metrics, skip generation+RAGAS+hallucination")
    parser.add_argument("--all", action="store_true",
                        help="Run full eval for all strategies with reranker")
    args = parser.parse_args()

    if args.all:
        for strategy in ["fixed", "recursive", "semantic"]:
            run_full_evaluation(strategy, use_reranker=True, skip_generation=args.skip_generation)
    elif args.strategy:
        run_full_evaluation(args.strategy, use_reranker=not args.no_reranker,
                           skip_generation=args.skip_generation)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
