"""
RAGAS evaluation: faithfulness, answer relevancy, context precision, context recall.
Reads raw pipeline outputs from run_full_eval.py and scores them.
Uses LOCAL models (Ollama + HuggingFace embeddings) instead of OpenAI defaults.
Fetches FULL chunk text from Qdrant (not the 200-char preview) for accurate scoring.

Usage:
    python -m src.evaluation.ragas_eval --input results/raw_outputs_fixed_reranked.json
    python -m src.evaluation.ragas_eval --input results/raw_outputs_fixed_reranked.json --output results/ragas_scores_fixed.json
"""
import argparse
import json
import os

STRATEGY_COLLECTIONS = {
    "fixed": "legal_fixed",
    "recursive": "legal_recursive",
    "semantic": "legal_semantic",
}


def detect_strategy_from_input(input_path: str) -> str:
    for strategy in STRATEGY_COLLECTIONS:
        if strategy in input_path:
            return strategy
    raise ValueError(f"Could not detect chunking strategy from filename: {input_path}")


def fetch_full_chunk_texts(chunk_ids: list, collection_name: str) -> dict:
    """Fetch full chunk text from Qdrant by chunk_id, returns {chunk_id: text}."""
    from src.indexing.qdrant_client import get_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_client()
    result = {}

    for cid in chunk_ids:
        points = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=cid))]
            ),
            limit=1,
            with_payload=True,
        )
        if points[0]:
            result[cid] = points[0][0].payload.get("text", "")
        else:
            result[cid] = ""

    return result


def get_local_llm_and_embeddings():
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model_name = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    chat_llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.0)
    ragas_llm = LangchainLLMWrapper(chat_llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    return ragas_llm, ragas_embeddings


def run_ragas_evaluation(input_path: str, output_path: str = None):
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    strategy = detect_strategy_from_input(input_path)
    collection_name = STRATEGY_COLLECTIONS[strategy]
    print(f"Detected strategy: {strategy} -> collection: {collection_name}")

    with open(input_path) as f:
        raw_outputs = json.load(f)

    valid = [r for r in raw_outputs if "error" not in r]
    print(f"Loaded {len(raw_outputs)} results, {len(valid)} valid (skipping {len(raw_outputs)-len(valid)} errors)")

    # Collect all chunk_ids we need to fetch full text for
    all_chunk_ids = set()
    for item in valid:
        for chunk in item.get("chunks_used", []):
            all_chunk_ids.add(chunk["chunk_id"])

    print(f"Fetching full text for {len(all_chunk_ids)} unique chunks from Qdrant...")
    full_text_map = fetch_full_chunk_texts(list(all_chunk_ids), collection_name)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in valid:
        questions.append(item["question"])
        answers.append(item["generated_answer"])

        ctx_texts = []
        for chunk in item.get("chunks_used", []):
            full_text = full_text_map.get(chunk["chunk_id"], "")
            if not full_text:
                # fallback to preview if fetch somehow failed
                text = chunk.get("text_preview", "")
                if text.endswith("..."):
                    text = text[:-3]
                full_text = text
            ctx_texts.append(full_text)
        contexts.append(ctx_texts)

        ground_truths.append(item["gold_answer"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print(f"\nRunning RAGAS evaluation on {len(dataset)} samples using LOCAL models (Ollama)...")
    print("  Metrics: faithfulness, answer_relevancy, context_precision, context_recall\n")

    local_llm, local_embeddings = get_local_llm_and_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=local_llm,
        embeddings=local_embeddings,
        run_config=RunConfig(timeout=600, max_workers=2),
    )

    scores_df = result.to_pandas()

    aggregate = {
        "faithfulness": float(scores_df["faithfulness"].mean()),
        "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
        "context_precision": float(scores_df["context_precision"].mean()),
        "context_recall": float(scores_df["context_recall"].mean()),
        "n_samples": len(scores_df),
    }

    print(f"\n=== RAGAS Aggregate Scores ===")
    for metric, score in aggregate.items():
        if metric != "n_samples":
            print(f"  {metric}: {score:.3f}")
    print(f"  n_samples: {aggregate['n_samples']}")

    per_question = []
    for i, row in scores_df.iterrows():
        per_question.append({
            "question": questions[i],
            "faithfulness": float(row["faithfulness"]) if not str(row["faithfulness"]) == "nan" else None,
            "answer_relevancy": float(row["answer_relevancy"]) if not str(row["answer_relevancy"]) == "nan" else None,
            "context_precision": float(row["context_precision"]) if not str(row["context_precision"]) == "nan" else None,
            "context_recall": float(row["context_recall"]) if not str(row["context_recall"]) == "nan" else None,
        })

    output = {
        "aggregate": aggregate,
        "per_question": per_question,
        "input_file": input_path,
    }

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"results/ragas_scores_{base}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved RAGAS scores → {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on pipeline outputs")
    parser.add_argument("--input", required=True, help="Path to raw_outputs JSON from run_full_eval")
    parser.add_argument("--output", default=None, help="Path to save RAGAS scores JSON")
    args = parser.parse_args()

    run_ragas_evaluation(args.input, args.output)


if __name__ == "__main__":
    main()
