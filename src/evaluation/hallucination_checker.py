"""
Claim-level hallucination checker — a custom evaluation module
independent of RAGAS.

Approach:
  1. Decompose the generated answer into atomic claims (via LLM)
  2. For each claim, verify whether it is supported by the retrieved chunks
  3. Report: % unsupported claims, list of unsupported claims per question

Uses FULL chunk text from Qdrant (not the 200-char preview) for accurate
verification -- same fix applied to ragas_eval.py.

Usage:
    python -m src.evaluation.hallucination_checker --input results/raw_outputs_fixed_reranked.json
"""
import argparse
import json
import os
import re
import time
from src.generation.generate import generate
from src.evaluation.ragas_eval import detect_strategy_from_input, fetch_full_chunk_texts, STRATEGY_COLLECTIONS


DECOMPOSE_PROMPT = """Break the following answer into a list of independent, atomic factual claims.
Each claim should be a single, verifiable statement.
Return the claims as a numbered list.

Answer:
{answer}

Claims:"""


VERIFY_PROMPT = """Given the following claim and source passages, determine if the claim is SUPPORTED or NOT SUPPORTED by the passages.

Claim: {claim}

Source Passages:
{context}

Respond with exactly one word: SUPPORTED or NOT_SUPPORTED"""


def decompose_answer(answer: str) -> list:
    prompt = DECOMPOSE_PROMPT.format(answer=answer)
    response = generate(prompt, temperature=0.0)
    text = response["text"]

    claims = []
    for line in text.strip().split("\n"):
        line = line.strip()
        match = re.match(r'^[\d]+[.)]\s*(.+)', line)
        if match:
            claims.append(match.group(1).strip())
        elif line.startswith("- "):
            claims.append(line[2:].strip())

    return claims


def verify_claim(claim: str, context_texts: list) -> dict:
    context = "\n\n".join([f"[Passage {i+1}]: {t}" for i, t in enumerate(context_texts)])
    prompt = VERIFY_PROMPT.format(claim=claim, context=context)
    response = generate(prompt, temperature=0.0)
    verdict = response["text"].strip().upper()

    supported = "SUPPORTED" in verdict and "NOT" not in verdict

    return {
        "claim": claim,
        "supported": supported,
        "raw_verdict": verdict,
        "tokens_used": response["tokens_in"] + response["tokens_out"],
    }


def check_hallucinations(input_path: str, output_path: str = None, max_questions: int = None):
    strategy = detect_strategy_from_input(input_path)
    collection_name = STRATEGY_COLLECTIONS[strategy]
    print(f"Detected strategy: {strategy} -> collection: {collection_name}")

    with open(input_path) as f:
        raw_outputs = json.load(f)

    valid = [
        r for r in raw_outputs
        if "error" not in r and "Insufficient information" not in r.get("generated_answer", "")
    ]

    if max_questions:
        valid = valid[:max_questions]

    print(f"\nLoaded {len(raw_outputs)} results, checking {len(valid)} valid answers")
    print("(Skipping errors and 'insufficient information' responses)\n")

    # Fetch full text for all unique chunks up front
    all_chunk_ids = set()
    for item in valid:
        for chunk in item.get("chunks_used", []):
            all_chunk_ids.add(chunk["chunk_id"])
    print(f"Fetching full text for {len(all_chunk_ids)} unique chunks from Qdrant...")
    full_text_map = fetch_full_chunk_texts(list(all_chunk_ids), collection_name)

    results = []
    total_claims = 0
    total_unsupported = 0
    total_tokens = 0

    start_time = time.time()

    for i, item in enumerate(valid):
        answer = item["generated_answer"]

        context_texts = []
        for chunk in item.get("chunks_used", []):
            full_text = full_text_map.get(chunk["chunk_id"], "")
            if not full_text:
                text = chunk.get("text_preview", "")
                if text.endswith("..."):
                    text = text[:-3]
                full_text = text
            context_texts.append(full_text)

        claims = decompose_answer(answer)

        if not claims:
            results.append({
                "question": item["question"],
                "answer": answer,
                "claims": [],
                "n_claims": 0,
                "n_unsupported": 0,
                "hallucination_rate": 0.0,
            })
            continue

        claim_results = []
        unsupported_count = 0
        question_tokens = 0

        for claim in claims:
            verdict = verify_claim(claim, context_texts)
            claim_results.append(verdict)
            if not verdict["supported"]:
                unsupported_count += 1
            question_tokens += verdict["tokens_used"]

        total_claims += len(claims)
        total_unsupported += unsupported_count
        total_tokens += question_tokens

        hall_rate = unsupported_count / len(claims) if claims else 0.0

        results.append({
            "question": item["question"],
            "answer": answer[:200] + "...",
            "claims": claim_results,
            "n_claims": len(claims),
            "n_unsupported": unsupported_count,
            "hallucination_rate": round(hall_rate, 3),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(valid) - i - 1) / rate / 60
            print(f"  checked {i+1}/{len(valid)} questions "
                  f"({total_claims} claims, {total_unsupported} unsupported, ~{eta:.1f} min remaining)")

    overall_hall_rate = total_unsupported / total_claims if total_claims > 0 else 0.0

    aggregate = {
        "total_questions": len(valid),
        "total_claims": total_claims,
        "total_unsupported": total_unsupported,
        "overall_hallucination_rate": round(overall_hall_rate, 3),
        "total_tokens_used": total_tokens,
    }

    print(f"\n=== Hallucination Check Summary ===")
    print(f"  Questions checked: {aggregate['total_questions']}")
    print(f"  Total claims: {aggregate['total_claims']}")
    print(f"  Unsupported claims: {aggregate['total_unsupported']}")
    print(f"  Overall hallucination rate: {aggregate['overall_hallucination_rate']:.1%}")
    print(f"  Tokens used: {aggregate['total_tokens_used']}")

    output = {
        "aggregate": aggregate,
        "per_question": results,
        "input_file": input_path,
    }

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"results/hallucination_{base}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Claim-level hallucination checker")
    parser.add_argument("--input", required=True, help="Path to raw_outputs JSON")
    parser.add_argument("--output", default=None, help="Path to save results")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to N questions (for cost control during testing)")
    args = parser.parse_args()

    check_hallucinations(args.input, args.output, args.max_questions)


if __name__ == "__main__":
    main()
