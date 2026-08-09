"""
Guardrail test: confirm the pipeline correctly returns "insufficient
information" for genuinely out-of-corpus questions, instead of
hallucinating a plausible-sounding but baseless answer.
"""
from src.generation.pipeline import query_pipeline

OUT_OF_CORPUS_QUESTIONS = [
    "What is the boiling point of liquid nitrogen at sea level?",
    "Who won the 2023 Cricket World Cup final?",
    "What are the main ingredients in a traditional French croissant recipe?",
    "How does photosynthesis convert sunlight into chemical energy?",
    "What is the current population of Tokyo, Japan?",
]

def main():
    correct_guardrail = 0
    for q in OUT_OF_CORPUS_QUESTIONS:
        result = query_pipeline(
            question=q,
            strategy="semantic",
            use_reranker=True,
        )
        is_insufficient = "Insufficient information" in result["answer"]
        status = "PASS (correctly declined)" if is_insufficient else "FAIL (attempted to answer)"
        if is_insufficient:
            correct_guardrail += 1

        print(f"Q: {q}")
        print(f"  Avg rerank score: {result.get('avg_rerank_score')}")
        print(f"  Answer: {result['answer'][:150]}")
        print(f"  {status}")
        print()

    print(f"=== Guardrail Test Summary ===")
    print(f"  Correctly declined: {correct_guardrail}/{len(OUT_OF_CORPUS_QUESTIONS)}")

if __name__ == "__main__":
    main()
