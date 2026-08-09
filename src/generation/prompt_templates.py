"""
Citation-forcing prompt templates for legal RAG generation.
The system prompt instructs the LLM to:
  1. Answer ONLY from the provided context chunks
  2. Cite each claim with [chunk_id] references
  3. Say "insufficient information" when context is inadequate
"""

SYSTEM_PROMPT = """You are a legal research assistant. You answer questions about Indian case law STRICTLY based on the provided context passages.

RULES:
1. Answer ONLY using information from the provided context passages. Do NOT use any external knowledge.
2. For every factual claim in your answer, cite the source using the format [Case Title, Doc ID].
3. If the provided context does not contain enough information to answer the question, respond with: "Insufficient information in the provided sources to answer this question."
4. Keep your answer concise and well-structured. Use bullet points for multiple findings.
5. If passages present conflicting information, note the conflict and cite both sources.
6. Do NOT speculate or infer beyond what the passages explicitly state."""


def build_prompt(question: str, chunks: list, include_scores: bool = False) -> str:
    """
    Build the full prompt with context chunks for the LLM.

    Args:
        question:       The user's question.
        chunks:         List of chunk dicts with keys: text, chunk_id, doc_id, case_title.
                        Optionally: rerank_score.
        include_scores: If True, include reranker scores in context (for debugging).

    Returns:
        The formatted prompt string (system + context + question).
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"--- Passage {i} ---"
        header += f"\nCase: {chunk.get('case_title', 'Unknown')}"
        header += f"\nDoc ID: {chunk.get('doc_id', 'Unknown')}"
        header += f"\nChunk ID: {chunk.get('chunk_id', 'Unknown')}"
        if include_scores and "rerank_score" in chunk:
            header += f"\nRelevance Score: {chunk['rerank_score']:.4f}"
        header += f"\n\n{chunk['text']}"
        context_parts.append(header)

    context_block = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT PASSAGES ===
{context_block}

=== QUESTION ===
{question}

=== YOUR ANSWER (cite sources using [Case Title, Doc ID]) ==="""

    return prompt


def build_sufficiency_check_prompt(question: str, chunks: list) -> str:
    """
    Build a prompt that asks the LLM to judge whether the context
    is sufficient to answer the question. Used as a pre-generation
    guardrail.

    Returns a prompt whose expected output is "SUFFICIENT" or "INSUFFICIENT".
    """
    context_texts = "\n\n".join([
        f"[Passage {i+1}]: {chunk['text'][:500]}..."
        for i, chunk in enumerate(chunks)
    ])

    return f"""Given the following question and context passages, determine if the passages contain enough information to provide a well-grounded answer.

Question: {question}

Context:
{context_texts}

Respond with exactly one word: SUFFICIENT or INSUFFICIENT"""
