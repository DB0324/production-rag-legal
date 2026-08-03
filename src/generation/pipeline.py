"""
End-to-end RAG pipeline: query → hybrid search → rerank → generate → response.

Configurable: chunking strategy, reranker on/off, LLM model.
Returns a structured result with answer, citations, latency breakdown.

Usage:
    from src.generation.pipeline import query_pipeline
    result = query_pipeline("What did the court hold in XYZ case?")
"""
import time
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank
from src.generation.prompt_templates import build_prompt
from src.generation.generate import generate, extract_citations

# Strategy → (collection_name, bm25_path)
STRATEGY_CONFIG = {
    "fixed": ("legal_fixed", "data/chunks/fixed_bm25.pkl"),
    "recursive": ("legal_recursive", "data/chunks/recursive_bm25.pkl"),
    "semantic": ("legal_semantic", "data/chunks/semantic_bm25.pkl"),
}

# Default reranker score threshold below which we trigger "insufficient info"
SUFFICIENCY_THRESHOLD = -5.0  # bge-reranker scores can be negative; tune after first runs


def query_pipeline(
    question: str,
    strategy: str = "fixed",
    use_reranker: bool = True,
    hybrid_top_k: int = 50,
    rerank_top_k: int = 10,
    generation_top_k: int = 5,
    temperature: float = 0.1,
    sufficiency_threshold: float = SUFFICIENCY_THRESHOLD,
) -> dict:
    """
    Full RAG pipeline.

    Args:
        question:              The user's question.
        strategy:              Chunking strategy: "fixed", "recursive", or "semantic".
        use_reranker:          Whether to apply cross-encoder reranking.
        hybrid_top_k:          How many results from hybrid search.
        rerank_top_k:          How many results to keep after reranking.
        generation_top_k:      How many chunks to include in the generation context.
        temperature:           LLM temperature.
        sufficiency_threshold: Minimum avg rerank score to proceed with generation.

    Returns:
        dict with keys: answer, citations, chunks_used, latency, tokens, confidence, strategy
    """
    timings = {}
    collection_name, bm25_path = STRATEGY_CONFIG[strategy]

    # ── Step 1: Hybrid retrieval ──
    t0 = time.time()
    hybrid_results = hybrid_search(
        query=question,
        collection_name=collection_name,
        bm25_path=bm25_path,
        top_k=hybrid_top_k,
    )
    timings["retrieval_s"] = round(time.time() - t0, 3)

    # ── Step 2: Reranking (optional) ──
    if use_reranker:
        t0 = time.time()
        reranked_results = rerank(question, hybrid_results, top_k=rerank_top_k)
        timings["rerank_s"] = round(time.time() - t0, 3)

        # Sufficiency check: if the best reranker score is too low, bail out
        avg_score = sum(r.get("rerank_score", 0) for r in reranked_results) / max(len(reranked_results), 1)
        context_chunks = reranked_results[:generation_top_k]
    else:
        timings["rerank_s"] = 0.0
        avg_score = None
        context_chunks = hybrid_results[:generation_top_k]

    # ── Step 3: Sufficiency guard ──
    if use_reranker and avg_score is not None and avg_score < sufficiency_threshold:
        return {
            "answer": "Insufficient information in the provided sources to answer this question.",
            "citations": [],
            "chunks_used": [_chunk_summary(c) for c in context_chunks],
            "latency": timings,
            "tokens": {"in": 0, "out": 0},
            "confidence": "low",
            "strategy": strategy,
            "reranker": "bge-reranker-v2-m3" if use_reranker else "none",
            "avg_rerank_score": avg_score,
        }

    # ── Step 4: Generation ──
    prompt = build_prompt(question, context_chunks)

    t0 = time.time()
    llm_response = generate(prompt, temperature=temperature)
    timings["generation_s"] = round(time.time() - t0, 3)

    answer = llm_response["text"]
    citations = extract_citations(answer)

    # Total latency
    timings["total_s"] = round(sum(timings.values()), 3)

    return {
        "answer": answer,
        "citations": citations,
        "chunks_used": [_chunk_summary(c) for c in context_chunks],
        "latency": timings,
        "tokens": {
            "in": llm_response["tokens_in"],
            "out": llm_response["tokens_out"],
        },
        "confidence": "high" if (avg_score is not None and avg_score > 0) else "medium",
        "strategy": strategy,
        "reranker": "bge-reranker-v2-m3" if use_reranker else "none",
        "model": llm_response["model"],
        "avg_rerank_score": avg_score,
    }


def _chunk_summary(chunk: dict) -> dict:
    """Extract a compact summary of a chunk for the response (exclude full text)."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "case_title": chunk.get("case_title"),
        "text_preview": chunk.get("text", "")[:200] + "...",
        "rerank_score": chunk.get("rerank_score"),
        "rrf_score": chunk.get("rrf_score"),
    }
