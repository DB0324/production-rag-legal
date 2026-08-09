"""
Cross-encoder reranker using bge-reranker-v2-m3.
Takes top-N candidates from hybrid retrieval and reranks
by true query-document relevance.

Usage:
    from src.retrieval.reranker import rerank
    reranked = rerank(query, chunks, top_k=5)
"""
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def get_reranker():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker model '{MODEL_NAME}' on: {device}")
        _model = CrossEncoder(MODEL_NAME, device=device)
    return _model


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Rerank a list of chunk dicts by cross-encoder relevance to the query.

    Args:
        query:  The user query string.
        chunks: List of dicts, each must have a "text" key.
        top_k:  How many top results to return after reranking.

    Returns:
        List of chunk dicts (top_k), each with an added "rerank_score" field,
        sorted by descending rerank score.
    """
    if not chunks:
        return []

    model = get_reranker()

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        scored_chunk = {**chunk, "rerank_score": float(score)}
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored_chunks[:top_k]
