"""
Hybrid retrieval: query both BM25 (sparse) and Qdrant (dense), then
fuse rankings using Reciprocal Rank Fusion (RRF).
"""
import pickle
from src.indexing.embed import embed_query
from src.indexing.qdrant_client import get_client
from src.indexing.bm25_index import tokenize

RRF_K = 60  # standard RRF constant
INTERNAL_TOP_K = 100  # retrieve more candidates before fusion


def load_bm25(bm25_path: str):
    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def bm25_search(bm25_data, query: str, top_k: int = 50):
    bm25 = bm25_data["bm25"]
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "chunk_id": bm25_data["chunk_ids"][i],
            "doc_id": bm25_data["doc_ids"][i],
            "case_title": bm25_data["case_titles"][i],
            "text": bm25_data["texts"][i],
            "score": float(scores[i]),
        }
        for i in ranked
    ]


def dense_search(collection_name: str, query: str, top_k: int = INTERNAL_TOP_K):
    client = get_client()
    query_vector = embed_query(query).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        {
            "chunk_id": r.payload["chunk_id"],
            "doc_id": r.payload["doc_id"],
            "case_title": r.payload["case_title"],
            "text": r.payload["text"],
            "score": float(r.score),
        }
        for r in response.points
    ]


def reciprocal_rank_fusion(result_lists, k: int = RRF_K, top_k: int = 20):
    """result_lists: list of ranked result lists (each a list of dicts with 'chunk_id')."""
    rrf_scores = {}
    chunk_lookup = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunk_lookup[cid] = item

    ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [
        {**chunk_lookup[cid], "rrf_score": rrf_scores[cid]}
        for cid in ranked_ids
    ]


def hybrid_search(query: str, collection_name: str, bm25_path: str, top_k: int = 20):
    bm25_data = load_bm25(bm25_path)
    bm25_results = bm25_search(bm25_data, query, top_k=INTERNAL_TOP_K)
    dense_results = dense_search(collection_name, query, top_k=INTERNAL_TOP_K)
    return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)
