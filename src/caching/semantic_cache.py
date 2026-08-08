"""
Semantic cache: embeds incoming queries and checks cosine similarity
against previously cached queries. On a hit (similarity >= threshold),
returns the cached response instead of re-running the full pipeline.

Substitution note: the original plan called for Redis, but this HPC
server has no root access (no `apt install redis-server`). A local,
file-persisted cache achieves the same functional goal (avoid redundant
LLM calls) without requiring a separate service -- a deliberate,
documented infrastructure-constraint substitution, consistent with the
same reasoning used for Qdrant's local mode instead of Docker.
"""
import json
import os
import numpy as np
from src.indexing.embed import embed_texts

CACHE_PATH = "data/semantic_cache.json"
SIMILARITY_THRESHOLD = 0.95
MAX_CACHE_SIZE = 500  # simple bound so the file doesn't grow unbounded

_cache = None  # in-memory: [{query, embedding, result}, ...]


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            raw = json.load(f)
        _cache = raw
    else:
        _cache = []
    return _cache


def _save_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(_cache[-MAX_CACHE_SIZE:], f)


def _cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def check_cache(question: str):
    """Returns cached result dict if a near-duplicate query exists, else None."""
    cache = _load_cache()
    if not cache:
        return None

    query_embedding = embed_texts([question], batch_size=1)[0]

    best_sim = -1.0
    best_entry = None
    for entry in cache:
        sim = _cosine_sim(query_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_sim >= SIMILARITY_THRESHOLD:
        return best_entry["result"]
    return None


def store_in_cache(question: str, result: dict):
    """Store a new query+result pair in the cache."""
    cache = _load_cache()
    query_embedding = embed_texts([question], batch_size=1)[0].tolist()

    cache.append({
        "query": question,
        "embedding": query_embedding,
        "result": result,
    })
    global _cache
    _cache = cache
    _save_cache()


def cache_stats():
    cache = _load_cache()
    return {"cache_size": len(cache), "cache_path": CACHE_PATH}
