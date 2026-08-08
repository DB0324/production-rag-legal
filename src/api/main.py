"""
FastAPI wrapper around the RAG pipeline.
Run: ./start_api.sh (sources .env, launches uvicorn)

Endpoints:
    POST /query    - ask a question, get a cited answer (cached + logged)
    GET  /health    - check Qdrant/Ollama connectivity
    GET  /metrics   - aggregate cost/latency/cache stats from logged queries
"""
import time
from fastapi import FastAPI
from pydantic import BaseModel
from src.generation.pipeline import query_pipeline
from src.observability.logger import log_query, get_metrics_summary
from src.caching.semantic_cache import check_cache, store_in_cache, cache_stats

app = FastAPI(title="Legal RAG API", version="1.0")


class QueryRequest(BaseModel):
    question: str
    strategy: str = "semantic"
    use_reranker: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list
    chunks_used: list
    latency: dict
    tokens: dict
    confidence: str
    avg_rerank_score: float | None = None
    cache_hit: bool = False


@app.get("/health")
def health():
    status = {"api": "ok"}

    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=5)
        status["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception as e:
        status["ollama"] = f"error: {e}"

    try:
        from src.indexing.qdrant_client import get_client
        client = get_client()
        collections = client.get_collections()
        status["qdrant"] = "ok"
        status["qdrant_collections"] = [c.name for c in collections.collections]
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    status["cache"] = cache_stats()

    return status


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()

    cached_result = check_cache(req.question)
    if cached_result is not None:
        cached_result = dict(cached_result)
        cached_result["cache_hit"] = True
        # Reflect actual (near-zero) cache-hit latency, not the original generation latency
        cached_result["latency"] = {**cached_result.get("latency", {}), "total_s": round(time.time() - t0, 3), "cache_hit": True}
        try:
            log_query(req.question, req.strategy, req.use_reranker, cached_result, cache_hit=True)
        except Exception as e:
            print(f"Warning: failed to log cached query: {e}")
        return cached_result

    result = query_pipeline(
        question=req.question,
        strategy=req.strategy,
        use_reranker=req.use_reranker,
    )
    result["cache_hit"] = False

    # Only cache successful, confident answers -- not "insufficient information" responses
    if result.get("confidence") != "low":
        try:
            store_in_cache(req.question, result)
        except Exception as e:
            print(f"Warning: failed to store in cache: {e}")

    try:
        log_query(req.question, req.strategy, req.use_reranker, result, cache_hit=False)
    except Exception as e:
        print(f"Warning: failed to log query: {e}")

    return result


@app.get("/metrics")
def metrics():
    return get_metrics_summary()


@app.get("/")
def root():
    return {
        "message": "Legal RAG API is running",
        "endpoints": ["/query (POST)", "/health (GET)", "/metrics (GET)"],
        "docs": "/docs",
    }
