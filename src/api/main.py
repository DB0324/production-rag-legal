"""
FastAPI wrapper around the RAG pipeline.
Run: ./start_api.sh (sources .env, launches uvicorn)

Endpoints:
    POST /query    - ask a question, get a cited answer (cached + logged)
    GET  /health   - check Qdrant/Ollama connectivity
    GET  /metrics  - aggregate cost/latency/cache stats from logged queries
"""
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.generation.pipeline import query_pipeline, STRATEGY_CONFIG
from src.observability.logger import log_query, get_metrics_summary
from src.caching.semantic_cache import check_cache, store_in_cache, cache_stats

app = FastAPI(title="Legal RAG API", version="1.1")

MAX_QUESTION_CHARS = 2000

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
        collections = get_client().get_collections()
        status["qdrant"] = "ok"
        status["qdrant_collections"] = [c.name for c in collections.collections]
    except Exception as e:
        status["qdrant"] = f"error: {e}"
    status["cache"] = cache_stats()
    return status

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()

    # ---- input validation (400s, not stack traces) ----
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"question exceeds {MAX_QUESTION_CHARS} characters",
        )
    if req.strategy not in STRATEGY_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy '{req.strategy}'. "
                   f"Valid: {sorted(STRATEGY_CONFIG)}",
        )

    # ---- cache ----
    try:
        cached_result = check_cache(question)
    except Exception as e:
        print(f"Warning: cache lookup failed: {e}")
        cached_result = None

    if cached_result is not None:
        cached_result = dict(cached_result)
        cached_result["cache_hit"] = True
        cached_result["latency"] = {
            **cached_result.get("latency", {}),
            "total_s": round(time.time() - t0, 3),
            "cache_hit": True,
        }
        try:
            log_query(question, req.strategy, req.use_reranker,
                      cached_result, cache_hit=True)
        except Exception as e:
            print(f"Warning: failed to log cached query: {e}")
        return cached_result

    # ---- pipeline, with graceful degradation ----
    try:
        result = query_pipeline(
            question=question,
            strategy=req.strategy,
            use_reranker=req.use_reranker,
        )
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e).lower()
        # Dependency outages are 503 (retryable), everything else 500
        if any(k in msg for k in
               ("connection", "timeout", "refused", "unavailable",
                "qdrant", "ollama", "already accessed")):
            raise HTTPException(
                status_code=503,
                detail=f"a backend dependency is unavailable: {e}",
            )
        raise HTTPException(status_code=500, detail=f"pipeline error: {e}")

    result["cache_hit"] = False

    if result.get("confidence") != "low":
        try:
            store_in_cache(question, result)
        except Exception as e:
            print(f"Warning: failed to store in cache: {e}")

    try:
        log_query(question, req.strategy, req.use_reranker, result, cache_hit=False)
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
