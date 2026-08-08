"""
FastAPI wrapper around the RAG pipeline.
Run: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query    - ask a question, get a cited answer (logged)
    GET  /health    - check Qdrant/Ollama connectivity
    GET  /metrics   - aggregate cost/latency/cache stats from logged queries
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.generation.pipeline import query_pipeline
from src.observability.logger import log_query, get_metrics_summary

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

    return status


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = query_pipeline(
        question=req.question,
        strategy=req.strategy,
        use_reranker=req.use_reranker,
    )
    try:
        log_query(req.question, req.strategy, req.use_reranker, result)
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
