"""
Per-request observability logging: persists every /query call's
latency breakdown, token counts, and estimated cost to a local SQLite DB.
"""
import sqlite3
import json
import time
import os

DB_PATH = "results/query_logs.db"

# Ollama is self-hosted (zero API cost), but we still track "cost" as an
# estimated compute-cost proxy using token counts, useful for comparing
# configs even without real per-token billing.
COST_PER_1K_TOKENS_IN = 0.0   # $0 -- self-hosted Ollama, no billing
COST_PER_1K_TOKENS_OUT = 0.0  # $0 -- self-hosted Ollama, no billing


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            question TEXT,
            strategy TEXT,
            use_reranker INTEGER,
            confidence TEXT,
            avg_rerank_score REAL,
            retrieval_s REAL,
            rerank_s REAL,
            generation_s REAL,
            total_s REAL,
            tokens_in INTEGER,
            tokens_out INTEGER,
            estimated_cost_usd REAL,
            cache_hit INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_query(question: str, strategy: str, use_reranker: bool, result: dict, cache_hit: bool = False):
    """Log a single query result to the DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)

    latency = result.get("latency", {})
    tokens = result.get("tokens", {})
    tokens_in = tokens.get("in", 0)
    tokens_out = tokens.get("out", 0)
    cost = (tokens_in / 1000 * COST_PER_1K_TOKENS_IN) + (tokens_out / 1000 * COST_PER_1K_TOKENS_OUT)

    conn.execute("""
        INSERT INTO query_logs
        (timestamp, question, strategy, use_reranker, confidence, avg_rerank_score,
         retrieval_s, rerank_s, generation_s, total_s, tokens_in, tokens_out,
         estimated_cost_usd, cache_hit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        time.time(), question, strategy, int(use_reranker),
        result.get("confidence"), result.get("avg_rerank_score"),
        latency.get("retrieval_s"), latency.get("rerank_s"),
        latency.get("generation_s"), latency.get("total_s"),
        tokens_in, tokens_out, cost, int(cache_hit),
    ))
    conn.commit()
    conn.close()


def get_metrics_summary() -> dict:
    """Compute aggregate p50/p95 latency, cost, and cache hit-rate."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM query_logs ORDER BY timestamp DESC").fetchall()
    conn.close()

    if not rows:
        return {"total_queries": 0}

    total_s_values = sorted([r["total_s"] for r in rows if r["total_s"] is not None])
    n = len(total_s_values)

    def percentile(data, p):
        if not data:
            return None
        idx = int(len(data) * p)
        idx = min(idx, len(data) - 1)
        return data[idx]

    cache_hits = sum(r["cache_hit"] for r in rows)

    return {
        "total_queries": len(rows),
        "p50_latency_s": percentile(total_s_values, 0.50),
        "p95_latency_s": percentile(total_s_values, 0.95),
        "avg_latency_s": round(sum(total_s_values) / n, 3) if n else None,
        "total_tokens_in": sum(r["tokens_in"] or 0 for r in rows),
        "total_tokens_out": sum(r["tokens_out"] or 0 for r in rows),
        "total_estimated_cost_usd": sum(r["estimated_cost_usd"] or 0 for r in rows),
        "cache_hit_rate": round(cache_hits / len(rows), 3) if rows else 0,
        "cache_hits": cache_hits,
    }
