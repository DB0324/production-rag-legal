"""
API contract tests. Requires the API to be running:
    bash start_api.sh
    pytest tests/test_api.py -v

Skips cleanly if the API is not up, so the suite stays runnable offline.
"""
import os
import pytest
import requests

BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")


def _api_up():
    try:
        return requests.get(f"{BASE}/health", timeout=10).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="API not running")


def test_health_reports_dependencies():
    h = requests.get(f"{BASE}/health", timeout=30).json()
    assert h["api"] == "ok"
    assert "qdrant" in h and "ollama" in h


def test_empty_question_returns_400():
    r = requests.post(f"{BASE}/query", json={"question": ""}, timeout=30)
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_whitespace_question_returns_400():
    r = requests.post(f"{BASE}/query", json={"question": "   "}, timeout=30)
    assert r.status_code == 400


def test_unknown_strategy_returns_400():
    r = requests.post(f"{BASE}/query",
                      json={"question": "test", "strategy": "nope"}, timeout=30)
    assert r.status_code == 400
    assert "semantic" in r.json()["detail"]


def test_malformed_body_returns_422():
    r = requests.post(f"{BASE}/query", json={"bad": "body"}, timeout=30)
    assert r.status_code == 422


def test_metrics_shape():
    m = requests.get(f"{BASE}/metrics", timeout=30).json()
    assert "total_queries" in m
