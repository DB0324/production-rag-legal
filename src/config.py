"""
Central configuration loader.

Frozen production settings live in config/config.yaml so they sit in one
auditable place instead of being duplicated across modules. Falls back to the
historical hardcoded defaults if the file is absent, so nothing breaks.

    from src.config import strategy_config, get
    threshold = get("guardrail.sufficiency_threshold", 0.10)
"""
import os

import yaml

_CONFIG_PATH = os.environ.get("RAG_CONFIG", "config/config.yaml")

_FALLBACK = {
    "active_strategy": "semantic",
    "strategies": {
        "fixed": {"collection": "legal_fixed",
                  "bm25_path": "data/chunks/fixed_bm25.pkl"},
        "recursive": {"collection": "legal_recursive",
                      "bm25_path": "data/chunks/recursive_bm25.pkl"},
        "semantic": {"collection": "legal_semantic",
                     "bm25_path": "data/chunks/semantic_bm25.pkl"},
    },
    "retrieval": {"rrf_k": 60, "hybrid_top_k": 50},
    "rerank": {"top_k": 10},
    "generation": {"top_k": 5, "temperature": 0.1},
    "guardrail": {"sufficiency_threshold": 0.10},
}


def load_config(path=None):
    path = path or _CONFIG_PATH
    if not os.path.exists(path):
        return _FALLBACK
    with open(path) as f:
        return yaml.safe_load(f) or _FALLBACK


CONFIG = load_config()


def strategy_config():
    """{name: (collection, bm25_path)} - the mapping duplicated across modules."""
    return {name: (s["collection"], s["bm25_path"])
            for name, s in CONFIG["strategies"].items()}


def get(path, default=None):
    """Dotted lookup, e.g. get('guardrail.sufficiency_threshold')."""
    node = CONFIG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
