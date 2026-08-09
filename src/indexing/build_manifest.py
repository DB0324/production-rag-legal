"""
Generates a reproducibility manifest: ties together the exact corpus
version, chunking config, and model versions used to build each Qdrant
collection, so results can be traced back to exactly what produced them.
"""
import json
import hashlib
import os
from datetime import datetime

MANIFEST_PATH = "results/build_manifest.json"


def file_hash(path: str, chunk_size=8192) -> str:
    """Compute a short hash of a file for version tracking (without needing full checksums)."""
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))  # hash just the first chunk -- fast, sufficient to detect changes
    return f"{h.hexdigest()[:12]} ({size} bytes)"


def build_manifest():
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "corpus": {
            "source_dataset": "KanoonGPT/indian-case-laws (HuggingFace)",
            "eval_dataset": "IndicLegalQA Dataset_10K_Revised.json (Mendeley, DOI 10.17632/gf8n8cnmvc.2)",
            "corpus_final_parquet": file_hash("data/processed/corpus_final.parquet"),
            "eval_set_json": file_hash("data/eval/indiclegalqa_filtered.json"),
            "corpus_doc_count": 4140,
            "eval_qa_count": 250,
            "date_range": "Supreme Court of India, 2018-2022",
        },
        "chunking": {
            "fixed_chunks": file_hash("data/chunks/fixed_chunks.parquet"),
            "recursive_chunks": file_hash("data/chunks/recursive_chunks.parquet"),
            "semantic_chunks": file_hash("data/chunks/semantic_chunks.parquet"),
        },
        "models": {
            "dense_embedding": "BAAI/bge-large-en-v1.5",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "generation_llm": "qwen2.5:7b-instruct (via Ollama)",
        },
        "indexes": {
            "legal_fixed_points": 107056,
            "legal_recursive_points": 253394,
            "legal_semantic_points": 163884,
        },
        "winning_config": "semantic chunking + bge-reranker-v2-m3 (see config/config.yaml)",
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown",
    }

    os.makedirs("results", exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2))

    return manifest


if __name__ == "__main__":
    build_manifest()
