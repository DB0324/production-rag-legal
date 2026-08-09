#!/bin/bash
# =============================================================
# Production RAG Legal — Full Ablation Pipeline Runner
# Run from project root: bash run_ablation.sh
# =============================================================
set -e  # stop on first error
mkdir -p logs


echo "=============================================="
echo "  Production RAG Legal — Ablation Pipeline"
echo "=============================================="
echo ""

# ── 0. Pre-flight checks ──────────────────────────────────────
echo "[Step 0] Pre-flight checks..."

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Validate data contents before installing dependencies or starting long jobs.
python -m src.ingestion.verify_data

echo ""

# ── 1. Install dependencies (if needed) ──────────────────────
echo "[Step 1] Installing dependencies..."
pip install -q sentence-transformers rank-bm25 qdrant-client pandas pyarrow google-generativeai ragas datasets openai

echo ""

# ── 2. Index recursive chunks (~180 min on GPU) ──────────────
echo "=============================================="
echo "[Step 2] Indexing recursive chunks → Qdrant + BM25"
echo "  (~180 min for 253K chunks on GPU)"
echo "=============================================="
python -m src.ingestion.run_indexing_recursive 2>&1 | tee logs/indexing_recursive.log

echo ""

# ── 3. Index semantic chunks (~115 min on GPU) ────────────────
echo "=============================================="
echo "[Step 3] Indexing semantic chunks → Qdrant + BM25"
echo "  (~115 min for 164K chunks on GPU)"
echo "=============================================="
python -m src.ingestion.run_indexing_semantic 2>&1 | tee logs/indexing_semantic.log

echo ""

# ── 4. Retrieval eval — no reranker (all 3 strategies) ────────
echo "=============================================="
echo "[Step 4] Retrieval metrics (no reranker) — 3 strategies"
echo "=============================================="
python -m src.evaluation.run_retrieval_eval --all 2>&1 | tee logs/retrieval_eval_no_reranker.log

echo ""
echo "Current ablation table (3 rows):"
cat results/ablation_table.csv
echo ""

# ── 5. Retrieval eval — with reranker (all 3 strategies) ──────
echo "=============================================="
echo "[Step 5] Retrieval metrics (with bge-reranker-v2-m3) — 3 strategies"
echo "  (First run downloads reranker model ~1.1GB)"
echo "=============================================="
python -m src.evaluation.run_retrieval_eval_reranked --all 2>&1 | tee logs/retrieval_eval_reranked.log

echo ""
echo "Current ablation table (6 rows):"
cat results/ablation_table.csv
echo ""

# ── 6. Full pipeline eval (generation) ────────────────────────
# NOTE: Set your LLM API key before running this step.
# Uncomment ONE of these blocks:
#
# --- Gemini (recommended) ---
# export LLM_PROVIDER=gemini
# export GOOGLE_API_KEY="your-key-here"
#
# --- OpenAI ---
# export LLM_PROVIDER=openai
# export OPENAI_API_KEY="your-key-here"
#
# --- Ollama (local, free) ---
# export LLM_PROVIDER=ollama
# export LLM_MODEL=llama3.1

echo "=============================================="
echo "[Step 6] Full pipeline eval (retrieve → rerank → generate)"
echo "=============================================="

if [ -z "$LLM_PROVIDER" ]; then
    echo ""
    echo "⚠ WARNING: LLM_PROVIDER not set. Skipping generation + RAGAS + hallucination steps."
    echo "  Set it before running:"
    echo "    export LLM_PROVIDER=gemini"
    echo "    export GOOGLE_API_KEY=your-key"
    echo "  Then re-run steps 6-8 with:"
    echo "    bash run_generation.sh"
    echo ""
    echo "=============================================="
    echo "  RETRIEVAL ABLATION COMPLETE"
    echo "  Results: results/ablation_table.csv"
    echo "=============================================="
    exit 0
fi

# Run generation for the best strategy (fixed as default, change if needed)
BEST_STRATEGY="${BEST_STRATEGY:-fixed}"
echo "  Using strategy: $BEST_STRATEGY"
echo "  LLM provider: $LLM_PROVIDER"

python -m src.evaluation.run_full_eval --strategy "$BEST_STRATEGY" 2>&1 | tee logs/full_eval_${BEST_STRATEGY}.log

echo ""

# ── 7. RAGAS evaluation ──────────────────────────────────────
echo "=============================================="
echo "[Step 7] RAGAS evaluation (faithfulness, relevancy, precision, recall)"
echo "=============================================="
python -m src.evaluation.ragas_eval --input "results/raw_outputs_${BEST_STRATEGY}_reranked.json" 2>&1 | tee logs/ragas_eval.log

echo ""

# ── 8. Hallucination checking ────────────────────────────────
echo "=============================================="
echo "[Step 8] Claim-level hallucination checking"
echo "=============================================="
python -m src.evaluation.hallucination_checker --input "results/raw_outputs_${BEST_STRATEGY}_reranked.json" 2>&1 | tee logs/hallucination_check.log

echo ""

# ── 9. Summary ────────────────────────────────────────────────
echo "=============================================="
echo "  ALL DONE"
echo "=============================================="
echo ""
echo "Final ablation table:"
cat results/ablation_table.csv
echo ""
echo "Output files:"
ls -la results/
echo ""
echo "Logs:"
ls -la logs/
