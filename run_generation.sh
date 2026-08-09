#!/bin/bash
# =============================================================
# Run generation + RAGAS + hallucination steps separately.
# Use this after run_ablation.sh completes the retrieval steps,
# once you've set your LLM API key.
#
# Usage:
#   export LLM_PROVIDER=gemini
#   export GOOGLE_API_KEY=your-key
#   bash run_generation.sh [strategy]
#
# Examples:
#   bash run_generation.sh fixed
#   bash run_generation.sh recursive
#   BEST_STRATEGY=semantic bash run_generation.sh
# =============================================================
set -e

STRATEGY="${1:-${BEST_STRATEGY:-fixed}}"

echo "=============================================="
echo "  Generation + Evaluation Pipeline"
echo "  Strategy: $STRATEGY"
echo "  LLM: $LLM_PROVIDER / ${LLM_MODEL:-default}"
echo "=============================================="

if [ -z "$LLM_PROVIDER" ]; then
    echo "ERROR: LLM_PROVIDER not set."
    echo "  export LLM_PROVIDER=gemini  (or openai / ollama)"
    echo "  export GOOGLE_API_KEY=your-key"
    exit 1
fi

mkdir -p logs

# ── Step 1: Full pipeline eval ────────────────────────────────
echo ""
echo "[1/3] Running full pipeline eval (250 questions)..."
python -m src.evaluation.run_full_eval --strategy "$STRATEGY" 2>&1 | tee "logs/full_eval_${STRATEGY}.log"

# ── Step 2: RAGAS ─────────────────────────────────────────────
echo ""
echo "[2/3] Running RAGAS evaluation..."
python -m src.evaluation.ragas_eval \
    --input "results/raw_outputs_${STRATEGY}_reranked.json" 2>&1 | tee "logs/ragas_${STRATEGY}.log"

# ── Step 3: Hallucination checker ─────────────────────────────
echo ""
echo "[3/3] Running hallucination checker..."
python -m src.evaluation.hallucination_checker \
    --input "results/raw_outputs_${STRATEGY}_reranked.json" 2>&1 | tee "logs/hallucination_${STRATEGY}.log"

echo ""
echo "=============================================="
echo "  DONE — $STRATEGY"
echo "=============================================="
echo ""
echo "Results:"
ls -la results/*${STRATEGY}*
