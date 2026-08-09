# Production RAG System — Indian Supreme Court Judgments (Legal Domain)

A hybrid RAG (BM25 + dense retrieval + reranking) system over 4,140 Indian
Supreme Court judgments (2018-2022), with a full evaluation harness
(RAGAS + custom hallucination checker) and a working FastAPI layer.

See `results/final_report.md` for the full technical write-up and results.

---

## Setup

### 1. Environment
```bash
conda create -n rag-legal python=3.10 -y
conda activate rag-legal
pip install -r requirements.txt
```

### 2. GPU / CUDA (if applicable)
This project uses a local NVIDIA GPU for embeddings, reranking, and (via
Ollama) generation. If `torch.cuda.is_available()` returns `False` despite
having a GPU, your installed PyTorch build may not match your driver's
CUDA version. Check with `nvidia-smi` (see "CUDA Version" in the header)
and reinstall PyTorch matching that version, e.g.:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Ollama (LLM generation)
This project uses a locally-hosted Ollama server (not a paid API) for
generation, avoiding external API quota/cost issues.
- If Ollama is already running system-wide on your machine (check with
  `curl http://127.0.0.1:11434/api/version`), just pull the model:
```bash
  ollama pull qwen2.5:7b-instruct
```
- If you don't have root/sudo access, Ollama supports a manual, user-space
  install (no admin rights needed) -- see `ollama.com/download` for the
  `.tar.zst` archive, extract to a local directory, and add it to PATH.

### 4. KNOWN MANUAL FIX REQUIRED: ragas + langchain-community conflict
`ragas` (as of 0.3.x/0.4.x) has a broken import chain: it unconditionally
imports `langchain_community.chat_models.vertexai.ChatVertexAI`, which was
removed from newer `langchain-community` versions. This causes
`ModuleNotFoundError` on `from ragas import evaluate`, even though this
project never uses VertexAI.

**Fix**: create a stub module at the exact broken import path:
```bash
COMMUNITY_PATH=$(python -c "import langchain_community; print(langchain_community.__path__[0])")
cat > "$COMMUNITY_PATH/chat_models/vertexai.py" << 'PYEOF'
class ChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Stub only -- not used, satisfies ragas's outdated import.")
PYEOF
```
This is required after every fresh `pip install` of `langchain-community`,
since it's a site-packages patch, not something `requirements.txt` can
capture automatically. See `results/experiment_log.md` for the full
investigation.

### 5. Environment variables
```bash
cp .env.example .env   # if provided, else create manually:
export LLM_PROVIDER=ollama
export LLM_MODEL="qwen2.5:7b-instruct"
```

---

## Running the pipeline (in order)

```bash
# 1. Corpus + eval set (see results/experiment_log.md for the full,
#    non-trivial data acquisition process -- PDF extraction from S3 tars,
#    IndicLegalQA fuzzy matching, etc.)
python -m src.ingestion.download_corpus
python -m src.ingestion.extract_pdfs
python -m src.ingestion.extract_text          # resumable, checkpointed
python -m src.ingestion.match_eval_set

# 2. Chunking (3 strategies)
python -m src.ingestion.run_fixed_chunking
python -m src.ingestion.run_recursive_chunking
python -m src.ingestion.run_semantic_chunking

# 3. Indexing (embed + push to local Qdrant + build BM25)
python -m src.ingestion.run_indexing_fixed
python -m src.indexing.bm25_index

# 4. Evaluation
python -m src.evaluation.retrieval_metrics
python -m src.evaluation.run_full_eval --strategy semantic
python -m src.evaluation.ragas_eval --input results/raw_outputs_semantic_reranked.json
python -m src.evaluation.hallucination_checker --input results/raw_outputs_semantic_reranked.json

# 5. Reproducibility manifest
python -m src.indexing.build_manifest
```

## Running the API

```bash
./start_api.sh   # sources .env, kills stale instances, polls /health until ready
```

Then visit `http://127.0.0.1:8000/docs` for interactive Swagger docs, or:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

**Note on startup time**: this project uses Qdrant in local/embedded mode
(not Docker) due to no-root access on the original development machine.
Cold-start time (loading 3 collections, ~524K points total) has been
observed to vary from ~15s to 5+ minutes depending on shared-server disk
cache pressure -- see `results/experiment_log.md` for the full
investigation. `start_api.sh` polls for readiness rather than using a
fixed timeout to handle this.

---

## Known infrastructure substitutions (and why)
| Blueprint called for | Used instead | Why |
|---|---|---|
| Qdrant via Docker | Qdrant local/embedded mode | No root access on dev HPC server |
| Redis (caching) | File-persisted semantic cache | Same reason |
| Paid LLM API (Gemini/OpenAI) | Local Ollama (qwen2.5:7b-instruct) | Free-tier API quota was `limit: 0` on available account |

All substitutions are functionally equivalent for this project's scope
and documented in detail in `results/experiment_log.md`.

## Repository structure
See `results/final_report.md` Section 2 for the full architecture, and
the original project structure doc for the folder layout.
