#!/bin/bash
# Rebuild the complete 2018-2022 Supreme Court corpus and chunk artifacts.
set -euo pipefail

export PATH="/home/student/.conda/envs/rag-legal/bin:$PATH"

mkdir -p data/raw/tars data/raw/pdfs data/processed data/chunks logs

python - <<'PY'
import pandas as pd
path = "data/raw/corpus_slice.parquet"
df = pd.read_parquet(path)
required = {"source_filename", "source_path_year", "case_title"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Metadata missing columns: {sorted(missing)}")
print(f"Metadata OK: {len(df)} rows, {df['source_filename'].nunique()} unique judgments")
PY

for year in 2018 2019 2020 2021 2022; do
    archive="data/raw/tars/english_${year}.tar"
    url="https://indian-supreme-court-judgments.s3.amazonaws.com/data/tar/year=${year}/english/english.tar"
    if [ ! -f "$archive" ] || ! tar -tf "$archive" >/dev/null 2>&1; then
        echo "Downloading official $year archive..."
        curl -fL --retry 5 --retry-delay 5 -C - -o "$archive" "$url"
    fi
    tar -tf "$archive" >/dev/null
    size_mb=$(( $(stat -c%s "$archive") / 1024 / 1024 ))
    echo "Archive $year OK: ${size_mb} MB"
done

python -m src.ingestion.extract_pdfs 2>&1 | tee logs/extract_pdfs.log
python -m src.ingestion.extract_text 2>&1 | tee logs/extract_text.log
python -m src.ingestion.build_corpus 2>&1 | tee logs/build_corpus.log
python -m src.ingestion.run_fixed_chunking 2>&1 | tee logs/chunking_fixed.log
python -m src.ingestion.run_recursive_chunking 2>&1 | tee logs/chunking_recursive.log
python -m src.ingestion.run_semantic_chunking 2>&1 | tee logs/chunking_semantic.log
python - <<'PY'
from src.indexing.bm25_index import build_bm25_index
build_bm25_index("data/chunks/fixed_chunks.parquet", "data/chunks/fixed_bm25.pkl")
PY
python -m src.ingestion.verify_data
