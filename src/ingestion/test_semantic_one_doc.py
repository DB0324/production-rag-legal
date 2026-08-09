"""
Quick sanity test: run semantic chunking on ONE document, check speed
and output quality before committing to the full 4,140-doc corpus.
"""
import time
import pandas as pd
from src.chunking.semantic_chunker import chunk_document_semantic

def main():
    df = pd.read_parquet("data/processed/corpus_final.parquet")
    row = df.iloc[1]  # a normal-sized doc, not the 1076-page outlier

    print(f"Testing on: {row['case_title']}")
    print(f"Doc length: {row['char_count']} chars, {row['n_pages']} pages")

    start = time.time()
    chunks = chunk_document_semantic(row["pdf_stem"], row["case_title"], row["text"])
    elapsed = time.time() - start

    print(f"\nTime taken: {elapsed:.2f} seconds")
    print(f"Chunks produced: {len(chunks)}")
    print(f"Avg chars per chunk: {sum(c['char_count'] for c in chunks) / len(chunks):.1f}")
    print(f"\nEstimated time for full corpus (4140 docs): {elapsed * 4140 / 60:.1f} minutes")
    print(f"\nSample chunk 0:\n{chunks[0]['text'][:300]}")

if __name__ == "__main__":
    main()
