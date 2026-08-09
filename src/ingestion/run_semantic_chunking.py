"""
Run semantic (paragraph-level) chunking over the full corpus.
Model loads once and is reused across all documents.
Run from project root: python -m src.ingestion.run_semantic_chunking
"""
import time
import pandas as pd
from src.chunking.semantic_chunker import chunk_document_semantic, get_model
from src.chunking.chunker_base import chunks_to_dataframe

INPUT_PATH = "data/processed/corpus_final.parquet"
OUTPUT_PATH = "data/chunks/semantic_chunks.parquet"


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} documents.")

    get_model()  # force model load once, upfront, before timing starts
    start = time.time()

    all_chunks = []
    for i, row in df.iterrows():
        doc_chunks = chunk_document_semantic(
            doc_id=row["pdf_stem"],
            case_title=row["case_title"],
            text=row["text"],
        )
        all_chunks.extend(doc_chunks)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta_min = (len(df) - (i + 1)) / rate / 60
            print(f"  chunked {i+1}/{len(df)} docs, {len(all_chunks)} chunks so far, "
                  f"~{eta_min:.1f} min remaining")

    chunk_df = chunks_to_dataframe(all_chunks)
    chunk_df.to_parquet(OUTPUT_PATH)

    total_min = (time.time() - start) / 60
    print(f"\nDone in {total_min:.1f} minutes. Total chunks: {len(chunk_df)}")
    print(f"Avg chunks per doc: {len(chunk_df) / len(df):.1f}")
    print(f"Avg chars per chunk: {chunk_df['char_count'].mean():.1f}")
    print(f"Min/Max chars per chunk: {chunk_df['char_count'].min()} / {chunk_df['char_count'].max()}")


if __name__ == "__main__":
    main()
