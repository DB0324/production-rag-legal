"""
Run fixed-size chunking over the full corpus and save results.
Run from project root: python src/ingestion/run_fixed_chunking.py
"""
import pandas as pd
from src.chunking.fixed_chunker import chunk_document_fixed
from src.chunking.chunker_base import chunks_to_dataframe

INPUT_PATH = "data/processed/corpus_final.parquet"
OUTPUT_PATH = "data/chunks/fixed_chunks.parquet"


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} documents.")

    all_chunks = []
    for i, row in df.iterrows():
        doc_chunks = chunk_document_fixed(
            doc_id=row["pdf_stem"],
            case_title=row["case_title"],
            text=row["text"],
        )
        all_chunks.extend(doc_chunks)

        if (i + 1) % 500 == 0:
            print(f"  chunked {i+1}/{len(df)} documents, {len(all_chunks)} chunks so far")

    chunk_df = chunks_to_dataframe(all_chunks)
    chunk_df.to_parquet(OUTPUT_PATH)

    print(f"\nDone. Total chunks: {len(chunk_df)}")
    print(f"Avg chunks per doc: {len(chunk_df) / len(df):.1f}")
    print(f"Avg chars per chunk: {chunk_df['char_count'].mean():.1f}")
    print(f"Min/Max chars per chunk: {chunk_df['char_count'].min()} / {chunk_df['char_count'].max()}")


if __name__ == "__main__":
    main()
