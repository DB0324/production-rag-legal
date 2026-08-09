"""
Embed and index the recursive-chunking chunk set into Qdrant,
then build its BM25 index.
Processes in batches; resumable if interrupted (checks existing
point count in the collection and skips already-indexed rows).
Run from project root: python -m src.ingestion.run_indexing_recursive
"""
import os
import time
import pandas as pd
from src.indexing.embed import embed_texts
from src.indexing.qdrant_client import get_client, create_collection
from src.indexing.bm25_index import build_bm25_index
from qdrant_client.models import PointStruct

CHUNKS_PATH = "data/chunks/recursive_chunks.parquet"
COLLECTION_NAME = "legal_recursive"
BM25_OUTPUT = "data/chunks/recursive_bm25.pkl"
BATCH_SIZE = 500  # embed + upsert this many chunks per batch


def main():
    # ── Step 1: Dense index (Qdrant) ──
    df = pd.read_parquet(CHUNKS_PATH)
    total = len(df)
    print(f"Loaded {total} chunks from {CHUNKS_PATH}")

    recreate = os.getenv("RECREATE_INDEX") == "1"
    client = create_collection(COLLECTION_NAME, recreate=recreate)

    # Resume support: check how many points already exist
    existing_count = client.count(COLLECTION_NAME).count
    print(f"Already indexed: {existing_count} points")

    if existing_count >= total:
        print("Collection already fully indexed. Skipping dense indexing.")
    else:
        start_row = existing_count
        remaining_df = df.iloc[start_row:]
        print(f"Resuming from row {start_row}, {len(remaining_df)} chunks remaining")

        start_time = time.time()
        for batch_start in range(0, len(remaining_df), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(remaining_df))
            batch_df = remaining_df.iloc[batch_start:batch_end]

            texts = batch_df["text"].tolist()
            embeddings = embed_texts(texts, batch_size=128)

            # upsert with correct absolute point IDs (offset by start_row + batch_start)
            client_local = get_client()
            points = [
                PointStruct(
                    id=start_row + batch_start + i,
                    vector=embeddings[i].tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "case_title": row["case_title"],
                        "text": row["text"],
                        "chunk_index": int(row["chunk_index"]),
                    },
                )
                for i, (_, row) in enumerate(batch_df.iterrows())
            ]
            client_local.upsert(collection_name=COLLECTION_NAME, points=points)

            done_so_far = start_row + batch_end
            elapsed = time.time() - start_time
            rate = (batch_end) / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining_df) - batch_end) / rate / 60 if rate > 0 else 0
            print(f"  indexed {done_so_far}/{total} total ({batch_end}/{len(remaining_df)} this run), "
                  f"~{eta_min:.1f} min remaining")

        final_count = client.count(COLLECTION_NAME).count
        print(f"\nDense indexing done. Final point count in '{COLLECTION_NAME}': {final_count}")

    # ── Step 2: BM25 index ──
    print(f"\nBuilding BM25 index...")
    build_bm25_index(CHUNKS_PATH, BM25_OUTPUT)
    print("All done.")


if __name__ == "__main__":
    main()
