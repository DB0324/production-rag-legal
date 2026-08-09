"""
Benchmark embedding speed on a small sample, properly separating
model warm-up from actual throughput measurement.
"""
import time
import pandas as pd
from src.indexing.embed import embed_texts, get_embed_model

def main():
    df = pd.read_parquet("data/chunks/fixed_chunks.parquet")
    sample = df["text"].tolist()[:1000]

    print("Warming up model (load + first CUDA call, not timed)...")
    get_embed_model()
    _ = embed_texts(sample[:32], batch_size=32)  # throwaway warm-up call

    print(f"\nTiming real run on {len(sample)} chunks, batch_size=128...")
    start = time.time()
    embeddings = embed_texts(sample, batch_size=128)
    elapsed = time.time() - start

    print(f"\nTime for {len(sample)} chunks: {elapsed:.2f} seconds")
    rate = len(sample) / elapsed
    print(f"Rate: {rate:.1f} chunks/sec")

    for name, count in [("fixed", 107056), ("recursive", 253394), ("semantic", 163884)]:
        est_min = count / rate / 60
        print(f"Estimated time for {name} ({count} chunks): {est_min:.1f} minutes")

if __name__ == "__main__":
    main()
