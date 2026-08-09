"""
Qdrant local-mode client setup and collection creation helper.
One collection per chunking strategy, storing both dense vectors
and the original text/metadata as payload.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_PATH = "data/qdrant_local"
VECTOR_SIZE = 1024  # bge-large-en-v1.5 output dimension

_client = None

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def create_collection(collection_name: str, recreate: bool = False):
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists, skipping creation.")
            return client

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: {collection_name}")
    return client


def upsert_chunks(collection_name: str, chunk_df, embeddings, batch_size=256):
    client = get_client()
    total = len(chunk_df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunk_df.iloc[start:end]
        batch_embeddings = embeddings[start:end]

        points = [
            PointStruct(
                id=start + i,
                vector=batch_embeddings[i].tolist(),
                payload={
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "case_title": row["case_title"],
                    "text": row["text"],
                    "chunk_index": int(row["chunk_index"]),
                },
            )
            for i, (_, row) in enumerate(batch.iterrows())
        ]
        client.upsert(collection_name=collection_name, points=points)

        if (end % 2000 == 0) or (end == total):
            print(f"  upserted {end}/{total} points into '{collection_name}'")
