"""
Dense embedding wrapper using bge-large-en-v1.5.
Reused across all three chunking-strategy indexes.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
_model = None

def get_embed_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{MODEL_NAME}' on: {device}")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def embed_texts(texts, batch_size=32):
    """Embed document texts (no instruction prefix)."""
    model = get_embed_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity = dot product
    )
    return embeddings


# BGE instruction prefix — tells the model this is a search query, not a document
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(query: str):
    """Embed a single query with the BGE instruction prefix for better retrieval."""
    model = get_embed_model()
    embedding = model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    return embedding
