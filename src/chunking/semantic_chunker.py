"""
Semantic chunking (paragraph-level): split text into paragraphs (reusing
the structure-aware splitter), embed each paragraph, and merge consecutive
paragraphs into one chunk while similarity stays high; cut when it drops
or the max size is hit.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chunking.chunker_base import make_chunk
from src.chunking.recursive_chunker import PARAGRAPH_SPLIT_PATTERN, merge_small_pieces, split_oversized

MAX_CHUNK_WORDS = 380
SIMILARITY_THRESHOLD = 0.45
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on: {device}")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_paragraphs(text: str):
    raw = PARAGRAPH_SPLIT_PATTERN.split(text)
    raw = [s.strip() for s in raw if s.strip()]
    pieces = []
    for section in raw:
        if len(section.split()) <= MAX_CHUNK_WORDS:
            pieces.append(section)
        else:
            pieces.extend(split_oversized(section))
    return merge_small_pieces(pieces)


def chunk_document_semantic(doc_id: str, case_title: str, text: str):
    paragraphs = get_paragraphs(text)
    if len(paragraphs) == 0:
        return []
    if len(paragraphs) == 1:
        return [make_chunk(doc_id, case_title, paragraphs[0], 0)]

    model = get_model()
    embeddings = model.encode(paragraphs, show_progress_bar=False, batch_size=32)

    pieces = []
    current = [paragraphs[0]]
    current_words = len(paragraphs[0].split())

    for i in range(1, len(paragraphs)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        next_words = len(paragraphs[i].split())
        should_cut = (sim < SIMILARITY_THRESHOLD) or (current_words + next_words > MAX_CHUNK_WORDS)

        if should_cut:
            pieces.append(" ".join(current))
            current = [paragraphs[i]]
            current_words = next_words
        else:
            current.append(paragraphs[i])
            current_words += next_words

    if current:
        pieces.append(" ".join(current))

    chunks = []
    for idx, piece in enumerate(pieces):
        if piece.strip():
            chunks.append(make_chunk(doc_id, case_title, piece, idx))
    return chunks
