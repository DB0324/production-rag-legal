"""
Fixed-size chunking: split text into chunks of ~N tokens (approximated by
words) with overlap. Simplest baseline strategy.
"""
from src.chunking.chunker_base import make_chunk

CHUNK_SIZE_WORDS = 380      # roughly ~512 tokens for English legal text
OVERLAP_WORDS = 40


def chunk_document_fixed(doc_id: str, case_title: str, text: str):
    words = text.split()
    chunks = []
    start = 0
    idx = 0

    if len(words) == 0:
        return chunks

    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(make_chunk(doc_id, case_title, chunk_text, idx))
            idx += 1

        if end >= len(words):
            break
        start = end - OVERLAP_WORDS  # step forward with overlap

    return chunks
