"""
Recursive/structure-aware chunking: split on paragraph/section boundaries
first, fall back to fixed-size for oversized sections, then merge any
undersized fragments (page-margin artifacts, stray short lines) into
the following chunk so nothing tiny survives as its own unit.
"""
import re
from src.chunking.chunker_base import make_chunk

MAX_CHUNK_WORDS = 380
OVERLAP_WORDS = 40
MIN_CHUNK_CHARS = 40  # below this, merge into the next piece

PARAGRAPH_SPLIT_PATTERN = re.compile(r'\n\s*\d{1,3}\.\s+|\n{2,}')


def split_oversized(text: str):
    words = text.split()
    parts = []
    start = 0
    while start < len(words):
        end = start + MAX_CHUNK_WORDS
        parts.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS
    return parts


def merge_small_pieces(pieces, min_chars=MIN_CHUNK_CHARS):
    """Merge any piece shorter than min_chars into the next piece.
    If it's the last piece, merge into the previous one instead."""
    merged = []
    buffer = ""
    for piece in pieces:
        candidate = (buffer + " " + piece).strip() if buffer else piece
        if len(candidate) < min_chars:
            buffer = candidate  # keep accumulating
        else:
            merged.append(candidate)
            buffer = ""
    if buffer:  # leftover small buffer at the end
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)
    return merged


def chunk_document_recursive(doc_id: str, case_title: str, text: str):
    raw_sections = PARAGRAPH_SPLIT_PATTERN.split(text)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    final_pieces = []
    for section in raw_sections:
        word_count = len(section.split())
        if word_count <= MAX_CHUNK_WORDS:
            final_pieces.append(section)
        else:
            final_pieces.extend(split_oversized(section))

    final_pieces = merge_small_pieces(final_pieces)

    chunks = []
    for idx, piece in enumerate(final_pieces):
        if piece.strip():
            chunks.append(make_chunk(doc_id, case_title, piece, idx))

    return chunks
