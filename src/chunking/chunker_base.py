"""
Shared interface for all chunking strategies.
Every chunker takes a document dict and returns a list of Chunk dicts.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class Chunk:
    chunk_id: str       # unique id: f"{doc_id}_chunk_{n}"
    doc_id: str         # source document id (pdf_stem)
    case_title: str     # for citation/display later
    text: str           # the actual chunk text
    chunk_index: int    # position within the document
    char_count: int


def make_chunk(doc_id: str, case_title: str, text: str, chunk_index: int) -> Dict[str, Any]:
    return asdict(Chunk(
        chunk_id=f"{doc_id}_chunk_{chunk_index}",
        doc_id=doc_id,
        case_title=case_title,
        text=text.strip(),
        chunk_index=chunk_index,
        char_count=len(text.strip()),
    ))


def chunks_to_dataframe(all_chunks: List[Dict[str, Any]]):
    import pandas as pd
    return pd.DataFrame(all_chunks)
