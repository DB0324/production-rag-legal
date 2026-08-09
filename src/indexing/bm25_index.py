"""
BM25 sparse index builder. Tokenizes chunk text and builds a BM25 index
that can be queried alongside dense (Qdrant) retrieval for hybrid search.
Saved to disk via pickle so it doesn't need rebuilding on every query run.
"""
import re
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer

# Simple tokenizer that preserves legal citation patterns
# (avoids breaking "S.C.R." or "42 U.S.C." into meaningless fragments)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")

_stemmer = PorterStemmer()

# Hardcoded English stopwords (avoids NLTK data download requirement)
# Includes common legal filler words that hurt BM25 discrimination
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "up", "down", "also",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am",
    # Common legal filler words that don't help discrimination
    "said", "upon", "thereof", "herein", "aforesaid", "whereas",
})


def tokenize(text: str):
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(text)]
    return [_stemmer.stem(t) for t in tokens if t not in STOPWORDS]


def build_bm25_index(chunks_path: str, output_path: str):
    df = pd.read_parquet(chunks_path)
    print(f"Loaded {len(df)} chunks from {chunks_path}")

    print("Tokenizing...")
    tokenized_corpus = [tokenize(t) for t in df["text"].tolist()]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": df["chunk_id"].tolist(),
            "doc_ids": df["doc_id"].tolist(),
            "texts": df["text"].tolist(),
            "case_titles": df["case_title"].tolist(),
        }, f)

    print(f"Saved BM25 index to {output_path}")


if __name__ == "__main__":
    build_bm25_index("data/chunks/fixed_chunks.parquet", "data/chunks/fixed_bm25.pkl")
