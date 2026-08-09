"""Merge extracted judgment text with authoritative case metadata."""

from pathlib import Path

import pandas as pd

METADATA_PATH = Path("data/raw/corpus_slice.parquet")
EXTRACTED_PATH = Path("data/processed/judgments_extracted.parquet")
OUTPUT_PATH = Path("data/processed/corpus_final.parquet")
MIN_CHARS = 500
MIN_DOCUMENTS = 4_000


def main() -> None:
    metadata = pd.read_parquet(METADATA_PATH).copy()
    extracted = pd.read_parquet(EXTRACTED_PATH).copy()

    metadata["pdf_stem"] = (
        metadata["source_filename"].str.replace(".json", "", regex=False) + "_EN"
    )
    metadata = metadata.drop_duplicates("pdf_stem")
    extracted = extracted.drop_duplicates("pdf_stem", keep="last")

    good_text = extracted[
        extracted["text"].notna()
        & extracted["text"].str.strip().ne("")
        & extracted["char_count"].gt(MIN_CHARS)
    ].copy()

    corpus = good_text.merge(
        metadata[["pdf_stem", "case_title", "party_caption"]], on="pdf_stem", how="inner"
    )
    corpus = corpus.drop_duplicates("pdf_stem")
    corpus["case_title"] = corpus["case_title"].fillna(corpus["party_caption"])
    corpus["case_title"] = corpus["case_title"].fillna(corpus["pdf_stem"])

    missing_text = set(metadata["pdf_stem"]) - set(corpus["pdf_stem"])
    print(f"Metadata documents: {len(metadata)}")
    print(f"Extracted documents: {len(extracted)}")
    print(f"Full-text corpus documents: {len(corpus)}")
    print(f"Documents rejected/missing: {len(missing_text)}")
    if missing_text:
        print("First rejected/missing IDs:", sorted(missing_text)[:20])

    if len(corpus) < MIN_DOCUMENTS:
        raise RuntimeError(
            f"Corpus incomplete: {len(corpus)} documents; expected at least {MIN_DOCUMENTS}"
        )
    if corpus["case_title"].isna().any():
        raise RuntimeError("Corpus contains missing case titles")

    temporary = OUTPUT_PATH.with_suffix(".parquet.tmp")
    corpus.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
