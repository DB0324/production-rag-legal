"""
Match IndicLegalQA question-answer pairs against our corpus by case name,
using normalized fuzzy matching (case names differ in formatting between
the two sources). Filters to only QA pairs whose source judgment exists
in our corpus_final.parquet.
"""
import json
import re
import pandas as pd
from rapidfuzz import fuzz, process

CORPUS_PATH = "data/processed/corpus_final.parquet"
QA_PATH = "data/eval/raw/IndicLegalQA Dataset/IndicLegalQA Dataset_10K_Revised.json"
OUTPUT_PATH = "data/eval/indiclegalqa_filtered.json"
MATCH_THRESHOLD = 85


def normalize_case_name(name) -> str:
    if name is None or not isinstance(name, str) or not name.strip():
        return ""
    name = name.lower()
    name = re.sub(r'\bversus\b|\bvs\.?\b|\bv\.\b', 'v', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def main():
    corpus = pd.read_parquet(CORPUS_PATH)

    before = len(corpus)
    corpus = corpus[corpus["case_title"].notna()].copy()
    print(f"Dropped {before - len(corpus)} corpus rows with missing case_title")

    corpus["norm_title"] = corpus["case_title"].apply(normalize_case_name)
    corpus = corpus[corpus["norm_title"] != ""].reset_index(drop=True)

    with open(QA_PATH) as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QA pairs, {len(corpus)} usable corpus documents")

    corpus_titles = corpus["norm_title"].tolist()

    matched = []
    unmatched_count = 0

    for i, qa in enumerate(qa_data):
        norm_q_name = normalize_case_name(qa.get("case_name"))
        if not norm_q_name:
            unmatched_count += 1
            continue

        result = process.extractOne(norm_q_name, corpus_titles, scorer=fuzz.token_sort_ratio)
        if result is None:
            unmatched_count += 1
            continue

        match_text, score, idx = result
        if score >= MATCH_THRESHOLD:
            corpus_row = corpus.iloc[idx]
            matched.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "case_name_qa": qa["case_name"],
                "case_title_corpus": corpus_row["case_title"],
                "doc_id": corpus_row["pdf_stem"],
                "match_score": score,
            })
        else:
            unmatched_count += 1

        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(qa_data)}, matched so far: {len(matched)}")

    print(f"\nTotal matched: {len(matched)}")
    print(f"Total unmatched: {unmatched_count}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(matched, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
