"""
Step 1 (corrected): Filter to a small slice using streaming mode
BEFORE ever converting to pandas. Reads from local HF cache — no re-download.
"""

from datasets import load_dataset
import pandas as pd

TARGET_COURT = "Supreme Court of India"   # confirm exact string in next step
MIN_YEAR = 2018
MAX_YEAR = 2022
MAX_ROWS = 5000  # hard safety cap

def keep_record(record):
    court = record.get("court_name") or ""
    year = record.get("decision_year")
    if TARGET_COURT.lower() not in court.lower():
        return False
    if year is None:
        return False
    try:
        year = int(year)
    except (ValueError, TypeError):
        return False
    return MIN_YEAR <= year <= MAX_YEAR


def main():
    print("Loading dataset in streaming mode (reads from local cache)...")
    ds = load_dataset("KanoonGPT/indian-case-laws", split="train", streaming=True)

    print(f"Filtering for court containing '{TARGET_COURT}', years {MIN_YEAR}-{MAX_YEAR}...")
    filtered = ds.filter(keep_record)

    rows = []
    for i, record in enumerate(filtered):
        rows.append(record)
        if len(rows) % 200 == 0:
            print(f"  collected {len(rows)} matching rows so far...")
        if len(rows) >= MAX_ROWS:
            print(f"Hit safety cap of {MAX_ROWS} rows, stopping.")
            break

    print(f"\nTotal matching rows collected: {len(rows)}")

    if len(rows) == 0:
        print("No rows matched — check TARGET_COURT string against real court_name values.")
        return

    df = pd.DataFrame(rows)
    df.to_parquet("data/raw/corpus_slice.parquet")
    print("Saved filtered slice to data/raw/corpus_slice.parquet")
    print("\nColumns:", list(df.columns))
    print("\nSample row:")
    print(df.iloc[0][["case_title", "court_name", "decision_year"]])


if __name__ == "__main__":
    main()
