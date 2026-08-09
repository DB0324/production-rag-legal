"""
Extract text from judgment PDFs, parallelized, with periodic checkpointing
so network drops or crashes don't lose progress. Skips already-done files
if re-run.
"""
import pdfplumber
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count

PDF_DIR = Path("data/raw/pdfs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "judgments_extracted.parquet"
MIN_CHARS_PER_PAGE = 200
CHECKPOINT_EVERY = 200

def extract_one(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            text_parts = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n\n".join(text_parts)
        chars_per_page = len(full_text) / max(n_pages, 1)
        return {
            "pdf_stem": pdf_path.stem, "n_pages": n_pages,
            "char_count": len(full_text), "chars_per_page": round(chars_per_page, 1),
            "suspect_low_text": chars_per_page < MIN_CHARS_PER_PAGE, "text": full_text,
        }
    except Exception as e:
        return {
            "pdf_stem": pdf_path.stem, "n_pages": None, "char_count": 0,
            "chars_per_page": 0, "suspect_low_text": True, "text": "", "error": str(e),
        }

def main():
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    done_stems = set()
    if CHECKPOINT_PATH.exists():
        try:
            existing = pd.read_parquet(CHECKPOINT_PATH)
            done_stems = set(existing["pdf_stem"])
            print(f"Found existing checkpoint with {len(done_stems)} already-processed files.")
        except Exception as e:
            print(f"WARNING: Checkpoint file is corrupted ({e}). Starting fresh.")
            CHECKPOINT_PATH.unlink()  # delete the bad file

    remaining = [p for p in all_pdfs if p.stem not in done_stems]
    print(f"Total PDFs: {len(all_pdfs)}. Remaining to process: {len(remaining)}")

    if not remaining:
        print("Nothing left to do.")
        return

    n_workers = min(cpu_count(), 16)
    results = []
    if CHECKPOINT_PATH.exists():
        results = pd.read_parquet(CHECKPOINT_PATH).to_dict("records")

    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_one, remaining, chunksize=10)):
            results.append(r)
            if (i + 1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_parquet(CHECKPOINT_PATH)
                print(f"  checkpoint saved at {i+1}/{len(remaining)} new files ({len(results)} total)")

    pd.DataFrame(results).to_parquet(CHECKPOINT_PATH)
    df = pd.DataFrame(results)
    print(f"\nDone. Total in checkpoint: {len(df)}")
    print(f"Flagged as suspect: {df['suspect_low_text'].sum()}")

if __name__ == "__main__":
    main()
