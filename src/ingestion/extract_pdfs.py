"""Extract the corpus PDFs from verified year TAR archives safely."""

import os
import tarfile
from pathlib import Path

import pandas as pd

TAR_DIR = Path("data/raw/tars")
OUT_DIR = Path("data/raw/pdfs")
METADATA_PATH = Path("data/raw/corpus_slice.parquet")


def _is_complete_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            return False
        tail_size = min(path.stat().st_size, 16_384)
        handle.seek(-tail_size, os.SEEK_END)
        return b"%%EOF" in handle.read()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(METADATA_PATH).copy()
    metadata["pdf_stem"] = metadata["source_filename"].str.replace(
        ".json", "", regex=False
    )
    metadata = metadata.drop_duplicates("pdf_stem")

    restored = 0
    already_valid = 0
    missing = []

    for year in sorted(metadata["source_path_year"].astype(int).unique()):
        tar_path = TAR_DIR / f"english_{year}.tar"
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing archive: {tar_path}")

        year_rows = metadata[metadata["source_path_year"].astype(int) == year]
        print(f"Year {year}: restoring {len(year_rows)} PDFs from {tar_path.name}")

        with tarfile.open(tar_path) as archive:
            members = {
                Path(member.name).name: member
                for member in archive.getmembers()
                if member.isfile()
            }

            for stem in year_rows["pdf_stem"]:
                destination = OUT_DIR / f"{stem}_EN.pdf"
                if _is_complete_pdf(destination):
                    already_valid += 1
                    continue

                candidates = (f"{stem}_EN.pdf", f"{stem}.pdf")
                member = next((members[name] for name in candidates if name in members), None)
                if member is None:
                    missing.append(f"{year}:{stem}")
                    continue

                source = archive.extractfile(member)
                if source is None:
                    missing.append(f"{year}:{stem}")
                    continue

                temporary = destination.with_suffix(".pdf.part")
                with source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)

                if not _is_complete_pdf(temporary):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"Archive member is not a complete PDF: {member.name}")
                os.replace(temporary, destination)
                restored += 1

    expected = metadata["pdf_stem"].nunique()
    valid = sum(_is_complete_pdf(path) for path in OUT_DIR.glob("*.pdf"))
    print(f"\nAlready valid: {already_valid}")
    print(f"Restored: {restored}")
    print(f"Valid PDFs: {valid}/{expected}")
    print(f"Missing archive members: {len(missing)}")
    if missing:
        print("First missing members:", missing[:20])
        raise RuntimeError(f"Could not restore {len(missing)} required PDFs")
    if valid != expected:
        raise RuntimeError(f"Expected {expected} valid PDFs, found {valid}")


if __name__ == "__main__":
    main()
