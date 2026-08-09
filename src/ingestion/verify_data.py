"""Validate the data artifacts required by the ablation pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    Path("data/chunks/fixed_chunks.parquet"): "parquet",
    Path("data/chunks/recursive_chunks.parquet"): "parquet",
    Path("data/chunks/semantic_chunks.parquet"): "parquet",
    Path("data/eval/indiclegalqa_filtered.json"): "json",
}


def main() -> int:
    failures = []
    failed_sizes = []

    for path, file_type in REQUIRED_FILES.items():
        try:
            if file_type == "parquet":
                dataframe = pd.read_parquet(path)
                print(f"  OK  {path}: {len(dataframe)} rows")
            else:
                with path.open(encoding="utf-8") as file:
                    data = json.load(file)
                print(f"  OK  {path}: {len(data)} entries")
        except Exception as error:
            failures.append(path)
            if path.exists():
                failed_sizes.append(path.stat().st_size)
            print(f"  FAIL {path}: {error}")

    if not failures:
        return 0

    print(
        "\nERROR: Required data files are missing or invalid. "
        "The pipeline cannot continue.",
        file=sys.stderr,
    )
    if len(failed_sizes) == len(failures) and len(set(failed_sizes)) == 1:
        size = failed_sizes[0]
        print(
            f"All failed files have the identical size {size:,} bytes. "
            "This strongly indicates that the files were truncated before "
            "they were added to data_transfer.zip.",
            file=sys.stderr,
        )
    print(
        "Recreate data_transfer.zip from the intact source files, verify the "
        "source Parquets with pandas.read_parquet(), and upload the new archive.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
