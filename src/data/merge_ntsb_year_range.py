from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ntsb_evidence_maturation"


def candidate_path(prefix: str, start_year: int, end_year: int) -> Path:
    return DATA_DIR / f"{prefix}_{start_year}_{end_year}.csv"


def collect_paths(prefix: str, start_year: int, end_year: int) -> list[Path]:
    paths: list[Path] = []
    year = start_year
    while year <= end_year:
        yearly = candidate_path(prefix, year, year)
        if yearly.exists():
            paths.append(yearly)
            year += 1
            continue

        found = None
        for stop in range(end_year, year, -1):
            block = candidate_path(prefix, year, stop)
            if block.exists():
                found = block
                break
        if found is None:
            raise FileNotFoundError(f"Missing {prefix} data for year {year}")
        paths.append(found)
        year = int(found.stem.rsplit("_", 1)[-1]) + 1
    return paths


def merge(prefix: str, start_year: int, end_year: int) -> tuple[str, int]:
    paths = collect_paths(prefix, start_year, end_year)
    out_path = candidate_path(prefix, start_year, end_year)
    tmp_path = out_path.with_suffix(".tmp")
    total = 0
    header: list[str] | None = None
    with tmp_path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = None
        for path in paths:
            with path.open("r", newline="", encoding="utf-8-sig") as src:
                reader = csv.reader(src)
                current_header = next(reader)
                if header is None:
                    header = current_header
                    writer = csv.writer(out)
                    writer.writerow(header)
                elif current_header != header:
                    raise ValueError(f"Header mismatch in {path}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    total += 1
    tmp_path.replace(out_path)
    return out_path.name, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    rows = [
        merge("finding_rows", args.start_year, args.end_year),
        merge("case_aircraft_rows", args.start_year, args.end_year),
    ]
    summary_path = DATA_DIR / f"merge_summary_{args.start_year}_{args.end_year}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "rows"])
        writer.writerows(rows)
    for name, count in rows:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
