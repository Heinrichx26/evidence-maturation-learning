from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv"
OUT_DIR = ROOT / "results" / "data_quality"
HASH_TARGETS = [
    ROOT / "data" / "raw" / "ntsb_avdata" / "avall.zip",
    ROOT / "data" / "raw" / "ntsb_avdata" / "avall" / "avall.mdb",
    ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv",
    ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "case_aircraft_rows_2008_2025.csv",
]


def nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_completeness(data_file: Path) -> pd.DataFrame:
    usecols = [
        "ev_id",
        "Aircraft_Key",
        "ev_year",
        "narr_accp",
        "narr_accf",
        "narr_cause",
        "finding_code",
        "finding_description",
    ]
    df = pd.read_csv(data_file, dtype=str, usecols=usecols, low_memory=False)
    unit_cols = ["ev_id", "Aircraft_Key"]
    rows: list[dict] = []
    for year, group in df.groupby("ev_year", dropna=False):
        units = group.drop_duplicates(unit_cols)
        unit_count = len(units)
        finding_units = group[nonempty(group["finding_description"]) | nonempty(group["finding_code"])].drop_duplicates(unit_cols)
        prelim_units = group[nonempty(group["narr_accp"])].drop_duplicates(unit_cols)
        factual_units = group[nonempty(group["narr_accf"])].drop_duplicates(unit_cols)
        cause_units = group[nonempty(group["narr_cause"])].drop_duplicates(unit_cols)
        row = {
            "year": int(float(year)),
            "event_aircraft_units": unit_count,
            "finding_rows": len(group),
            "units_with_final_finding": len(finding_units),
            "final_finding_rate": len(finding_units) / unit_count if unit_count else 0.0,
            "units_with_preliminary_narrative": len(prelim_units),
            "preliminary_narrative_rate": len(prelim_units) / unit_count if unit_count else 0.0,
            "units_with_mature_factual_narrative": len(factual_units),
            "mature_factual_narrative_rate": len(factual_units) / unit_count if unit_count else 0.0,
            "units_with_probable_cause_text": len(cause_units),
            "probable_cause_text_rate": len(cause_units) / unit_count if unit_count else 0.0,
        }
        row["analysis_role"] = (
            "main forward test" if row["year"] in {2023, 2024}
            else "freshness audit" if row["year"] == 2025
            else "training or robustness"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def build_summary(audit: pd.DataFrame) -> list[dict]:
    groups = [
        ("Closed-label benchmark", audit["year"].between(2008, 2025)),
        ("Prospective fitting window", audit["year"].between(2020, 2024)),
        ("Main forward test", audit["year"].isin([2023, 2024])),
        ("Freshness audit", audit["year"].eq(2025)),
    ]
    rows: list[dict] = []
    for name, mask in groups:
        part = audit[mask]
        units = int(part["event_aircraft_units"].sum())
        rows.append(
            {
                "set": name,
                "years": f"{int(part['year'].min())}-{int(part['year'].max())}" if not part.empty else "",
                "event_aircraft_units": units,
                "final_finding_rate": part["units_with_final_finding"].sum() / units if units else 0.0,
                "preliminary_narrative_rate": part["units_with_preliminary_narrative"].sum() / units if units else 0.0,
                "mature_factual_narrative_rate": part["units_with_mature_factual_narrative"].sum() / units if units else 0.0,
            }
        )
    return rows


def build_hash_manifest() -> list[dict]:
    rows: list[dict] = []
    for path in HASH_TARGETS:
        if path.exists():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_completeness(args.data_file)
    audit.to_csv(args.out_dir / "ntsb_stage_completeness_by_year.csv", index=False, encoding="utf-8-sig")
    write_csv(args.out_dir / "ntsb_stage_completeness_summary.csv", build_summary(audit))
    write_csv(args.out_dir / "source_hash_manifest.csv", build_hash_manifest())
    print(
        {
            "data_file": str(args.data_file),
            "years": [int(audit["year"].min()), int(audit["year"].max())],
            "rows": int(audit["finding_rows"].sum()),
            "output_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
