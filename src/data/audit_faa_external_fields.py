from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path


_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = int(_limit / 10)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "faa_asias_aids"
OUT_DIR = ROOT / "results" / "external_robustness"

FIELD_GROUPS = {
    "event_date": ["c6", "c7", "c8", "c9"],
    "location": ["c11", "c12", "c13", "c14", "c16", "c17"],
    "aircraft_identity": ["c22", "c23", "c24", "c30"],
    "aircraft_system": ["c34", "c35", "c157", "c158"],
    "injury_damage": ["c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c76", "c97"],
    "phase_operation": ["c95", "c96"],
    "airport_surface": ["c117", "c143"],
    "narrative": ["c119"],
}


def nonmissing(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text not in {"\\N", "N/A", "NA", "NULL", "."})


def read_zip(path: Path) -> tuple[str, list[dict[str, str]]]:
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        target = names[0]
        text = zf.read(target).decode("latin-1", errors="replace")
    return target, list(csv.DictReader(text.splitlines(), delimiter="\t"))


def group_rate(records: list[dict[str, str]], fields: list[str]) -> float:
    if not records:
        return 0.0
    hits = 0
    for row in records:
        if any(nonmissing(row.get(field)) for field in fields):
            hits += 1
    return hits / len(records)


def summarize_file(path: Path) -> dict[str, object]:
    inner, records = read_zip(path)
    columns = list(records[0].keys()) if records else []
    row: dict[str, object] = {
        "source": "FAA ASIAS AIDS",
        "archive": path.name,
        "inner_file": inner,
        "records": len(records),
        "columns": len(columns),
    }
    years = []
    for record in records:
        value = record.get("c6") or ""
        if value.isdigit():
            years.append(int(value))
    row["year_min"] = min(years) if years else ""
    row["year_max"] = max(years) if years else ""
    for group, fields in FIELD_GROUPS.items():
        row[f"{group}_coverage"] = group_rate(records, fields)
    common_groups = ["event_date", "location", "aircraft_identity", "injury_damage", "phase_operation", "narrative"]
    row["ntsb_stage_field_group_overlap"] = len(common_groups)
    row["mean_common_group_coverage"] = sum(float(row[f"{group}_coverage"]) for group in common_groups) / len(common_groups)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [summarize_file(path) for path in sorted(RAW_DIR.glob("a*.zip"))]
    write_csv(OUT_DIR / "faa_external_field_audit.csv", rows)
    print(f"Wrote {OUT_DIR / 'faa_external_field_audit.csv'}")


if __name__ == "__main__":
    main()
