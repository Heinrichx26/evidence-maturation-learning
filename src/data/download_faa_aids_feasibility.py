from __future__ import annotations

import csv
import html
import re
import http.cookiejar
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "faa_asias_aids"
OUT_DIR = ROOT / "results" / "external_robustness"
FAA_AIDS_URL = "https://www.asias.faa.gov/apex/f?p=100:189:::NO"
COOKIE_JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(COOKIE_JAR))
csv.field_size_limit(10_000_000)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with OPENER.open(req, timeout=90) as resp:
        return resp.read().decode("utf-8", errors="replace")


def download_binary(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with OPENER.open(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def parse_download_rows(page_html: str) -> list[dict[str, str]]:
    rows = []
    row_re = re.compile(r"<tr><td[^>]*headers=\"FILENAME\"[^>]*>(.*?)</td>.*?<a href=\"(apex_util\.get_blob.*?)\"", re.I | re.S)
    for filename, href in row_re.findall(page_html):
        rows.append({"filename": html.unescape(re.sub(r"<.*?>", "", filename)).strip(), "href": html.unescape(href)})
    return rows


def read_tab_file_from_zip(zip_path: Path) -> tuple[str, list[dict[str, str]]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        target = names[0]
        with zf.open(target) as f:
            text = f.read().decode("latin-1", errors="replace")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    return target, list(reader)


def year_from_record(record: dict[str, str]) -> int | None:
    candidates = [
        record.get("EVENT_DATE"),
        record.get("event_date"),
        record.get("DATE"),
        record.get("OCC_DATE"),
        record.get("occ_date"),
        record.get("c6"),
        record.get("c9"),
    ]
    for value in candidates:
        if not value:
            continue
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def summarize_records(filename: str, inner_name: str, records: list[dict[str, str]]) -> dict[str, object]:
    years = [year for record in records if (year := year_from_record(record)) is not None]
    columns = list(records[0].keys()) if records else []
    key_columns = [
        col
        for col in columns
        if col.upper() in {"AIDS_REPORT_NUMBER", "LOCAL_EVENT_ID", "EVENT_DATE", "LOCATION", "CITY", "STATE", "AIRCRAFT_MAKE", "AIRCRAFT_MODEL"}
        or col in {"c5", "c6", "c7", "c8", "c9"}
    ]
    return {
        "source": "FAA AIDS",
        "filename": filename,
        "inner_file": inner_name,
        "rows": len(records),
        "year_min": min(years) if years else "",
        "year_max": max(years) if years else "",
        "columns": len(columns),
        "key_columns": "; ".join(key_columns[:8]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page_html = fetch_text(FAA_AIDS_URL)
    (RAW_DIR / "faa_aids_download_page.html").write_text(page_html, encoding="utf-8")
    rows = parse_download_rows(page_html)
    targets = {"a2015_19.zip", "a2020_26.zip", "e2015_19.zip", "e3030_2026.zip"}
    selected = [row for row in rows if row["filename"] in targets]
    summaries = []
    for row in selected:
        full_url = urllib.parse.urljoin(FAA_AIDS_URL, row["href"])
        local_zip = RAW_DIR / row["filename"]
        if not local_zip.exists():
            download_binary(full_url, local_zip)
        inner_name, records = read_tab_file_from_zip(local_zip)
        summaries.append(summarize_records(row["filename"], inner_name, records))

    source_rows = [
        {
            "source": "FAA ASIAS AIDS",
            "role": "external preliminary/final accident-incident entry point",
            "downloadable": "yes",
            "local_check": f"{len(summaries)} zipped tab-delimited files inspected",
            "closed_final_findings": "no",
            "paper_use": "supplementary source-feasibility and matching audit",
        },
        {
            "source": "NASA ASRS Database Online",
            "role": "external safety narrative and expert-coded report source",
            "downloadable": "CSV export through database search",
            "local_check": "source policy inspected",
            "closed_final_findings": "no",
            "paper_use": "external text robustness",
        },
    ]
    write_csv(OUT_DIR / "faa_aids_file_summary.csv", summaries)
    write_csv(OUT_DIR / "external_source_feasibility.csv", source_rows)
    print(f"Wrote {OUT_DIR / 'faa_aids_file_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'external_source_feasibility.csv'}")


if __name__ == "__main__":
    main()
