from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
MDB_PATH = ROOT / "data" / "raw" / "ntsb_avdata" / "avall" / "avall.mdb"
OUT_DIR = ROOT / "data" / "processed" / "ntsb_evidence_maturation"


CORE_TABLES = ["events", "aircraft", "narratives", "Findings"]


FINDING_ROWS_SQL = """
SELECT
    e.ev_id,
    e.ntsb_no,
    e.ev_type,
    e.ev_date,
    e.ev_year,
    e.ev_month,
    e.ev_city,
    e.ev_state,
    e.ev_country,
    e.latitude,
    e.longitude,
    e.light_cond,
    e.sky_cond_nonceil,
    e.sky_nonceil_ht,
    e.sky_ceil_ht,
    e.sky_cond_ceil,
    e.vis_sm,
    e.wx_temp,
    e.wx_dew_pt,
    e.wind_dir_deg,
    e.wind_vel_kts,
    e.gust_kts,
    e.altimeter,
    e.metar,
    e.wx_cond_basic,
    e.ev_highest_injury,
    e.inj_tot_f,
    e.inj_tot_s,
    e.inj_tot_m,
    e.inj_tot_n,
    a.Aircraft_Key,
    a.regis_no,
    a.far_part,
    a.damage,
    a.acft_make,
    a.acft_model,
    a.acft_series,
    a.cert_max_gr_wt,
    a.acft_category,
    a.num_eng,
    a.type_fly,
    a.phase_flt_spec,
    a.dprt_apt_id,
    a.dest_apt_id,
    n.narr_accp,
    n.narr_accf,
    n.narr_cause,
    n.narr_inc,
    f.finding_no,
    f.finding_code,
    f.finding_description,
    f.category_no,
    f.subcategory_no,
    f.section_no,
    f.subsection_no,
    f.modifier_no,
    f.Cause_Factor
FROM
    (((events AS e
    INNER JOIN aircraft AS a ON e.ev_id = a.ev_id)
    LEFT JOIN narratives AS n ON a.ev_id = n.ev_id AND a.Aircraft_Key = n.Aircraft_Key)
    INNER JOIN Findings AS f ON a.ev_id = f.ev_id AND a.Aircraft_Key = f.Aircraft_Key)
WHERE
    e.ev_year >= {start_year} AND e.ev_year <= {end_year}
"""


CASE_ROWS_SQL = """
SELECT
    e.ev_id,
    e.ntsb_no,
    e.ev_type,
    e.ev_date,
    e.ev_year,
    e.ev_month,
    e.ev_city,
    e.ev_state,
    e.ev_country,
    e.light_cond,
    e.vis_sm,
    e.wx_temp,
    e.wx_dew_pt,
    e.wind_dir_deg,
    e.wind_vel_kts,
    e.metar,
    e.wx_cond_basic,
    e.ev_highest_injury,
    a.Aircraft_Key,
    a.damage,
    a.acft_make,
    a.acft_model,
    a.acft_category,
    a.num_eng,
    a.phase_flt_spec,
    n.narr_accp,
    n.narr_accf,
    n.narr_cause,
    n.narr_inc
FROM
    ((events AS e
    INNER JOIN aircraft AS a ON e.ev_id = a.ev_id)
    LEFT JOIN narratives AS n ON a.ev_id = n.ev_id AND a.Aircraft_Key = n.Aircraft_Key)
WHERE
    e.ev_year >= {start_year} AND e.ev_year <= {end_year}
"""


def connect(mdb_path: Path):
    import win32com.client  # type: ignore

    conn = win32com.client.Dispatch("ADODB.Connection")
    conn.Open(
        "Provider=Microsoft.ACE.OLEDB.16.0;"
        f"Data Source={mdb_path};"
        "Persist Security Info=False;"
    )
    return conn


def iter_recordset(rs) -> Iterable[list[str]]:
    field_count = rs.Fields.Count
    while not rs.EOF:
        row = []
        for idx in range(field_count):
            value = rs.Fields.Item(idx).Value
            row.append("" if value is None else str(value))
        yield row
        rs.MoveNext()


def export_sql(conn, sql: str, out_path: Path) -> int:
    rs = conn.Execute(sql)[0]
    headers = [rs.Fields.Item(i).Name for i in range(rs.Fields.Count)]
    count = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in iter_recordset(rs):
            writer.writerow(row)
            count += 1
    rs.Close()
    return count


def export_table(conn, table_name: str, out_path: Path) -> int:
    return export_sql(conn, f"SELECT * FROM [{table_name}]", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--export-core", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(MDB_PATH)
    rows: list[tuple[str, int]] = []
    try:
        if args.export_core:
            for table in CORE_TABLES:
                rows.append((f"core_{table}.csv", export_table(conn, table, OUT_DIR / f"core_{table}.csv")))

        finding_sql = FINDING_ROWS_SQL.format(start_year=args.start_year, end_year=args.end_year)
        case_sql = CASE_ROWS_SQL.format(start_year=args.start_year, end_year=args.end_year)
        rows.append(
            (
                f"finding_rows_{args.start_year}_{args.end_year}.csv",
                export_sql(conn, finding_sql, OUT_DIR / f"finding_rows_{args.start_year}_{args.end_year}.csv"),
            )
        )
        rows.append(
            (
                f"case_aircraft_rows_{args.start_year}_{args.end_year}.csv",
                export_sql(conn, case_sql, OUT_DIR / f"case_aircraft_rows_{args.start_year}_{args.end_year}.csv"),
            )
        )
    finally:
        conn.Close()

    summary_path = OUT_DIR / "export_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "rows"])
        writer.writerows(rows)

    for name, count in rows:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
