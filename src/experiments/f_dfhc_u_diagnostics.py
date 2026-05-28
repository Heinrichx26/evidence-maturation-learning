from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import ROOT, build_document, load_samples, predict_stage  # noqa: E402
from src.experiments.f_dfhc_u_closure import (  # noqa: E402
    OUT_DIR,
    evaluate_thresholds,
    exchangeability_diagnostics,
    fit_lr_stage,
    hierarchy_maps,
    prepare_fold,
    summarize_budgeted_positive_closure,
)


def write_csv(path: Path, rows: list[dict]) -> None:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_thresholds(path: Path, method: str, test_year: int, labels: list[str]) -> tuple[np.ndarray, np.ndarray, float, float]:
    df = pd.read_csv(path)
    part = df[(df["method"] == method) & (df["test_year"].astype(int) == int(test_year))].copy()
    by_label = part.set_index("label")
    pos = np.array([float(by_label.loc[label, "close_positive_threshold"]) for label in labels], dtype=float)
    neg = np.array([float(by_label.loc[label, "close_negative_threshold"]) for label in labels], dtype=float)
    aircraft_child_floor = float(part["aircraft_child_floor"].dropna().iloc[0]) if not part.empty else 0.0
    child_margin = float(part["child_margin"].dropna().iloc[0]) if not part.empty else 0.0
    return pos, neg, aircraft_child_floor, child_margin


def run(args: argparse.Namespace) -> None:
    start = time.time()
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    samples = load_samples(
        args.label_level,
        args.min_global_support,
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.data_file,
    )
    test_years = [int(year.strip()) for year in args.test_years.split(",") if year.strip()]
    suffix = "full" if args.full else "smoke"
    threshold_path = OUT_DIR / f"f_dfhc_u_thresholds_{'full' if args.threshold_source == 'full' else suffix}.csv"

    budget_rows: list[dict] = []
    drift_rows: list[dict] = []
    for test_year in test_years:
        calib_year = test_year - 1
        fit, cal, test, y_fit, y_cal, y_test, labels, _parents, _children = prepare_fold(samples, test_year, args.min_train_support)
        if fit.empty or cal.empty or test.empty or not labels:
            continue
        docs_fit = [build_document(row, "preliminary") for _, row in fit.iterrows()]
        docs_cal = [build_document(row, "preliminary") for _, row in cal.iterrows()]
        docs_test = [build_document(row, "preliminary") for _, row in test.iterrows()]
        vectorizer, model = fit_lr_stage(docs_fit, y_fit, jobs, args.max_features, args.c_value)
        scores_cal = predict_stage(vectorizer, model, docs_cal)
        scores_test = predict_stage(vectorizer, model, docs_test)
        maps = hierarchy_maps(labels)
        pos, neg, aircraft_child_floor, child_margin = load_thresholds(threshold_path, "F-DFHC-U full", test_year, labels)
        cal_metrics = evaluate_thresholds(
            scores_cal,
            y_cal,
            pos,
            neg,
            labels,
            maps,
            use_hierarchy=True,
            aircraft_child_floor=aircraft_child_floor,
            child_margin=child_margin,
        )
        test_metrics = evaluate_thresholds(
            scores_test,
            y_test,
            pos,
            neg,
            labels,
            maps,
            use_hierarchy=True,
            aircraft_child_floor=aircraft_child_floor,
            child_margin=child_margin,
        )
        budget_rows.extend(
            summarize_budgeted_positive_closure(
                "F-DFHC-U full",
                test_metrics,
                scores_test,
                y_test,
                test_year,
                calib_year,
            )
        )
        diag = exchangeability_diagnostics(y_cal, y_test, scores_cal, scores_test, cal_metrics, test_metrics)
        drift_rows.append(
            {
                "test_year": test_year,
                "calibration_year": calib_year,
                "calibration_samples": int(len(cal)),
                "test_samples": int(len(test)),
                **diag,
            }
        )

    write_csv(OUT_DIR / f"f_dfhc_u_budgeted_closure_{suffix}.csv", budget_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_exchangeability_diagnostics_{suffix}.csv", drift_rows)
    meta = {
        "seconds": round(time.time() - start, 3),
        "jobs": jobs,
        "test_years": test_years,
        "threshold_source": str(threshold_path),
        "output_suffix": suffix,
    }
    (OUT_DIR / f"run_meta_diagnostics_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--test-years", type=str, default="2023")
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--threshold-source", choices=["full", "same"], default="full")
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
