from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import ROOT, build_document, load_samples, predict_stage, tune_thresholds  # noqa: E402
from src.experiments.innovation_smoke_methods import LATE_EMERGING_LABELS, fit_es3d, fit_lr_stage  # noqa: E402
from src.experiments.lfct_dfhc_closure import THRESHOLD_GRID  # noqa: E402


OUT_DIR = ROOT / "results" / "experiments" / "f_dfhc_closure"


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def empirical_bernstein_upper(losses: np.ndarray, candidate_count: int, delta: float) -> float:
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        return 1.0
    mean = float(values.mean())
    if values.size == 1:
        return min(1.0, mean)
    variance = float(values.var(ddof=1))
    log_term = math.log(max(2.0, 2.0 * max(1, candidate_count) / delta))
    radius = math.sqrt(2.0 * variance * log_term / values.size)
    radius += 7.0 * log_term / (3.0 * max(1, values.size - 1))
    return min(1.0, mean + radius)


def parent_of(label: str) -> str:
    return label.split("-", 1)[0].strip()


def hierarchy_projection(close_pos: np.ndarray, close_neg: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    projected_pos = close_pos.copy()
    projected_neg = close_neg.copy()
    projected_neg[projected_pos] = False

    violations = 0
    parents = sorted({parent_of(label) for label in labels})
    child_index = {parent: [idx for idx, label in enumerate(labels) if parent_of(label) == parent] for parent in parents}
    for row in range(projected_pos.shape[0]):
        for indices in child_index.values():
            pos_any = bool(projected_pos[row, indices].any())
            neg_all = bool(projected_neg[row, indices].all())
            if pos_any and neg_all:
                violations += 1
                projected_neg[row, indices] = False
    defer = ~(projected_pos | projected_neg)
    return projected_pos, projected_neg, defer, violations


def action_losses(
    scores: np.ndarray,
    y_true: np.ndarray,
    pos_thresholds: np.ndarray,
    neg_thresholds: np.ndarray,
    labels: list[str],
) -> dict:
    raw_pos = scores >= pos_thresholds
    raw_neg = scores <= neg_thresholds
    close_pos, close_neg, defer, violations = hierarchy_projection(raw_pos, raw_neg, labels)
    closed = close_pos | close_neg
    wrong = np.logical_or(np.logical_and(close_pos, y_true == 0), np.logical_and(close_neg, y_true == 1))
    false_positive = np.logical_and(close_pos, y_true == 0)

    late_idx = [idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS]
    k = y_true.shape[1]
    all_loss = np.logical_and(wrong, closed).sum(axis=1) / max(1, k)
    plus_loss = false_positive.sum(axis=1) / np.maximum(1, close_pos.sum(axis=1))
    hierarchy_loss = np.zeros(y_true.shape[0], dtype=float)
    if late_idx:
        late_loss = false_positive[:, late_idx].sum(axis=1) / len(late_idx)
    else:
        late_loss = np.zeros(y_true.shape[0], dtype=float)

    closed_total = int(closed.sum())
    wrong_total = int(np.logical_and(wrong, closed).sum())
    close_positive_total = int(close_pos.sum())
    close_positive_wrong = int(false_positive.sum())
    positive_truth = int(y_true.sum())
    true_positive_closed = int(np.logical_and(close_pos, y_true == 1).sum())
    late_close_positive = int(close_pos[:, late_idx].sum()) if late_idx else 0
    late_false_positive = int(false_positive[:, late_idx].sum()) if late_idx else 0

    return {
        "close_pos": close_pos,
        "close_neg": close_neg,
        "defer": defer,
        "closed": closed,
        "wrong": wrong,
        "all_loss": all_loss,
        "plus_loss": plus_loss,
        "hierarchy_loss": hierarchy_loss,
        "late_loss": late_loss,
        "hierarchy_violations": violations,
        "coverage": float(closed.mean()),
        "positive_coverage": float(true_positive_closed / max(1, positive_truth)),
        "instance_closed_precision": float(1.0 - all_loss.mean()),
        "aggregate_closed_precision": float(1.0 - wrong_total / closed_total) if closed_total else 0.0,
        "close_positive_precision": float(1.0 - close_positive_wrong / close_positive_total) if close_positive_total else 0.0,
        "close_positive_decisions": close_positive_total,
        "late_premature_closure_risk": float(late_loss.mean()),
        "late_close_positive_decisions": late_close_positive,
        "late_false_positive_closures": late_false_positive,
        "closed_decisions": closed_total,
        "wrong_closed_decisions": wrong_total,
    }


def candidate_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    positive_alpha: float,
    negative_alpha: float,
    late_positive_alpha: float,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    late = {idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS}
    n_labels = y_true.shape[1]
    pos = np.ones(n_labels, dtype=float)
    neg = np.zeros(n_labels, dtype=float)
    for idx in range(n_labels):
        grid = np.unique(
            np.concatenate(
                [
                    THRESHOLD_GRID,
                    np.quantile(scores[:, idx], [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
                ]
            )
        )
        pos_alpha = late_positive_alpha if idx in late else positive_alpha
        best_pos = (1.0, -1)
        for threshold in grid:
            mask = scores[:, idx] >= threshold
            count = int(mask.sum())
            if count < min_count:
                continue
            error = float(np.logical_and(mask, y_true[:, idx] == 0).sum() / count)
            if error <= pos_alpha and count > best_pos[1]:
                best_pos = (float(threshold), count)
        pos[idx] = best_pos[0]

        best_neg = (0.0, -1)
        for threshold in grid:
            mask = scores[:, idx] <= threshold
            count = int(mask.sum())
            if count < min_count:
                continue
            error = float(np.logical_and(mask, y_true[:, idx] == 1).sum() / count)
            if error <= negative_alpha and count > best_neg[1]:
                best_neg = (float(threshold), count)
        neg[idx] = best_neg[0]
    return pos, neg


def select_f_dfhc(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    labels: list[str],
    alpha_all: float,
    alpha_plus: float,
    alpha_late: float,
    delta: float,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    positive_grid = [0.00]
    negative_grid = [0.055, 0.060, 0.065, 0.070, 0.072, 0.073, 0.075, 0.080, 0.090]
    late_positive_grid = [0.00, 0.01, 0.02]
    min_count_grid = [5]
    candidate_count = len(positive_grid) * len(negative_grid) * len(late_positive_grid) * len(min_count_grid)
    candidates: list[dict] = []

    for pos_alpha in positive_grid:
        for neg_alpha in negative_grid:
            for late_alpha in late_positive_grid:
                for min_count in min_count_grid:
                    pos, neg = candidate_thresholds(scores_cal, y_cal, labels, pos_alpha, neg_alpha, late_alpha, min_count)
                    losses = action_losses(scores_cal, y_cal, pos, neg, labels)
                    row = {
                        "positive_alpha": pos_alpha,
                        "negative_alpha": neg_alpha,
                        "late_positive_alpha": late_alpha,
                        "min_count": min_count,
                        "candidate_count": candidate_count,
                        "coverage": losses["coverage"],
                        "positive_coverage": losses["positive_coverage"],
                        "empirical_all_risk": float(losses["all_loss"].mean()),
                        "empirical_plus_risk": float(losses["plus_loss"].mean()),
                        "empirical_hierarchy_risk": float(losses["hierarchy_loss"].mean()),
                        "empirical_late_risk": float(losses["late_loss"].mean()),
                        "upper_all_risk": empirical_bernstein_upper(losses["all_loss"], candidate_count, delta),
                        "upper_plus_risk": empirical_bernstein_upper(losses["plus_loss"], candidate_count, delta),
                        "upper_hierarchy_risk": 0.0,
                        "upper_late_risk": empirical_bernstein_upper(losses["late_loss"], candidate_count, delta),
                        "pos_thresholds": pos,
                        "neg_thresholds": neg,
                    }
                    row["feasible"] = bool(
                        row["upper_all_risk"] <= alpha_all
                        and row["upper_plus_risk"] <= alpha_plus
                        and row["upper_hierarchy_risk"] <= 0.0
                        and row["upper_late_risk"] <= alpha_late
                    )
                    candidates.append(row)

    feasible = [row for row in candidates if row["feasible"]]
    if feasible:
        best = max(feasible, key=lambda row: (row["coverage"], row["positive_coverage"], -row["upper_all_risk"]))
    else:
        best = min(candidates, key=lambda row: (row["upper_all_risk"], -row["coverage"]))
    candidate_rows = [{key: value for key, value in row.items() if not key.endswith("_thresholds")} for row in candidates]
    return best, best["pos_thresholds"], best["neg_thresholds"], candidate_rows


def prepare_fold(samples: pd.DataFrame, test_year: int, min_train_support: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    fit = samples[samples["ev_year"] < test_year - 1].copy()
    cal = samples[samples["ev_year"] == test_year - 1].copy()
    test = samples[samples["ev_year"] == test_year].copy()
    counts = Counter(label for labels in fit["labels"] for label in labels)
    labels = sorted(label for label, count in counts.items() if count >= min_train_support)
    mlb = MultiLabelBinarizer(classes=labels)
    y_fit = mlb.fit_transform(fit["labels"].map(lambda xs: [x for x in xs if x in labels]))
    y_cal = mlb.transform(cal["labels"].map(lambda xs: [x for x in xs if x in labels]))
    y_test = mlb.transform(test["labels"].map(lambda xs: [x for x in xs if x in labels]))
    keep = y_test.sum(axis=1) > 0
    test = test.loc[keep].copy()
    y_test = y_test[keep]
    return fit, cal, test, y_fit, y_cal, y_test, labels


def standard_threshold_actions(scores_cal: np.ndarray, y_cal: np.ndarray, scores_test: np.ndarray, y_test: np.ndarray, labels: list[str]) -> dict:
    thresholds = tune_thresholds(y_cal, scores_cal)
    pos = thresholds
    neg = thresholds
    metrics = action_losses(scores_test, y_test, pos, neg, labels)
    return row_from_losses("BR-TFIDF threshold", metrics, {}, {}, 0)


def row_from_losses(method: str, metrics: dict, selected: dict, calibration: dict, test_year: int) -> dict:
    return {
        "method": method,
        "test_year": test_year,
        "closed_precision": metrics["instance_closed_precision"],
        "aggregate_closed_precision": metrics["aggregate_closed_precision"],
        "close_positive_precision": metrics["close_positive_precision"],
        "positive_coverage": metrics["positive_coverage"],
        "overall_coverage": metrics["coverage"],
        "empirical_all_risk": float(metrics["all_loss"].mean()),
        "certified_all_bound": selected.get("upper_all_risk", math.nan),
        "certified_plus_bound": selected.get("upper_plus_risk", math.nan),
        "certified_late_bound": selected.get("upper_late_risk", math.nan),
        "hierarchy_violations": metrics["hierarchy_violations"],
        "late_premature_closure_risk": metrics["late_premature_closure_risk"],
        "closed_decisions": metrics["closed_decisions"],
        "wrong_closed_decisions": metrics["wrong_closed_decisions"],
        "close_positive_decisions": metrics["close_positive_decisions"],
        "calibration_coverage": calibration.get("coverage", math.nan),
        "calibration_all_risk": calibration.get("empirical_all_risk", math.nan),
        "positive_alpha": selected.get("positive_alpha", math.nan),
        "negative_alpha": selected.get("negative_alpha", math.nan),
        "late_positive_alpha": selected.get("late_positive_alpha", math.nan),
    }


def es3d_actions(scores_cal: np.ndarray, y_cal: np.ndarray, scores_test: np.ndarray, y_test: np.ndarray, labels: list[str], test_year: int) -> dict:
    es3d = fit_es3d(y_cal, y_test, scores_cal, scores_test, labels)
    late_set = {idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS}
    pos = np.array([es3d["late_close_positive_threshold"] if idx in late_set else es3d["other_close_positive_threshold"] for idx in range(len(labels))], dtype=float)
    neg = np.array([es3d["late_close_negative_threshold"] if idx in late_set else es3d["other_close_negative_threshold"] for idx in range(len(labels))], dtype=float)
    metrics = action_losses(scores_test, y_test, pos, neg, labels)
    return row_from_losses("ES3D", metrics, {}, {}, test_year)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    samples = load_samples(args.label_level, args.min_global_support)
    test_years = [2023, 2024, 2025] if args.full else [2023]
    suffix = "full" if args.full else "smoke"

    metric_rows: list[dict] = []
    candidate_rows: list[dict] = []
    threshold_rows: list[dict] = []
    fold_rows: list[dict] = []

    for test_year in test_years:
        fit, cal, test, y_fit, y_cal, y_test, labels = prepare_fold(samples, test_year, args.min_train_support)
        if fit.empty or cal.empty or test.empty or not labels:
            continue
        docs_fit = [build_document(row, "preliminary") for _, row in fit.iterrows()]
        docs_cal = [build_document(row, "preliminary") for _, row in cal.iterrows()]
        docs_test = [build_document(row, "preliminary") for _, row in test.iterrows()]
        vectorizer, model = fit_lr_stage(docs_fit, y_fit, jobs, args.max_features, args.c_value)
        scores_cal = predict_stage(vectorizer, model, docs_cal)
        scores_test = predict_stage(vectorizer, model, docs_test)

        metric_rows.append(standard_threshold_actions(scores_cal, y_cal, scores_test, y_test, labels) | {"test_year": test_year})
        metric_rows.append(es3d_actions(scores_cal, y_cal, scores_test, y_test, labels, test_year))

        selected, pos_thresholds, neg_thresholds, rows = select_f_dfhc(
            scores_cal,
            y_cal,
            labels,
            args.alpha_all,
            args.alpha_plus,
            args.alpha_late,
            args.delta,
        )
        for row in rows:
            row["test_year"] = test_year
        candidate_rows.extend(rows)
        cal_metrics = action_losses(scores_cal, y_cal, pos_thresholds, neg_thresholds, labels)
        test_metrics = action_losses(scores_test, y_test, pos_thresholds, neg_thresholds, labels)
        metric_rows.append(row_from_losses("F-DFHC", test_metrics, selected, selected, test_year))
        for idx, label in enumerate(labels):
            threshold_rows.append(
                {
                    "test_year": test_year,
                    "label": label,
                    "parent": parent_of(label),
                    "is_late_emerging": label in LATE_EMERGING_LABELS,
                    "close_positive_threshold": float(pos_thresholds[idx]),
                    "close_negative_threshold": float(neg_thresholds[idx]),
                }
            )
        fold_rows.append(
            {
                "test_year": test_year,
                "fit_years": ",".join(map(str, sorted(fit["ev_year"].unique()))),
                "calibration_year": test_year - 1,
                "fit_samples": int(len(fit)),
                "calibration_samples": int(len(cal)),
                "test_samples": int(len(test)),
                "labels": int(len(labels)),
                "selected_positive_alpha": selected["positive_alpha"],
                "selected_negative_alpha": selected["negative_alpha"],
                "selected_late_positive_alpha": selected["late_positive_alpha"],
            }
        )

    write_csv(OUT_DIR / f"f_dfhc_metrics_{suffix}.csv", metric_rows)
    write_csv(OUT_DIR / f"f_dfhc_candidates_{suffix}.csv", candidate_rows)
    write_csv(OUT_DIR / f"f_dfhc_thresholds_{suffix}.csv", threshold_rows)
    write_csv(OUT_DIR / f"f_dfhc_folds_{suffix}.csv", fold_rows)

    df = pd.DataFrame(metric_rows)
    summary_rows = []
    if not df.empty:
        summary_rows = (
            df.groupby("method", as_index=False)
            .agg(
                test_years=("test_year", "nunique"),
                closed_precision=("closed_precision", "mean"),
                aggregate_closed_precision=("aggregate_closed_precision", "mean"),
                close_positive_precision=("close_positive_precision", "mean"),
                positive_coverage=("positive_coverage", "mean"),
                overall_coverage=("overall_coverage", "mean"),
                empirical_all_risk=("empirical_all_risk", "mean"),
                certified_all_bound=("certified_all_bound", "mean"),
                hierarchy_violations=("hierarchy_violations", "sum"),
                late_premature_closure_risk=("late_premature_closure_risk", "mean"),
            )
            .to_dict("records")
        )
    write_csv(OUT_DIR / f"f_dfhc_summary_{suffix}.csv", summary_rows)

    fdfhc = df[df["method"] == "F-DFHC"]
    lines = [f"# F-DFHC assessment ({suffix})", ""]
    if not fdfhc.empty:
        lines.append("| Test year | Closed P | Close+ P | Coverage | Bound | Hier. viol. | Late risk | Pass |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in fdfhc.to_dict("records"):
            passed = (
                row["closed_precision"] >= 0.955
                and row["close_positive_precision"] >= 0.900
                and row["overall_coverage"] >= 0.730
                and row["hierarchy_violations"] == 0
                and row["certified_all_bound"] <= 0.080
            )
            lines.append(
                f"| {int(row['test_year'])} | {row['closed_precision']:.3f} | {row['close_positive_precision']:.3f} | "
                f"{row['overall_coverage']:.3f} | {row['certified_all_bound']:.3f} | {int(row['hierarchy_violations'])} | "
                f"{row['late_premature_closure_risk']:.4f} | {'yes' if passed else 'no'} |"
            )
    (OUT_DIR / f"F_DFHC_ASSESSMENT_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "full": args.full,
        "seconds": round(time.time() - start, 3),
        "jobs": jobs,
        "alpha_all": args.alpha_all,
        "alpha_plus": args.alpha_plus,
        "alpha_late": args.alpha_late,
        "delta": args.delta,
        "test_years": test_years,
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / f"run_meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--alpha-all", type=float, default=0.060)
    parser.add_argument("--alpha-plus", type=float, default=0.10)
    parser.add_argument("--alpha-late", type=float, default=0.03)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
