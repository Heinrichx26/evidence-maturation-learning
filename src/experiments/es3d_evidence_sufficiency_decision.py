from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import (
    ROOT,
    build_document,
    global_metrics,
    load_samples,
    predict_stage,
    tune_thresholds,
)
from src.experiments.innovation_smoke_methods import (
    LATE_EMERGING_LABELS,
    fit_es3d,
    fit_lr_stage,
)


OUT_DIR = ROOT / "results" / "experiments" / "es3d_evidence_sufficiency_decision"


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


def decision_thresholds(row: dict, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    late_set = {labels.index(label) for label in LATE_EMERGING_LABELS if label in labels}
    pos = np.array(
        [
            float(row["late_close_positive_threshold"]) if j in late_set else float(row["other_close_positive_threshold"])
            for j in range(len(labels))
        ],
        dtype=float,
    )
    neg = np.array(
        [
            float(row["late_close_negative_threshold"]) if j in late_set else float(row["other_close_negative_threshold"])
            for j in range(len(labels))
        ],
        dtype=float,
    )
    return pos, neg


def per_label_actions(
    test_year: int,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    labels: list[str],
    es3d_row: dict,
) -> list[dict]:
    pos, neg = decision_thresholds(es3d_row, labels)
    close_pos = scores_test >= pos
    close_neg = scores_test <= neg
    acquire = ~(close_pos | close_neg)
    rows = []
    for idx, label in enumerate(labels):
        closed = close_pos[:, idx] | close_neg[:, idx]
        correct_closed = np.logical_or(
            np.logical_and(close_pos[:, idx], y_test[:, idx] == 1),
            np.logical_and(close_neg[:, idx], y_test[:, idx] == 0),
        )
        rows.append(
            {
                "test_year": test_year,
                "label": label,
                "support": int(y_test[:, idx].sum()),
                "is_late_emerging": label in LATE_EMERGING_LABELS,
                "close_positive": int(close_pos[:, idx].sum()),
                "close_negative": int(close_neg[:, idx].sum()),
                "acquire_more_evidence": int(acquire[:, idx].sum()),
                "coverage": float(closed.mean()),
                "closed_precision": float(correct_closed[closed].mean()) if closed.any() else 0.0,
                "positive_defer_rate": float(np.logical_and(acquire[:, idx], y_test[:, idx] == 1).sum() / max(1, y_test[:, idx].sum())),
            }
        )
    return rows


def run_fold(
    samples: pd.DataFrame,
    test_year: int,
    jobs: int,
    max_features: int,
    min_train_support: int,
    max_labels: int,
) -> tuple[dict | None, list[dict], list[dict]]:
    train_all = samples[samples["ev_year"] < test_year].copy()
    test = samples[samples["ev_year"] == test_year].copy()
    if train_all.empty or test.empty:
        return None, [], []
    val_year = int(train_all["ev_year"].max())
    fit = train_all[train_all["ev_year"] < val_year].copy()
    val = train_all[train_all["ev_year"] == val_year].copy()
    if fit.empty:
        fit = train_all
        val = train_all

    train_label_counts = Counter(label for labels in train_all["labels"] for label in labels)
    labels = sorted(label for label, count in train_label_counts.items() if count >= min_train_support)
    if max_labels:
        selected = {label for label, _ in train_label_counts.most_common(max_labels)}
        labels = sorted(label for label in labels if label in selected)
    if not labels:
        return None, [], []

    mlb = MultiLabelBinarizer(classes=labels)
    y_fit = mlb.fit_transform(fit["labels"].map(lambda xs: [x for x in xs if x in labels]))
    y_val = mlb.transform(val["labels"].map(lambda xs: [x for x in xs if x in labels]))
    y_train = mlb.transform(train_all["labels"].map(lambda xs: [x for x in xs if x in labels]))
    y_test = mlb.transform(test["labels"].map(lambda xs: [x for x in xs if x in labels]))
    keep = y_test.sum(axis=1) > 0
    test = test.loc[keep].copy()
    y_test = y_test[keep]
    if len(test) == 0:
        return None, [], []

    fit_docs = [build_document(row, "preliminary") for _, row in fit.iterrows()]
    val_docs = [build_document(row, "preliminary") for _, row in val.iterrows()]
    train_docs = [build_document(row, "preliminary") for _, row in train_all.iterrows()]
    test_docs = [build_document(row, "preliminary") for _, row in test.iterrows()]

    start = time.time()
    cal_vectorizer, cal_model = fit_lr_stage(fit_docs, y_fit, jobs, max_features)
    val_scores = predict_stage(cal_vectorizer, cal_model, val_docs)
    default_thresholds = tune_thresholds(y_val, val_scores)

    vectorizer, model = fit_lr_stage(train_docs, y_train, jobs, max_features)
    test_scores = predict_stage(vectorizer, model, test_docs)
    default_pred = test_scores >= default_thresholds
    default_metrics = global_metrics(y_test, default_pred, test_scores, "preliminary", test_year, len(labels))

    es3d_row = fit_es3d(y_val, y_test, val_scores, test_scores, labels)
    es3d_row.update(
        {
            "test_year": test_year,
            "fit_years": ",".join(map(str, sorted(fit["ev_year"].unique()))),
            "validation_year": val_year,
            "train_samples": int(len(train_all)),
            "test_samples": int(len(test)),
            "n_labels": int(len(labels)),
            "standard_micro_f1": float(default_metrics["micro_f1"]),
            "standard_macro_f1": float(default_metrics["macro_f1"]),
            "standard_samples_f1": float(default_metrics["samples_f1"]),
            "standard_jaccard": float(default_metrics["jaccard_samples"]),
            "runtime_seconds": round(time.time() - start, 3),
        }
    )
    action_rows = per_label_actions(test_year, y_test, test_scores, labels, es3d_row)
    threshold_rows = [
        {
            "test_year": test_year,
            "label": label,
            "is_late_emerging": label in LATE_EMERGING_LABELS,
            "close_positive_threshold": float(decision_thresholds(es3d_row, labels)[0][idx]),
            "close_negative_threshold": float(decision_thresholds(es3d_row, labels)[1][idx]),
            "standard_threshold": float(default_thresholds[idx]),
        }
        for idx, label in enumerate(labels)
    ]
    return es3d_row, action_rows, threshold_rows


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    return {
        "test_years": int(df["test_year"].nunique()),
        "mean_closed_precision": float(df["closed_precision"].mean()),
        "mean_coverage": float(df["coverage"].mean()),
        "mean_late_emerging_share_in_deferred_positives": float(df["late_emerging_share_in_deferred_positives"].mean()),
        "mean_late_positive_defer_rate": float(df["late_positive_defer_rate"].mean()),
        "mean_cost_reduction": float(df["cost_reduction"].mean()),
        "mean_standard_micro_f1": float(df["standard_micro_f1"].mean()),
        "mean_standard_macro_f1": float(df["standard_macro_f1"].mean()),
        "total_runtime_seconds": float(df["runtime_seconds"].sum()),
    }


def write_assessment(rows: list[dict], summary: dict, suffix: str) -> None:
    lines = [f"# ES3D evidence-sufficiency decision assessment ({suffix})", ""]
    if not rows:
        lines.append("No valid folds were produced.")
        (OUT_DIR / f"ES3D_ASSESSMENT_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append("| Test year | Closed precision | Coverage | Late deferred-positive share | Cost reduction | Standard Micro-F1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['test_year']} | {row['closed_precision']:.3f} | {row['coverage']:.3f} | "
            f"{row['late_emerging_share_in_deferred_positives']:.3f} | {row['cost_reduction']:.3f} | "
            f"{row['standard_micro_f1']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Mean closed precision is {summary['mean_closed_precision']:.3f}, coverage is {summary['mean_coverage']:.3f}, "
        f"late deferred-positive share is {summary['mean_late_emerging_share_in_deferred_positives']:.3f}, "
        f"and cost reduction is {summary['mean_cost_reduction']:.3f}."
    )
    if (
        summary["mean_closed_precision"] >= 0.85
        and summary["mean_coverage"] >= 0.45
        and summary["mean_late_emerging_share_in_deferred_positives"] >= 0.70
        and summary["mean_cost_reduction"] >= 0.15
    ):
        lines.append("Decision: ES3D remains usable under forward-year validation.")
    else:
        lines.append("Decision: ES3D is not stable enough for the main manuscript.")
    (OUT_DIR / f"ES3D_ASSESSMENT_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    samples = load_samples(args.label_level, args.min_global_support)
    test_years = [2023] if args.smoke else [2023, 2024, 2025]
    rows: list[dict] = []
    action_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for test_year in test_years:
        row, actions, thresholds = run_fold(
            samples,
            test_year,
            jobs,
            args.max_features,
            args.min_train_support,
            args.max_labels,
        )
        if row:
            rows.append(row)
            action_rows.extend(actions)
            threshold_rows.extend(thresholds)
    suffix = "smoke" if args.smoke else "full"
    write_csv(OUT_DIR / f"es3d_metrics_by_year_{suffix}.csv", rows)
    write_csv(OUT_DIR / f"es3d_label_actions_by_year_{suffix}.csv", action_rows)
    write_csv(OUT_DIR / f"es3d_thresholds_by_year_{suffix}.csv", threshold_rows)
    summary = summarize(rows)
    write_csv(OUT_DIR / f"es3d_summary_{suffix}.csv", [summary] if summary else [])
    write_assessment(rows, summary, suffix)
    meta = {
        "smoke": args.smoke,
        "seconds": round(time.time() - start, 3),
        "jobs": jobs,
        "max_features": args.max_features,
        "min_global_support": args.min_global_support,
        "min_train_support": args.min_train_support,
        "test_years": test_years,
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / f"run_meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
