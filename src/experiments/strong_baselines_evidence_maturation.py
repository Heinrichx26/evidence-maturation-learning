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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import (
    ROOT,
    build_document,
    entropy,
    global_metrics,
    load_samples,
    predict_stage,
    tune_thresholds,
)


OUT_DIR = ROOT / "results" / "experiments" / "strong_baselines_evidence_maturation"
TABLE_DIR = ROOT / "results" / "tables"
REFERENCE_DIR = ROOT / "results" / "references"

STAGES = ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]
BASELINE_GRID = np.array(
    [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
)

METHOD_SOURCES = [
    {
        "method": "MatchXML-adapted",
        "title": "MatchXML: An Efficient Text-Label Matching Framework for Extreme Multi-Label Text Classification",
        "venue": "IEEE Transactions on Knowledge and Data Engineering",
        "year": 2024,
        "doi": "10.1109/TKDE.2024.3374750",
        "implemented_signal": "text-label matching with label-name and positive-evidence profiles",
    },
    {
        "method": "VCLDL-adapted",
        "title": "Variational Continuous Label Distribution Learning for Multi-Label Text Classification",
        "venue": "IEEE Transactions on Knowledge and Data Engineering",
        "year": 2024,
        "doi": "10.1109/TKDE.2023.3323401",
        "implemented_signal": "continuous label distribution propagation over training co-occurrence",
    },
    {
        "method": "HALB-adapted",
        "title": "Hierarchy-Aware and Label Balanced Model for Hierarchical Text Classification",
        "venue": "Knowledge-Based Systems",
        "year": 2024,
        "doi": "10.1016/j.knosys.2024.112153",
        "implemented_signal": "label-balanced child classifier combined with parent-level hierarchy scores",
    },
    {
        "method": "LSPCL-adapted",
        "title": "LSPCL: Label-specific supervised prototype contrastive learning for multi-label text classification",
        "venue": "Knowledge-Based Systems",
        "year": 2025,
        "doi": "10.1016/j.knosys.2024.112887",
        "implemented_signal": "label-specific positive prototypes with negative-centroid separation",
    },
    {
        "method": "DyLas-adapted",
        "title": "DyLas: A dynamic label alignment strategy for large-scale multi-label text classification",
        "venue": "Information Fusion",
        "year": 2025,
        "doi": "10.1016/j.inffus.2025.103081",
        "implemented_signal": "validation-tuned dynamic alignment between classifier and label-profile scores",
    },
    {
        "method": "SIHTC-adapted",
        "title": "Hierarchical Text Classification Optimization via Structural Entropy and Singular Smoothing",
        "venue": "IEEE Transactions on Knowledge and Data Engineering",
        "year": 2025,
        "doi": "10.1109/TKDE.2025.3579810",
        "implemented_signal": "low-rank structural smoothing over the label co-occurrence graph",
    },
]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parent_label(label: str) -> str:
    return label.split("-", 1)[0].strip()


def label_text(label: str) -> str:
    return label.replace("-", " ").replace("/", " ")


def tune_thresholds_for_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(y_true.shape[1], dtype=float)
    for j in range(y_true.shape[1]):
        values = np.asarray(scores[:, j], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            thresholds[j] = 0.5
            continue
        candidates = list(BASELINE_GRID)
        candidates.extend(np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
        candidates.extend([float(finite.min()), float(finite.max())])
        best_t = float(np.median(finite))
        best_score = -1.0
        for t in sorted(set(candidates)):
            pred = values >= t
            score = f1_score(y_true[:, j], pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds[j] = best_t
    return thresholds


def predict_with_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.asarray(scores >= thresholds, dtype=bool)


def fit_lr_from_docs(
    docs: list[str],
    y: np.ndarray,
    jobs: int,
    max_features: int,
    c_value: float,
) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(docs)
    estimator = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        C=c_value,
        random_state=42,
    )
    model = OneVsRestClassifier(estimator, n_jobs=jobs)
    model.fit(x, y)
    return vectorizer, model


def label_transition(y: np.ndarray, rank: int | None = None) -> np.ndarray:
    n_labels = y.shape[1]
    cooc = y.T @ y
    cooc = cooc.astype(float) + np.eye(n_labels) * 1e-3
    if rank and 0 < rank < n_labels:
        u, s, vt = np.linalg.svd(cooc, full_matrices=False)
        cooc = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        cooc = np.maximum(cooc, 0.0)
        cooc = cooc + np.eye(n_labels) * 1e-3
    row_sum = cooc.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return cooc / row_sum


def choose_smoothed_scores(
    y_val: np.ndarray,
    base_val: np.ndarray,
    transition: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    best_alpha = 1.0
    best_score = -1.0
    best_thresholds = tune_thresholds(y_val, base_val)
    best_scores = base_val
    propagated = base_val @ transition
    for alpha in alpha_grid:
        scores = alpha * base_val + (1.0 - alpha) * propagated
        thresholds = tune_thresholds(y_val, scores)
        pred = predict_with_thresholds(scores, thresholds)
        score = f1_score(y_val, pred, average="micro", zero_division=0)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_thresholds = thresholds
            best_scores = scores
    return best_alpha, best_thresholds, best_scores


def fit_profile_model(
    docs: list[str],
    y: np.ndarray,
    labels: list[str],
    max_features: int,
    mode: str,
) -> tuple[TfidfVectorizer, np.ndarray]:
    training_texts = docs + [label_text(label) for label in labels]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(training_texts)
    x = vectorizer.transform(docs)
    label_x = vectorizer.transform([label_text(label) for label in labels]).toarray()
    profiles = np.zeros((len(labels), x.shape[1]), dtype=np.float64)
    for j in range(len(labels)):
        pos_mask = y[:, j] == 1
        neg_mask = ~pos_mask
        if pos_mask.any():
            pos = np.asarray(x[pos_mask].mean(axis=0)).ravel()
        else:
            pos = np.zeros(x.shape[1], dtype=np.float64)
        if neg_mask.any():
            neg = np.asarray(x[neg_mask].mean(axis=0)).ravel()
        else:
            neg = np.zeros(x.shape[1], dtype=np.float64)

        if mode == "matchxml":
            profile = 0.70 * pos + 0.30 * label_x[j]
        elif mode == "lspcl":
            profile = pos - 0.15 * neg + 0.10 * label_x[j]
        else:
            profile = pos
        profile = np.maximum(profile, 0.0)
        if not np.any(profile):
            profile = label_x[j]
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        profiles[j] = profile
    return vectorizer, profiles


def profile_scores(vectorizer: TfidfVectorizer, profiles: np.ndarray, docs: list[str]) -> np.ndarray:
    x = normalize(vectorizer.transform(docs), norm="l2", copy=False)
    scores = x @ profiles.T
    if sparse.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=float)


def minmax_scale_by_train(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    lo = np.nanmin(train_scores, axis=0)
    hi = np.nanmax(train_scores, axis=0)
    denom = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    scaled = (scores - lo) / denom
    return np.clip(scaled, 0.0, 1.0)


def parent_matrix(label_sets: pd.Series, parents: list[str]) -> np.ndarray:
    parent_to_idx = {name: i for i, name in enumerate(parents)}
    y = np.zeros((len(label_sets), len(parents)), dtype=int)
    for i, labels in enumerate(label_sets):
        for label in labels:
            p = parent_label(label)
            if p in parent_to_idx:
                y[i, parent_to_idx[p]] = 1
    return y


def tune_hierarchy_blend(
    y_val: np.ndarray,
    child_val: np.ndarray,
    parent_val: np.ndarray,
    child_parent_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    best_beta = 1.0
    best_score = -1.0
    best_thresholds = tune_thresholds(y_val, child_val)
    for beta in [0.45, 0.60, 0.75, 0.90, 1.00]:
        parent_child = parent_val[:, child_parent_idx]
        scores = beta * child_val + (1.0 - beta) * parent_child
        thresholds = tune_thresholds(y_val, scores)
        pred = predict_with_thresholds(scores, thresholds)
        score = f1_score(y_val, pred, average="micro", zero_division=0)
        if score > best_score:
            best_score = score
            best_beta = float(beta)
            best_thresholds = thresholds
    return best_beta, best_thresholds


def tune_dynamic_alignment(
    y_val: np.ndarray,
    base_val: np.ndarray,
    match_val: np.ndarray,
) -> tuple[float, np.ndarray]:
    best_gamma = 0.0
    best_score = -1.0
    best_thresholds = tune_thresholds(y_val, base_val)
    for gamma in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]:
        scores = (1.0 - gamma) * base_val + gamma * match_val
        thresholds = tune_thresholds(y_val, scores)
        pred = predict_with_thresholds(scores, thresholds)
        score = f1_score(y_val, pred, average="micro", zero_division=0)
        if score > best_score:
            best_score = score
            best_gamma = float(gamma)
            best_thresholds = thresholds
    return best_gamma, best_thresholds


def evaluate_scores(
    rows: list[dict],
    method: str,
    stage: str,
    test_year: int,
    y_test: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    n_labels: int,
    runtime_seconds: float,
    extra: dict | None = None,
) -> None:
    pred = predict_with_thresholds(scores, thresholds)
    row = global_metrics(y_test, pred, scores, stage, test_year, n_labels)
    row["method"] = method
    row["runtime_seconds"] = round(runtime_seconds, 3)
    if extra:
        row.update(extra)
    rows.append(row)


def run_fold_stage(
    rows: list[dict],
    fold_rows: list[dict],
    fit: pd.DataFrame,
    val: pd.DataFrame,
    train_all: pd.DataFrame,
    test: pd.DataFrame,
    y_fit: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    stage: str,
    test_year: int,
    jobs: int,
    max_features: int,
) -> None:
    fit_docs = [build_document(row, stage) for _, row in fit.iterrows()]
    val_docs = [build_document(row, stage) for _, row in val.iterrows()]
    train_docs = [build_document(row, stage) for _, row in train_all.iterrows()]
    test_docs = [build_document(row, stage) for _, row in test.iterrows()]

    stage_start = time.time()
    cal_vectorizer, cal_model = fit_lr_from_docs(fit_docs, y_fit, jobs, max_features, c_value=1.0)
    val_base = predict_stage(cal_vectorizer, cal_model, val_docs)
    base_thresholds = tune_thresholds(y_val, val_base)

    vectorizer, model = fit_lr_from_docs(train_docs, y_train, jobs, max_features, c_value=1.0)
    test_base = predict_stage(vectorizer, model, test_docs)
    base_runtime = time.time() - stage_start
    evaluate_scores(
        rows,
        "BR-TFIDF",
        stage,
        test_year,
        y_test,
        test_base,
        base_thresholds,
        len(labels),
        base_runtime,
        {"source_doi": ""},
    )

    vcldl_start = time.time()
    alpha, vcldl_thresholds, _ = choose_smoothed_scores(
        y_val,
        val_base,
        label_transition(y_fit),
        [0.35, 0.50, 0.65, 0.80, 0.90, 1.00],
    )
    test_vcldl = alpha * test_base + (1.0 - alpha) * (test_base @ label_transition(y_train))
    evaluate_scores(
        rows,
        "VCLDL-adapted",
        stage,
        test_year,
        y_test,
        test_vcldl,
        vcldl_thresholds,
        len(labels),
        base_runtime + (time.time() - vcldl_start),
        {"source_doi": "10.1109/TKDE.2023.3323401", "blend_alpha": alpha},
    )

    sihtc_start = time.time()
    rank = max(2, min(6, len(labels) // 2))
    s_alpha, sihtc_thresholds, _ = choose_smoothed_scores(
        y_val,
        val_base,
        label_transition(y_fit, rank=rank),
        [0.45, 0.60, 0.75, 0.85, 0.95, 1.00],
    )
    test_sihtc = s_alpha * test_base + (1.0 - s_alpha) * (test_base @ label_transition(y_train, rank=rank))
    evaluate_scores(
        rows,
        "SIHTC-adapted",
        stage,
        test_year,
        y_test,
        test_sihtc,
        sihtc_thresholds,
        len(labels),
        base_runtime + (time.time() - sihtc_start),
        {"source_doi": "10.1109/TKDE.2025.3579810", "blend_alpha": s_alpha, "svd_rank": rank},
    )

    profile_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    profile_runtime: dict[str, float] = {}
    for method, mode, doi in [
        ("MatchXML-adapted", "matchxml", "10.1109/TKDE.2024.3374750"),
        ("LSPCL-adapted", "lspcl", "10.1016/j.knosys.2024.112887"),
    ]:
        method_start = time.time()
        p_vectorizer, profiles = fit_profile_model(fit_docs, y_fit, labels, max_features, mode=mode)
        fit_profile = profile_scores(p_vectorizer, profiles, fit_docs)
        val_profile = minmax_scale_by_train(fit_profile, profile_scores(p_vectorizer, profiles, val_docs))
        thresholds = tune_thresholds_for_scores(y_val, val_profile)

        final_vectorizer, final_profiles = fit_profile_model(train_docs, y_train, labels, max_features, mode=mode)
        train_profile = profile_scores(final_vectorizer, final_profiles, train_docs)
        test_profile = minmax_scale_by_train(train_profile, profile_scores(final_vectorizer, final_profiles, test_docs))
        method_runtime = time.time() - method_start
        profile_cache[mode] = (val_profile, test_profile, thresholds)
        profile_runtime[mode] = method_runtime
        evaluate_scores(
            rows,
            method,
            stage,
            test_year,
            y_test,
            test_profile,
            thresholds,
            len(labels),
            method_runtime,
            {"source_doi": doi},
        )

    halb_start = time.time()
    parents = sorted({parent_label(label) for label in labels})
    parent_idx = {name: i for i, name in enumerate(parents)}
    child_parent_idx = np.array([parent_idx[parent_label(label)] for label in labels], dtype=int)
    y_parent_fit = parent_matrix(fit["labels"], parents)
    y_parent_train = parent_matrix(train_all["labels"], parents)
    p_cal_vectorizer, p_cal_model = fit_lr_from_docs(fit_docs, y_parent_fit, jobs, max_features, c_value=0.8)
    val_parent = predict_stage(p_cal_vectorizer, p_cal_model, val_docs)
    beta, halb_thresholds = tune_hierarchy_blend(y_val, val_base, val_parent, child_parent_idx)
    p_vectorizer, p_model = fit_lr_from_docs(train_docs, y_parent_train, jobs, max_features, c_value=0.8)
    test_parent = predict_stage(p_vectorizer, p_model, test_docs)
    test_halb = beta * test_base + (1.0 - beta) * test_parent[:, child_parent_idx]
    evaluate_scores(
        rows,
        "HALB-adapted",
        stage,
        test_year,
        y_test,
        test_halb,
        halb_thresholds,
        len(labels),
        base_runtime + (time.time() - halb_start),
        {"source_doi": "10.1016/j.knosys.2024.112153", "hierarchy_beta": beta},
    )

    dylas_start = time.time()
    match_val, match_test, _ = profile_cache["matchxml"]
    gamma, dylas_thresholds = tune_dynamic_alignment(y_val, val_base, match_val)
    test_dylas = (1.0 - gamma) * test_base + gamma * match_test
    evaluate_scores(
        rows,
        "DyLas-adapted",
        stage,
        test_year,
        y_test,
        test_dylas,
        dylas_thresholds,
        len(labels),
        base_runtime + profile_runtime.get("matchxml", 0.0) + (time.time() - dylas_start),
        {"source_doi": "10.1016/j.inffus.2025.103081", "alignment_gamma": gamma},
    )

    fold_rows.append(
        {
            "test_year": test_year,
            "stage": stage,
            "fit_samples": int(len(fit)),
            "validation_samples": int(len(val)),
            "train_samples": int(len(train_all)),
            "test_samples": int(len(test)),
            "labels": int(len(labels)),
            "stage_seconds": round(time.time() - stage_start, 3),
        }
    )


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["method", "stage"], as_index=False)
        .agg(
            test_years=("test_year", "nunique"),
            n_samples=("n_samples", "sum"),
            n_labels=("n_labels", "mean"),
            micro_f1=("micro_f1", "mean"),
            macro_f1=("macro_f1", "mean"),
            samples_f1=("samples_f1", "mean"),
            jaccard_samples=("jaccard_samples", "mean"),
            mean_binary_entropy=("mean_binary_entropy", "mean"),
            runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values(["stage", "micro_f1"], ascending=[True, False])
    )


def write_result_summary(avg: pd.DataFrame, suffix: str) -> None:
    if avg.empty:
        return
    lines = ["# Strong baseline comparison", ""]
    for stage in STAGES:
        part = avg[avg["stage"] == stage].sort_values("micro_f1", ascending=False)
        if part.empty:
            continue
        lines.append(f"## {stage}")
        lines.append("")
        lines.append("| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in part.to_dict("records"):
            lines.append(
                f"| {row['method']} | {row['micro_f1']:.3f} | {row['macro_f1']:.3f} | "
                f"{row['samples_f1']:.3f} | {row['jaccard_samples']:.3f} | {row['runtime_seconds']:.1f} |"
            )
        lines.append("")
    (OUT_DIR / f"STRONG_BASELINE_SUMMARY_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REFERENCE_DIR / "strong_baseline_method_sources.csv", METHOD_SOURCES)

    samples = load_samples(args.label_level, args.min_global_support)
    if args.smoke:
        samples = samples[samples["ev_year"].isin([2020, 2021, 2022])].copy()
        test_years = [2022]
    else:
        test_years = [2023, 2024, 2025]

    workers = max(1, min(args.jobs, os.cpu_count() or 1))
    global_rows: list[dict] = []
    fold_rows: list[dict] = []

    for test_year in test_years:
        train_all = samples[samples["ev_year"] < test_year].copy()
        test = samples[samples["ev_year"] == test_year].copy()
        if train_all.empty or test.empty:
            continue

        val_year = int(train_all["ev_year"].max())
        fit = train_all[train_all["ev_year"] < val_year].copy()
        val = train_all[train_all["ev_year"] == val_year].copy()
        if fit.empty:
            fit = train_all
            val = train_all

        train_label_counts = Counter(label for labels in train_all["labels"] for label in labels)
        labels = sorted(label for label, count in train_label_counts.items() if count >= args.min_train_support)
        if args.max_labels:
            selected = {label for label, _ in train_label_counts.most_common(args.max_labels)}
            labels = sorted(label for label in labels if label in selected)
        if not labels:
            continue

        mlb = MultiLabelBinarizer(classes=labels)
        y_fit = mlb.fit_transform(fit["labels"].map(lambda xs: [x for x in xs if x in labels]))
        y_val = mlb.transform(val["labels"].map(lambda xs: [x for x in xs if x in labels]))
        y_train = mlb.transform(train_all["labels"].map(lambda xs: [x for x in xs if x in labels]))
        y_test = mlb.transform(test["labels"].map(lambda xs: [x for x in xs if x in labels]))

        keep = y_test.sum(axis=1) > 0
        test = test.loc[keep].copy()
        y_test = y_test[keep]
        if len(test) == 0:
            continue

        for stage in STAGES:
            run_fold_stage(
                global_rows,
                fold_rows,
                fit,
                val,
                train_all,
                test,
                y_fit,
                y_val,
                y_train,
                y_test,
                labels,
                stage,
                test_year,
                workers,
                args.max_features,
            )

    suffix = "smoke" if args.smoke else "full"
    metrics_path = OUT_DIR / f"baseline_metrics_by_year_{suffix}.csv"
    write_csv(metrics_path, global_rows)
    write_csv(OUT_DIR / f"fold_stage_notes_{suffix}.csv", fold_rows)
    avg = aggregate(global_rows)
    avg_path = OUT_DIR / f"baseline_metrics_average_{suffix}.csv"
    avg.to_csv(avg_path, index=False, encoding="utf-8-sig")
    write_result_summary(avg, suffix)

    if not args.smoke and not avg.empty:
        avg.to_csv(TABLE_DIR / "tab_strong_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    run_meta = {
        "smoke": args.smoke,
        "label_level": args.label_level,
        "min_global_support": args.min_global_support,
        "min_train_support": args.min_train_support,
        "max_features": args.max_features,
        "jobs": workers,
        "seconds": round(time.time() - start, 3),
        "samples": int(len(samples)),
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / f"run_meta_{suffix}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
