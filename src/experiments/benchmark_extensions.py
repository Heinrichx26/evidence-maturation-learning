from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from sklearn.pipeline import make_pipeline

import sys

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import (  # noqa: E402
    ROOT,
    build_document,
    load_samples,
)


OUT_DIR = ROOT / "results" / "experiments" / "benchmark_extensions"
STAGES = ["preliminary", "mature_factual"]


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


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, method: str, split: str, stage: str, test_year: int) -> dict:
    return {
        "split": split,
        "test_year": test_year,
        "stage": stage,
        "method": method,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "mean_binary_entropy": entropy(proba),
    }


def entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def tune_thresholds_local(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.30)
    base_grid = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        quantiles = np.quantile(proba[:, j], [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
        grid = np.unique(np.clip(np.concatenate([base_grid, quantiles]), 0.0, 1.0))
        best_score = -1.0
        best_threshold = 0.30
        for threshold in grid:
            pred = proba[:, j] >= threshold
            score = f1_score(y_true[:, j], pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[j] = best_threshold
    return thresholds


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or y_true.shape[0] == 0:
        return math.nan, math.nan
    values = []
    n = y_true.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(f1_score(y_true[idx], y_pred[idx], average="micro", zero_division=0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    if n_bootstrap <= 0 or y_true.shape[0] == 0:
        return math.nan, math.nan, math.nan
    diffs = []
    n = y_true.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = f1_score(y_true[idx], y_a[idx], average="micro", zero_division=0)
        b = f1_score(y_true[idx], y_b[idx], average="micro", zero_division=0)
        diffs.append(a - b)
    diffs_arr = np.asarray(diffs)
    p_two_sided = float(2 * min((diffs_arr <= 0).mean(), (diffs_arr >= 0).mean()))
    return float(np.quantile(diffs_arr, 0.025)), float(np.quantile(diffs_arr, 0.975)), min(1.0, p_two_sided)


def vectorize(fit_docs: list[str], val_docs: list[str], test_docs: list[str], max_features: int):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_fit = vectorizer.fit_transform(fit_docs)
    x_val = vectorizer.transform(val_docs)
    x_test = vectorizer.transform(test_docs)
    return vectorizer, x_fit, x_val, x_test


def fit_br(x_fit, y_fit, jobs: int, c_value: float = 1.0) -> OneVsRestClassifier:
    estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", C=c_value)
    model = OneVsRestClassifier(estimator, n_jobs=jobs)
    model.fit(x_fit, y_fit)
    return model


def predict_br(model: OneVsRestClassifier, x) -> np.ndarray:
    cols = []
    for estimator in model.estimators_:
        if hasattr(estimator, "predict_proba"):
            cols.append(estimator.predict_proba(x)[:, 1])
        else:
            cols.append(estimator.decision_function(x))
    return np.vstack(cols).T


def fit_chain(x_fit, y_fit, order: np.ndarray) -> ClassifierChain:
    base = LogisticRegression(max_iter=800, solver="liblinear", class_weight="balanced")
    model = ClassifierChain(base_estimator=base, order=order, random_state=13)
    model.fit(x_fit, y_fit)
    return model


def predict_chain(model: ClassifierChain, x) -> np.ndarray:
    proba = model.predict_proba(x)
    return np.asarray(proba, dtype=float)


def dense_projection(x_fit, x_val, x_test, dims: int, seed: int):
    max_dims = max(2, min(dims, x_fit.shape[1] - 1, x_fit.shape[0] - 1))
    reducer = make_pipeline(TruncatedSVD(n_components=max_dims, random_state=seed), Normalizer(copy=False))
    return reducer.fit_transform(x_fit), reducer.transform(x_val), reducer.transform(x_test)


def fit_mlknn(x_fit_dense, y_fit, neighbors: int) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", metric="cosine")
    model.fit(x_fit_dense, y_fit)
    return model


def predict_multioutput_proba(model, x_dense, n_labels: int) -> np.ndarray:
    raw = model.predict_proba(x_dense)
    if isinstance(raw, list):
        cols = []
        for item in raw:
            if item.shape[1] == 1:
                cols.append(np.zeros(item.shape[0]))
            else:
                cols.append(item[:, 1])
        return np.vstack(cols).T
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1].T
    return np.asarray(model.predict(x_dense), dtype=float).reshape(-1, n_labels)


def fit_rakel(x_fit, y_fit, n_subsets: int, subset_size: int, seed: int):
    rng = np.random.default_rng(seed)
    n_labels = y_fit.shape[1]
    subsets = []
    models = []
    class_maps = []
    for _ in range(n_subsets):
        subset = np.sort(rng.choice(n_labels, size=min(subset_size, n_labels), replace=False))
        y_sub = y_fit[:, subset]
        powers = np.packbits(y_sub.astype(np.uint8), axis=1)
        classes = np.array([int.from_bytes(row.tobytes(), "big") for row in powers])
        if np.unique(classes).size < 2:
            continue
        model = LogisticRegression(max_iter=700, solver="lbfgs", class_weight="balanced")
        model.fit(x_fit, classes)
        subsets.append(subset)
        models.append(model)
        class_maps.append(model.classes_)
    return subsets, models, class_maps


def predict_rakel(fitted, x, n_labels: int) -> np.ndarray:
    subsets, models, class_maps = fitted
    scores = np.zeros((x.shape[0], n_labels), dtype=float)
    counts = np.zeros(n_labels, dtype=float)
    for subset, model, classes in zip(subsets, models, class_maps):
        proba = model.predict_proba(x)
        for class_idx, cls in enumerate(classes):
            bits = np.unpackbits(np.array([cls], dtype=">u8").view(np.uint8))[-len(subset) :]
            scores[:, subset] += proba[:, [class_idx]] * bits
        counts[subset] += 1
    counts[counts == 0] = 1
    return scores / counts


def parent_of(label: str) -> str:
    return label.split("-", 1)[0].strip() if "-" in label else label


def hierarchy_gated_scores(scores: np.ndarray, labels: list[str], parent_thresholds: dict[str, float]) -> np.ndarray:
    parent_scores: dict[str, np.ndarray] = {}
    for parent in sorted({parent_of(label) for label in labels}):
        idx = [i for i, label in enumerate(labels) if parent_of(label) == parent]
        parent_scores[parent] = scores[:, idx].max(axis=1)
    gated = scores.copy()
    for idx, label in enumerate(labels):
        parent = parent_of(label)
        mask = parent_scores[parent] < parent_thresholds.get(parent, 0.30)
        gated[mask, idx] *= 0.25
    return gated


def tune_parent_thresholds(scores_val: np.ndarray, y_val: np.ndarray, labels: list[str]) -> dict[str, float]:
    parent_thresholds = {}
    for parent in sorted({parent_of(label) for label in labels}):
        idx = [i for i, label in enumerate(labels) if parent_of(label) == parent]
        y_parent = y_val[:, idx].max(axis=1)
        s_parent = scores_val[:, idx].max(axis=1)
        best = (0.30, -1.0)
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            score = f1_score(y_parent, s_parent >= threshold, zero_division=0)
            if score > best[1]:
                best = (threshold, score)
        parent_thresholds[parent] = best[0]
    return parent_thresholds


def fit_label_enhanced_forest(x_fit_dense, y_fit, seed: int):
    model = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_fit_dense, y_fit)
    return model


def predict_forest(model, x_dense, n_labels: int) -> np.ndarray:
    return predict_multioutput_proba(model, x_dense, n_labels)


def select_labels(train_all: pd.DataFrame, min_train_support: int, max_labels: int) -> list[str]:
    counts = Counter(label for labels in train_all["labels"] for label in labels)
    labels = sorted(label for label, count in counts.items() if count >= min_train_support)
    if max_labels:
        head = {label for label, _ in counts.most_common(max_labels)}
        labels = sorted(label for label in labels if label in head)
    return labels


def binarize(samples: pd.DataFrame, labels: list[str]) -> np.ndarray:
    mlb = MultiLabelBinarizer(classes=labels)
    return mlb.fit_transform(samples["labels"].map(lambda xs: [x for x in xs if x in labels]))


def make_split(samples: pd.DataFrame, split: str, test_year: int):
    if split == "fixed_historical":
        fit = samples[(samples["ev_year"] >= 2008) & (samples["ev_year"] <= 2018)].copy()
        val = samples[samples["ev_year"] == 2019].copy()
        test = samples[samples["ev_year"] == test_year].copy()
        return fit, val, test
    train_all = samples[samples["ev_year"] < test_year].copy()
    val_year = int(train_all["ev_year"].max())
    fit = train_all[train_all["ev_year"] < val_year].copy()
    val = train_all[train_all["ev_year"] == val_year].copy()
    test = samples[samples["ev_year"] == test_year].copy()
    return fit, val, test


def run_fold(args, samples: pd.DataFrame, split: str, test_year: int, stage: str, rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict]]:
    fit, val, test = make_split(samples, split, test_year)
    labels = select_labels(pd.concat([fit, val], ignore_index=True), args.min_train_support, args.max_labels)
    if fit.empty or val.empty or test.empty or not labels:
        return [], [], []

    y_fit = binarize(fit, labels)
    y_val = binarize(val, labels)
    y_test = binarize(test, labels)
    keep = y_test.sum(axis=1) > 0
    test = test.loc[keep].copy()
    y_test = y_test[keep]
    if y_test.shape[0] == 0:
        return [], [], []

    fit_docs = [build_document(row, stage) for _, row in fit.iterrows()]
    val_docs = [build_document(row, stage) for _, row in val.iterrows()]
    test_docs = [build_document(row, stage) for _, row in test.iterrows()]
    _, x_fit, x_val, x_test = vectorize(fit_docs, val_docs, test_docs, args.max_features)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))

    metrics: list[dict] = []
    ci_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    requested_methods = {method.strip() for method in args.methods.split(",") if method.strip()}

    def wants(method: str) -> bool:
        return not requested_methods or method in requested_methods

    br = fit_br(x_fit, y_fit, jobs=jobs)
    br_val = predict_br(br, x_val)
    br_test = predict_br(br, x_test)
    br_thresholds = tune_thresholds_local(y_val, br_val)
    br_pred = br_test >= br_thresholds
    predictions["BR-TFIDF"] = br_pred
    if wants("BR-TFIDF"):
        metrics.append(metric_row(y_test, br_pred, br_test, "BR-TFIDF", split, stage, test_year))

    if wants("Classifier Chains"):
        order = np.argsort(-y_fit.sum(axis=0))
        chain = fit_chain(x_fit, y_fit, order=order)
        chain_val = predict_chain(chain, x_val)
        chain_test = predict_chain(chain, x_test)
        chain_thresholds = tune_thresholds_local(y_val, chain_val)
        chain_pred = chain_test >= chain_thresholds
        predictions["Classifier Chains"] = chain_pred
        metrics.append(metric_row(y_test, chain_pred, chain_test, "Classifier Chains", split, stage, test_year))

    need_dense = wants("ML-KNN") or wants("Label-enhanced forest")
    if need_dense:
        x_fit_dense, x_val_dense, x_test_dense = dense_projection(x_fit, x_val, x_test, args.svd_dims, args.seed)
    else:
        x_fit_dense = x_val_dense = x_test_dense = None

    if wants("ML-KNN"):
        mlknn = fit_mlknn(x_fit_dense, y_fit, args.neighbors)
        mlknn_val = predict_multioutput_proba(mlknn, x_val_dense, len(labels))
        mlknn_test = predict_multioutput_proba(mlknn, x_test_dense, len(labels))
        mlknn_thresholds = tune_thresholds_local(y_val, mlknn_val)
        mlknn_pred = mlknn_test >= mlknn_thresholds
        predictions["ML-KNN"] = mlknn_pred
        metrics.append(metric_row(y_test, mlknn_pred, mlknn_test, "ML-KNN", split, stage, test_year))

    if wants("RAkEL"):
        rakel = fit_rakel(x_fit, y_fit, args.rakel_subsets, args.rakel_subset_size, args.seed)
        rakel_val = predict_rakel(rakel, x_val, len(labels))
        rakel_test = predict_rakel(rakel, x_test, len(labels))
        rakel_thresholds = tune_thresholds_local(y_val, rakel_val)
        rakel_pred = rakel_test >= rakel_thresholds
        predictions["RAkEL"] = rakel_pred
        metrics.append(metric_row(y_test, rakel_pred, rakel_test, "RAkEL", split, stage, test_year))

    if wants("Hierarchy-gated BR"):
        parent_thresholds = tune_parent_thresholds(br_val, y_val, labels)
        h_val = hierarchy_gated_scores(br_val, labels, parent_thresholds)
        h_test = hierarchy_gated_scores(br_test, labels, parent_thresholds)
        h_thresholds = tune_thresholds_local(y_val, h_val)
        h_pred = h_test >= h_thresholds
        predictions["Hierarchy-gated BR"] = h_pred
        metrics.append(metric_row(y_test, h_pred, h_test, "Hierarchy-gated BR", split, stage, test_year))

    if wants("Label-enhanced forest"):
        forest = fit_label_enhanced_forest(x_fit_dense, y_fit, args.seed)
        forest_val = predict_forest(forest, x_val_dense, len(labels))
        forest_test = predict_forest(forest, x_test_dense, len(labels))
        forest_thresholds = tune_thresholds_local(y_val, forest_val)
        forest_pred = forest_test >= forest_thresholds
        predictions["Label-enhanced forest"] = forest_pred
        metrics.append(metric_row(y_test, forest_pred, forest_test, "Label-enhanced forest", split, stage, test_year))

    for row in metrics:
        method = row["method"]
        lo, hi = bootstrap_ci(y_test, predictions[method], rng, args.bootstrap)
        diff_lo, diff_hi, p_value = paired_bootstrap_diff(y_test, predictions[method], predictions["BR-TFIDF"], rng, args.bootstrap)
        ci_rows.append(
            {
                **{key: row[key] for key in ["split", "test_year", "stage", "method"]},
                "micro_f1": row["micro_f1"],
                "micro_f1_ci_low": lo,
                "micro_f1_ci_high": hi,
                "diff_vs_br_ci_low": diff_lo,
                "diff_vs_br_ci_high": diff_hi,
                "paired_bootstrap_p": p_value,
            }
        )

    fold_rows = [
        {
            "split": split,
            "test_year": test_year,
            "stage": stage,
            "fit_samples": int(len(fit)),
            "validation_samples": int(len(val)),
            "test_samples": int(len(test)),
            "n_labels": int(len(labels)),
            "fit_years": ",".join(map(str, sorted(fit["ev_year"].unique()))),
            "validation_years": ",".join(map(str, sorted(val["ev_year"].unique()))),
        }
    ]
    return metrics, ci_rows, fold_rows


def audit_rows() -> list[dict]:
    return [
        {
            "method": "BR-TFIDF",
            "baseline_group": "exact_standard",
            "source": "Binary relevance with TF-IDF and one-vs-rest logistic regression",
            "implementation": "scikit-learn primitives",
            "adaptation": "stage-specific evidence text",
            "status": "main",
        },
        {
            "method": "Classifier Chains",
            "baseline_group": "exact_standard",
            "source": "Classifier chains for multi-label classification",
            "implementation": "scikit-learn ClassifierChain",
            "adaptation": "support-ordered chain using the same TF-IDF view",
            "status": "main",
        },
        {
            "method": "ML-KNN",
            "baseline_group": "exact_standard",
            "source": "Multi-label k-nearest neighbor family",
            "implementation": "cosine KNN on truncated TF-IDF representation",
            "adaptation": "dimensionality reduction for sparse text distance",
            "status": "main",
        },
        {
            "method": "RAkEL",
            "baseline_group": "exact_standard",
            "source": "Random k-labelset ensemble",
            "implementation": "label-powerset logistic ensembles",
            "adaptation": "random label subsets with validation thresholds",
            "status": "main",
        },
        {
            "method": "Hierarchy-gated BR",
            "baseline_group": "hierarchical_standard",
            "source": "top-down hierarchy-consistent multi-label decision",
            "implementation": "parent score gate over BR label scores",
            "adaptation": "parent induced from NTSB level-2 taxonomy",
            "status": "main",
        },
        {
            "method": "Label-enhanced forest",
            "baseline_group": "recent_mechanism_proxy",
            "source": "label-enhancement and feature-label correlation direction",
            "implementation": "multi-output ExtraTrees over dense TF-IDF factors",
            "adaptation": "proxy, reported as mechanism baseline",
            "status": "main_with_audit",
        },
        {
            "method": "VCLDL/HALB/DyLas/SIHTC adapted",
            "baseline_group": "proxy",
            "source": "recent journal mechanisms summarized through same-input proxy signals",
            "implementation": "reported in supplementary adaptation tables",
            "adaptation": "reported as a supplementary mechanism audit",
            "status": "supplementary",
        },
    ]


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return years


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples = load_samples(
        args.label_level,
        args.min_global_support,
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.data_file,
    )
    if args.smoke and args.max_smoke_samples > 0:
        samples = samples.sort_values(["ev_year", "ev_id"]).groupby("ev_year", group_keys=False).head(args.max_smoke_samples)
    split_names = [name.strip() for name in args.splits.split(",") if name.strip()]
    test_years = parse_years(args.test_years)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    suffix = args.suffix or ("smoke" if args.smoke else "full")

    metric_rows: list[dict] = []
    ci_rows: list[dict] = []
    fold_rows: list[dict] = []
    for split in split_names:
        for test_year in test_years:
            for stage in stages:
                m_rows, c_rows, f_rows = run_fold(args, samples, split, test_year, stage, rng)
                metric_rows.extend(m_rows)
                ci_rows.extend(c_rows)
                fold_rows.extend(f_rows)

    write_csv(OUT_DIR / f"exact_baseline_metrics_{suffix}.csv", metric_rows)
    write_csv(OUT_DIR / f"exact_baseline_ci_{suffix}.csv", ci_rows)
    write_csv(OUT_DIR / f"exact_baseline_folds_{suffix}.csv", fold_rows)
    write_csv(OUT_DIR / f"baseline_audit_{suffix}.csv", audit_rows())

    if metric_rows:
        df = pd.DataFrame(metric_rows)
        summary = (
            df.groupby(["split", "stage", "method"], as_index=False)
            .agg(
                test_years=("test_year", "nunique"),
                micro_f1=("micro_f1", "mean"),
                macro_f1=("macro_f1", "mean"),
                samples_f1=("samples_f1", "mean"),
                jaccard=("jaccard", "mean"),
                mean_binary_entropy=("mean_binary_entropy", "mean"),
            )
            .sort_values(["split", "stage", "micro_f1"], ascending=[True, True, False])
        )
        summary.to_csv(OUT_DIR / f"exact_baseline_summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    meta = {
        "seconds": round(time.time() - start, 3),
        "samples": int(len(samples)),
        "splits": split_names,
        "test_years": test_years,
        "stages": stages,
        "suffix": suffix,
        "data_file": args.data_file,
        "start_year": args.start_year,
        "end_year": args.end_year,
    }
    (OUT_DIR / f"run_meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--data-file", default=str(ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv"))
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--test-years", default="2023")
    parser.add_argument("--splits", default="rolling")
    parser.add_argument("--stages", default="preliminary")
    parser.add_argument("--methods", default="")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=30)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--svd-dims", type=int, default=96)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--rakel-subsets", type=int, default=18)
    parser.add_argument("--rakel-subset-size", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-smoke-samples", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
