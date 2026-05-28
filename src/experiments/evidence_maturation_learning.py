from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ntsb_evidence_maturation"
DATA_CANDIDATES = [
    DATA_DIR / "finding_rows_2020_2025.csv",
    DATA_DIR / "finding_rows_2020_2025_reference.csv",
]
OUT_DIR = ROOT / "results" / "experiments" / "evidence_maturation_learning"

LEAKAGE_TERMS = [
    "probable cause",
    "probable causes",
    "finding",
    "findings",
    "cause",
    "causes",
    "contributing factor",
    "contributing factors",
]

TEXT_FIELDS = {
    "event_aircraft_core": [],
    "event_aircraft_weather": [],
    "preliminary": ["narr_accp"],
    "mature_factual": ["narr_accp", "narr_accf"],
}

STRUCTURED_CORE = [
    "ev_month",
    "ev_state",
    "ev_country",
    "ev_highest_injury",
    "inj_tot_f",
    "inj_tot_s",
    "damage",
    "acft_make",
    "acft_model",
    "acft_category",
    "cert_max_gr_wt",
    "num_eng",
    "phase_flt_spec",
]

STRUCTURED_WEATHER = [
    "light_cond",
    "sky_cond_nonceil",
    "sky_cond_ceil",
    "vis_sm",
    "wx_temp",
    "wx_dew_pt",
    "wind_dir_deg",
    "wind_vel_kts",
    "gust_kts",
    "altimeter",
    "wx_cond_basic",
]

THRESHOLD_GRID = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def norm_token(value: object) -> str:
    text = safe_text(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "missing"


def redact_leakage(text: str) -> str:
    out = text
    for term in LEAKAGE_TERMS:
        out = re.sub(re.escape(term), " ", out, flags=re.IGNORECASE)
    return out


def split_label(description: object, level: int) -> str | None:
    text = safe_text(description).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if not parts:
        return None
    return "-".join(parts[:level])


def load_samples(
    label_level: int,
    min_global_support: int,
    max_rows: int | None = None,
    start_year: int = 2020,
    end_year: int = 2025,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    usecols = [
        "ev_id",
        "ntsb_no",
        "ev_year",
        "ev_month",
        "ev_state",
        "ev_country",
        "ev_highest_injury",
        "inj_tot_f",
        "inj_tot_s",
        "Aircraft_Key",
        "damage",
        "acft_make",
        "acft_model",
        "acft_category",
        "cert_max_gr_wt",
        "num_eng",
        "phase_flt_spec",
        "narr_accp",
        "narr_accf",
        "narr_cause",
        "narr_inc",
        "finding_code",
        "finding_description",
        "Cause_Factor",
    ]
    selected_path = Path(data_path) if data_path else next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[-1])
    available = pd.read_csv(selected_path, nrows=0).columns.tolist()
    requested = [c for c in usecols + STRUCTURED_WEATHER if c in available]
    df = pd.read_csv(selected_path, dtype=str, usecols=requested)
    if max_rows:
        df = df.head(max_rows).copy()

    df["label"] = df["finding_description"].map(lambda x: split_label(x, label_level))
    df = df[df["label"].notna()].copy()

    support = Counter(df["label"])
    valid_labels = {label for label, count in support.items() if count >= min_global_support}
    df = df[df["label"].isin(valid_labels)].copy()

    group_cols = ["ev_id", "Aircraft_Key"]
    records = []
    for (ev_id, aircraft_key), group in df.groupby(group_cols, sort=False):
        first = group.iloc[0].to_dict()
        labels = sorted(set(group["label"].dropna()))
        if not labels:
            continue
        first["ev_id"] = ev_id
        first["Aircraft_Key"] = aircraft_key
        first["ev_year"] = int(float(first.get("ev_year") or 0))
        first["labels"] = labels
        first["label_count"] = len(labels)
        first["finding_count"] = int(group["finding_description"].fillna("").str.len().gt(0).sum())
        records.append(first)

    samples = pd.DataFrame(records)
    samples = samples[(samples["ev_year"] >= start_year) & (samples["ev_year"] <= end_year)].copy()
    samples["labels_joined"] = samples["labels"].map(lambda xs: "||".join(xs))
    return samples


def structured_tokens(row: pd.Series, include_weather: bool) -> list[str]:
    tokens = []
    for col in STRUCTURED_CORE:
        if col in row.index:
            tokens.append(f"{col}_{norm_token(row[col])}")
    if include_weather:
        for col in STRUCTURED_WEATHER:
            if col in row.index:
                tokens.append(f"{col}_{norm_token(row[col])}")
    return tokens


def build_document(row: pd.Series, stage: str) -> str:
    include_weather = stage != "event_aircraft_core"
    parts = structured_tokens(row, include_weather=include_weather)
    for field in TEXT_FIELDS[stage]:
        parts.append(redact_leakage(safe_text(row.get(field, ""))))
    return " ".join(parts)


def entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.30)
    for j in range(y_true.shape[1]):
        best_score = -1.0
        best_t = 0.30
        if y_true[:, j].sum() == 0:
            thresholds[j] = best_t
            continue
        for t in THRESHOLD_GRID:
            pred = proba[:, j] >= t
            score = f1_score(y_true[:, j], pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds[j] = best_t
    return thresholds


def fit_stage(docs: list[str], y: np.ndarray, jobs: int, max_features: int) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
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
        C=1.0,
        random_state=42,
    )
    model = OneVsRestClassifier(estimator, n_jobs=jobs)
    model.fit(x, y)
    return vectorizer, model


def predict_stage(vectorizer: TfidfVectorizer, model: OneVsRestClassifier, docs: list[str]) -> np.ndarray:
    x = vectorizer.transform(docs)
    probs = model.predict_proba(x)
    if isinstance(probs, list):
        probs = np.vstack([p[:, 1] for p in probs]).T
    return np.asarray(probs)


def label_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], stage: str, test_year: int) -> list[dict]:
    rows = []
    for idx, label in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        rows.append(
            {
                "test_year": test_year,
                "stage": stage,
                "label": label,
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
            }
        )
    return rows


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, stage: str, test_year: int, n_labels: int) -> dict:
    return {
        "test_year": test_year,
        "stage": stage,
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(n_labels),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "mean_binary_entropy": entropy(proba),
    }


def top_terms(vectorizer: TfidfVectorizer, model: OneVsRestClassifier, labels: list[str], stage: str, test_year: int) -> list[dict]:
    names = np.asarray(vectorizer.get_feature_names_out())
    rows = []
    for label, estimator in zip(labels, model.estimators_):
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            continue
        weights = coef.ravel()
        top_idx = np.argsort(weights)[-15:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "test_year": test_year,
                    "stage": stage,
                    "label": label,
                    "rank": rank,
                    "term": names[idx],
                    "weight": float(weights[idx]),
                }
            )
    return rows


def classify_maturation(row: pd.Series) -> str:
    prelim = row.get("preliminary_f1", 0.0)
    mature = row.get("mature_factual_f1", 0.0)
    support = row.get("support", 0.0)
    prelim_precision = row.get("preliminary_precision", 0.0)
    delta = mature - prelim
    if support < 20:
        return "low_support"
    if prelim >= 0.35 and delta <= 0.10:
        return "early_closable"
    if mature >= 0.25 and delta >= 0.15:
        return "late_emerging"
    if prelim_precision < 0.25 and row.get("preliminary_predicted_positive", 0.0) > support:
        return "unstable"
    return "mixed"


def aggregate_label_maturation(label_rows: pd.DataFrame) -> pd.DataFrame:
    if label_rows.empty:
        return label_rows
    agg = (
        label_rows.groupby(["stage", "label"], as_index=False)
        .agg(
            support=("support", "sum"),
            predicted_positive=("predicted_positive", "sum"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
    )
    wide = agg.pivot(index="label", columns="stage")
    flat = pd.DataFrame(index=wide.index)
    for stage in ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]:
        for metric in ["support", "predicted_positive", "precision", "recall", "f1"]:
            if (metric, stage) in wide.columns:
                flat[f"{stage}_{metric}"] = wide[(metric, stage)]
    flat = flat.reset_index()
    support_cols = [c for c in flat.columns if c.endswith("_support")]
    flat["support"] = flat[support_cols].max(axis=1)
    for col in flat.columns:
        if col != "label":
            flat[col] = flat[col].fillna(0.0)
    flat["mature_minus_prelim_f1"] = flat.get("mature_factual_f1", 0.0) - flat.get("preliminary_f1", 0.0)
    flat["prelim_minus_structured_f1"] = flat.get("preliminary_f1", 0.0) - flat.get("event_aircraft_weather_f1", 0.0)
    flat["maturation_class"] = flat.apply(classify_maturation, axis=1)
    return flat.sort_values(["maturation_class", "support", "mature_factual_f1"], ascending=[True, False, False])


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        args.label_level,
        args.min_global_support,
        max_rows=None,
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.data_file,
    )
    if args.smoke:
        samples = samples[samples["ev_year"].isin([2020, 2021, 2022])].copy()

    sample_path = OUT_DIR / ("samples_smoke.csv" if args.smoke else "samples_full.csv")
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")

    stages = ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]
    if args.smoke:
        test_years = [2022]
    else:
        test_years = [2023, 2024, 2025]

    global_rows: list[dict] = []
    per_label_rows: list[dict] = []
    term_rows: list[dict] = []
    fold_notes: list[dict] = []

    workers = max(1, min(args.jobs, os.cpu_count() or 1))
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
            labels = [label for label, _ in train_label_counts.most_common(args.max_labels) if label in set(labels)]
            labels = sorted(labels)
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

        fold_notes.append(
            {
                "test_year": test_year,
                "fit_years": ",".join(map(str, sorted(fit["ev_year"].unique()))),
                "validation_year": val_year,
                "train_samples": int(len(train_all)),
                "test_samples": int(len(test)),
                "labels": int(len(labels)),
            }
        )

        for stage in stages:
            fit_docs = [build_document(row, stage) for _, row in fit.iterrows()]
            val_docs = [build_document(row, stage) for _, row in val.iterrows()]
            train_docs = [build_document(row, stage) for _, row in train_all.iterrows()]
            test_docs = [build_document(row, stage) for _, row in test.iterrows()]

            cal_vectorizer, cal_model = fit_stage(fit_docs, y_fit, workers, args.max_features)
            val_proba = predict_stage(cal_vectorizer, cal_model, val_docs)
            thresholds = tune_thresholds(y_val, val_proba)

            vectorizer, model = fit_stage(train_docs, y_train, workers, args.max_features)
            test_proba = predict_stage(vectorizer, model, test_docs)
            test_pred = test_proba >= thresholds

            global_rows.append(global_metrics(y_test, test_pred, test_proba, stage, test_year, len(labels)))
            per_label_rows.extend(label_metrics(y_test, test_pred, labels, stage, test_year))
            term_rows.extend(top_terms(vectorizer, model, labels, stage, test_year))

    suffix = "smoke" if args.smoke else "full"
    write_csv(OUT_DIR / f"stage_metrics_by_year_{suffix}.csv", global_rows)
    write_csv(OUT_DIR / f"label_metrics_by_year_{suffix}.csv", per_label_rows)
    write_csv(OUT_DIR / f"interpretable_top_terms_{suffix}.csv", term_rows)
    write_csv(OUT_DIR / f"fold_notes_{suffix}.csv", fold_notes)

    label_df = pd.DataFrame(per_label_rows)
    maturation = aggregate_label_maturation(label_df)
    maturation.to_csv(OUT_DIR / f"label_maturation_taxonomy_{suffix}.csv", index=False, encoding="utf-8-sig")

    global_df = pd.DataFrame(global_rows)
    if not global_df.empty:
        stage_avg = (
            global_df.groupby("stage", as_index=False)
            .agg(
                test_years=("test_year", "nunique"),
                n_samples=("n_samples", "sum"),
                micro_f1=("micro_f1", "mean"),
                macro_f1=("macro_f1", "mean"),
                samples_f1=("samples_f1", "mean"),
                jaccard_samples=("jaccard_samples", "mean"),
                mean_binary_entropy=("mean_binary_entropy", "mean"),
            )
            .sort_values("stage")
        )
        stage_avg.to_csv(OUT_DIR / f"stage_metrics_average_{suffix}.csv", index=False, encoding="utf-8-sig")

        lookup = stage_avg.set_index("stage")
        acquisition_rows = []
        pairs = [
            ("weather_structured", "event_aircraft_core", "event_aircraft_weather"),
            ("preliminary_narrative", "event_aircraft_weather", "preliminary"),
            ("mature_factual_narrative", "preliminary", "mature_factual"),
        ]
        for name, base, added in pairs:
            if base in lookup.index and added in lookup.index:
                acquisition_rows.append(
                    {
                        "evidence_type": name,
                        "base_stage": base,
                        "added_stage": added,
                        "delta_micro_f1": float(lookup.loc[added, "micro_f1"] - lookup.loc[base, "micro_f1"]),
                        "delta_macro_f1": float(lookup.loc[added, "macro_f1"] - lookup.loc[base, "macro_f1"]),
                        "delta_samples_f1": float(lookup.loc[added, "samples_f1"] - lookup.loc[base, "samples_f1"]),
                        "uncertainty_reduction": float(
                            lookup.loc[base, "mean_binary_entropy"] - lookup.loc[added, "mean_binary_entropy"]
                        ),
                    }
                )
        write_csv(OUT_DIR / f"counterfactual_evidence_acquisition_{suffix}.csv", acquisition_rows)

    run_meta = {
        "smoke": args.smoke,
        "label_level": args.label_level,
        "min_global_support": args.min_global_support,
        "min_train_support": args.min_train_support,
        "max_features": args.max_features,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "data_file": args.data_file,
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
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
