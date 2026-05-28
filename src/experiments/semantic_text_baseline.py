from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize

import sys

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import (  # noqa: E402
    ROOT,
    build_document,
    fit_stage,
    load_samples,
    predict_stage,
)
from src.experiments.benchmark_extensions import (  # noqa: E402
    binarize,
    make_split,
    metric_row as benchmark_metric_row,
    select_labels,
    tune_thresholds_local,
)


OUT_DIR = ROOT / "results" / "experiments" / "semantic_text_baseline"
CACHE_DIR = ROOT / "results" / "cache" / "semantic_text_baseline"


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


def entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, method: str, stage: str, test_year: int, seconds: float) -> dict:
    return {
        "method": method,
        "stage": stage,
        "test_year": int(test_year),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "mean_binary_entropy": entropy(scores),
        "seconds": round(seconds, 3),
    }


def chunk_document(doc: str, chunk_words: int, max_chunks: int) -> list[str]:
    words = doc.split()
    if not words:
        return [""]
    chunks = [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]
    return chunks[:max_chunks] or [doc]


def cache_key(model_name: str, docs: list[str], chunk_words: int, max_chunks: int) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(chunk_words).encode("ascii"))
    h.update(str(max_chunks).encode("ascii"))
    for doc in docs:
        h.update(b"\0")
        h.update(doc.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def segmented_embeddings(
    encoder: SentenceTransformer,
    model_name: str,
    docs: list[str],
    chunk_words: int,
    max_chunks: int,
    batch_size: int,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key(model_name, docs, chunk_words, max_chunks)}.npy"
    if path.exists():
        return np.load(path)

    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for doc in docs:
        start = len(chunks)
        chunks.extend(chunk_document(doc, chunk_words, max_chunks))
        spans.append((start, len(chunks)))
    chunk_vectors = np.asarray(
        encoder.encode(chunks, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )
    pooled = []
    for start, end in spans:
        mat = chunk_vectors[start:end]
        pooled.append(np.concatenate([mat.mean(axis=0), mat.max(axis=0)]))
    out = normalize(np.vstack(pooled), norm="l2").astype(np.float32)
    np.save(path, out)
    return out


def fit_ovr(x, y: np.ndarray, jobs: int, c_value: float) -> OneVsRestClassifier:
    estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", C=c_value)
    model = OneVsRestClassifier(estimator, n_jobs=jobs)
    model.fit(x, y)
    return model


def predict_ovr(model: OneVsRestClassifier, x) -> np.ndarray:
    probs = model.predict_proba(x)
    if isinstance(probs, list):
        probs = np.vstack([p[:, 1] for p in probs]).T
    return np.asarray(probs)


def build_fusion_features(
    docs_fit: list[str],
    docs_val: list[str],
    docs_test: list[str],
    emb_fit: np.ndarray,
    emb_val: np.ndarray,
    emb_test: np.ndarray,
    max_features: int,
    semantic_weight: float,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )
    x_fit_t = vectorizer.fit_transform(docs_fit)
    x_val_t = vectorizer.transform(docs_val)
    x_test_t = vectorizer.transform(docs_test)
    x_fit = sparse.hstack([x_fit_t, sparse.csr_matrix(emb_fit * semantic_weight)], format="csr")
    x_val = sparse.hstack([x_val_t, sparse.csr_matrix(emb_val * semantic_weight)], format="csr")
    x_test = sparse.hstack([x_test_t, sparse.csr_matrix(emb_test * semantic_weight)], format="csr")
    return x_fit, x_val, x_test


def run_stage(args: argparse.Namespace, samples, stage: str, test_year: int) -> list[dict]:
    fit, val, test = make_split(samples, "rolling", test_year)
    labels = select_labels(samples[samples["ev_year"] < test_year].copy(), args.min_train_support, args.max_labels)
    if fit.empty or val.empty or test.empty or not labels:
        return []

    y_fit = binarize(fit, labels)
    y_val = binarize(val, labels)
    y_test = binarize(test, labels)
    keep = y_test.sum(axis=1) > 0
    test = test.loc[keep].copy()
    y_test = y_test[keep]
    if y_test.shape[0] == 0:
        return []

    docs_fit = [build_document(row, stage) for _, row in fit.iterrows()]
    docs_val = [build_document(row, stage) for _, row in val.iterrows()]
    docs_test = [build_document(row, stage) for _, row in test.iterrows()]

    rows: list[dict] = []
    start = time.time()
    tfidf_vec, tfidf_model = fit_stage(docs_fit, y_fit, jobs=args.jobs, max_features=args.tfidf_features)
    tfidf_val = predict_stage(tfidf_vec, tfidf_model, docs_val)
    tfidf_test = predict_stage(tfidf_vec, tfidf_model, docs_test)
    thresholds = tune_thresholds_local(y_val, tfidf_val)
    row = benchmark_metric_row(y_test, tfidf_test >= thresholds, tfidf_test, "BR-TFIDF check", "rolling", stage, test_year)
    row["seconds"] = round(time.time() - start, 3)
    rows.append(row)

    encoder = SentenceTransformer(args.model_name)
    start = time.time()
    emb_fit = segmented_embeddings(encoder, args.model_name, docs_fit, args.chunk_words, args.max_chunks, args.batch_size)
    emb_val = segmented_embeddings(encoder, args.model_name, docs_val, args.chunk_words, args.max_chunks, args.batch_size)
    emb_test = segmented_embeddings(encoder, args.model_name, docs_test, args.chunk_words, args.max_chunks, args.batch_size)

    for c_value in args.c_values:
        start_c = time.time()
        emb_model = fit_ovr(emb_fit, y_fit, args.jobs, c_value)
        emb_val_scores = predict_ovr(emb_model, emb_val)
        emb_test_scores = predict_ovr(emb_model, emb_test)
        thresholds = tune_thresholds_local(y_val, emb_val_scores)
        row = metric_row(
            y_test,
            emb_test_scores >= thresholds,
            emb_test_scores,
            f"Segmented MiniLM-BR C={c_value:g}",
            stage,
            test_year,
            time.time() - start_c,
        )
        rows.append(row)

    for weight in args.semantic_weights:
        x_fit, x_val, x_test = build_fusion_features(
            docs_fit,
            docs_val,
            docs_test,
            emb_fit,
            emb_val,
            emb_test,
            args.tfidf_features,
            weight,
        )
        for c_value in args.c_values:
            start_c = time.time()
            fusion_model = fit_ovr(x_fit, y_fit, args.jobs, c_value)
            val_scores = predict_ovr(fusion_model, x_val)
            test_scores = predict_ovr(fusion_model, x_test)
            thresholds = tune_thresholds_local(y_val, val_scores)
            row = benchmark_metric_row(
                y_test,
                test_scores >= thresholds,
                test_scores,
                f"Frozen semantic-fusion BR w={weight:g} C={c_value:g}",
                "rolling",
                stage,
                test_year,
            )
            row["seconds"] = round(time.time() - start_c, 3)
            rows.append(row)

    for row in rows:
        row["model_name"] = args.model_name
        row["n_fit"] = len(fit)
        row["n_val"] = len(val)
        row["n_test"] = len(test)
        row["n_labels"] = len(labels)
        row["embedding_seconds"] = round(time.time() - start, 3)
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    br = next((r for r in rows if r["method"] == "BR-TFIDF check"), None)
    best = max((r for r in rows if r["method"] != "BR-TFIDF check"), key=lambda x: x["micro_f1"], default=None)
    lines = ["# Semantic text baseline smoke results", ""]
    for row in sorted(rows, key=lambda x: x["micro_f1"], reverse=True):
        lines.append(
            f"- {row['method']} ({row['stage']} {row['test_year']}): "
            f"Micro-F1={row['micro_f1']:.3f}, Macro-F1={row['macro_f1']:.3f}, "
            f"Samples-F1={row['samples_f1']:.3f}, Jaccard={row['jaccard']:.3f}"
        )
    if br and best:
        lines.append("")
        lines.append(f"Best semantic method change against BR-TFIDF: {best['micro_f1'] - br['micro_f1']:+.3f} Micro-F1.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--test-years", nargs="+", type=int, default=[2023])
    parser.add_argument("--stages", nargs="+", default=["preliminary"])
    parser.add_argument("--data-file", default=str(ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv"))
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=30)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tfidf-features", type=int, default=40000)
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--semantic-weights", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--suffix", default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        args.label_level,
        args.min_global_support,
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=Path(args.data_file),
    )
    rows: list[dict] = []
    for test_year in args.test_years:
        for stage in args.stages:
            rows.extend(run_stage(args, samples, stage, test_year))
    metrics_path = OUT_DIR / f"semantic_text_baseline_metrics_{args.suffix}.csv"
    write_csv(metrics_path, rows)
    write_summary(OUT_DIR / f"semantic_text_baseline_summary_{args.suffix}.md", rows)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
