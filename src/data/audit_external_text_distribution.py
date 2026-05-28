from __future__ import annotations

import csv
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.benchmark_extensions import binarize, predict_br, select_labels, tune_thresholds_local  # noqa: E402
from src.experiments.evidence_maturation_learning import (  # noqa: E402
    ROOT,
    build_document,
    fit_stage,
    load_samples,
)


csv.field_size_limit(2_147_483_647)

RAW_DIR = ROOT / "data" / "raw" / "faa_asias_aids"
OUT_DIR = ROOT / "results" / "external_robustness"
DATA_2008 = ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv"
TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


def nonmissing(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text not in {"\\N", "N/A", "NA", "NULL", "."})


def read_faa_records() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RAW_DIR.glob("a*.zip")):
        with zipfile.ZipFile(path) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            if not names:
                continue
            text = zf.read(names[0]).decode("latin-1", errors="replace")
        for row in csv.DictReader(text.splitlines(), delimiter="\t"):
            row["_archive"] = path.name
            rows.append(row)
    return rows


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)), axis=1)


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
    samples = load_samples(2, 30, start_year=2008, end_year=2025, data_path=DATA_2008)
    fit = samples[(samples["ev_year"] >= 2008) & (samples["ev_year"] <= 2022)].copy()
    ref = samples[samples["ev_year"].isin([2023, 2024])].copy()
    labels = select_labels(samples[samples["ev_year"] <= 2022].copy(), 30, 24)
    y_fit = binarize(fit, labels)

    fit_docs = [build_document(row, "preliminary") for _, row in fit.iterrows()]
    ref_docs = [build_document(row, "preliminary") for _, row in ref.iterrows()]

    faa_records = read_faa_records()
    faa_docs = []
    for row in faa_records:
        narrative = row.get("c119", "")
        parts = []
        for field in ["c6", "c11", "c12", "c22", "c23", "c24", "c34", "c35", "c61", "c62", "c95", "c96"]:
            if nonmissing(row.get(field)):
                parts.append(f"faa_{field}_{str(row[field]).strip().lower().replace(' ', '_')}")
        if nonmissing(narrative):
            parts.append(str(narrative))
        faa_docs.append(" ".join(parts))

    ntsb_counter = Counter(token for doc in fit_docs for token in tokens(doc))
    faa_counter = Counter(token for doc in faa_docs for token in tokens(doc))
    ntsb_vocab = set(ntsb_counter)
    faa_vocab = set(faa_counter)
    ntsb_top = {token for token, _ in ntsb_counter.most_common(5000)}
    faa_top = {token for token, _ in faa_counter.most_common(5000)}

    vectorizer, model = fit_stage(fit_docs, y_fit, jobs=2, max_features=25000)
    ref_scores = predict_br(model, vectorizer.transform(ref_docs))
    faa_scores = predict_br(model, vectorizer.transform(faa_docs))
    ref_entropy = entropy(ref_scores)
    faa_entropy = entropy(faa_scores)
    ref_max = ref_scores.max(axis=1)
    faa_max = faa_scores.max(axis=1)

    rows = [
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "narrative records",
            "result": sum(1 for row in faa_records if nonmissing(row.get("c119"))),
            "reference": "FAA c119 narrative field",
        },
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "token vocabulary overlap with NTSB preliminary",
            "result": len(faa_vocab & ntsb_vocab) / max(1, len(faa_vocab)),
            "reference": "unique token overlap",
        },
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "top-5000 vocabulary overlap",
            "result": len(faa_top & ntsb_top) / max(1, len(faa_top)),
            "reference": "frequent-token overlap",
        },
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "mean score entropy",
            "result": float(np.mean(faa_entropy)),
            "reference": f"NTSB 2023--2024 mean {np.mean(ref_entropy):.3f}",
        },
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "score entropy KS statistic",
            "result": float(ks_2samp(ref_entropy, faa_entropy).statistic),
            "reference": "NTSB preliminary 2023--2024",
        },
        {
            "source": "FAA ASIAS AIDS",
            "units": len(faa_docs),
            "years": "2015--2026",
            "measured_quantity": "mean maximum label score",
            "result": float(np.mean(faa_max)),
            "reference": f"NTSB 2023--2024 mean {np.mean(ref_max):.3f}",
        },
    ]
    write_csv(OUT_DIR / "external_text_distribution_audit.csv", rows)
    print(f"Wrote {OUT_DIR / 'external_text_distribution_audit.csv'}")


if __name__ == "__main__":
    main()
