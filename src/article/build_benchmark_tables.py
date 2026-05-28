from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.evidence_maturation_learning import load_samples  # noqa: E402


ARTICLE = ROOT / "article" / "elsarticle_manuscript"
SUPP = ROOT / "article" / "supplementary"
RESULTS = ROOT / "results" / "experiments" / "benchmark_extensions"
EVIDENCE = ROOT / "results" / "experiments" / "evidence_maturation_learning"
FDFHC = ROOT / "results" / "experiments" / "f_dfhc_u_closure"
SEMANTIC = ROOT / "results" / "experiments" / "semantic_text_baseline"
EXTERNAL = ROOT / "results" / "external_robustness"
QUALITY = ROOT / "results" / "data_quality"
RECENT_MECH = ROOT / "results" / "experiments" / "strong_baselines_evidence_maturation"
DATA_2008 = ROOT / "data" / "processed" / "ntsb_evidence_maturation" / "finding_rows_2008_2025.csv"

MAIN_TEST_YEARS = [2023, 2024]
FRESHNESS_YEAR = 2025
STAGE_ORDER = ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]
STAGE_LABELS = {
    "event_aircraft_core": "Core",
    "event_aircraft_weather": "Core + weather",
    "preliminary": "Preliminary",
    "mature_factual": "Mature factual",
}
KEEP_METHODS = [
    "BR-TFIDF",
    "Classifier Chains",
    "ML-KNN",
    "RAkEL",
    "Hierarchy-gated BR",
    "Label-enhanced forest",
    "Frozen semantic-fusion BR",
]
RECENT_METHODS = [
    "BR-TFIDF",
    "VCLDL-adapted",
    "SIHTC-adapted",
    "DyLas-adapted",
    "HALB-adapted",
    "LSPCL-adapted",
    "MatchXML-adapted",
]
RECENT_DISPLAY = {
    "BR-TFIDF": "BR-TFIDF",
    "VCLDL-adapted": "VCLDL",
    "SIHTC-adapted": "SIHTC",
    "DyLas-adapted": "DyLas",
    "HALB-adapted": "HALB",
    "LSPCL-adapted": "LSPCL",
    "MatchXML-adapted": "MatchXML",
}


def f3(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "--"
    return f"{x:.3f}"


def f4(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "--"
    return f"{x:.4f}"


def tex_escape(text: object) -> str:
    out = str(text)
    return (
        out.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    part = df[df["test_year"].isin(years)].copy()
    metric_cols = ["micro_f1", "macro_f1", "samples_f1", "jaccard", "mean_binary_entropy"]
    return part.groupby(["stage", "method"], as_index=False)[metric_cols].mean()


def build_data_summary() -> None:
    samples_full = load_samples(2, 30, start_year=2008, end_year=2025, data_path=DATA_2008)
    samples_main = load_samples(2, 30, start_year=2020, end_year=2025)
    labels_full = Counter(label for labels in samples_full["labels"] for label in labels)
    labels_main = Counter(label for labels in samples_main["labels"] for label in labels)
    main_test = samples_main[samples_main["ev_year"].isin(MAIN_TEST_YEARS)]
    freshness = samples_main[samples_main["ev_year"].eq(FRESHNESS_YEAR)]
    text = rf"""\begin{{table}}[t]
\centering
\caption{{Closed-label benchmark and validation summary.}}
\label{{tab:data_summary}}
\papertablesize
\begin{{tabularx}}{{\linewidth}}{{lY}}
\toprule
Item & Value \\
\midrule
Observation unit & NTSB event-aircraft record \\
Closed-label benchmark period & 2008--2025 \\
Closed-label benchmark units & {len(samples_full):,} \\
Benchmark final finding labels & {sum(1 for value in labels_full.values() if value >= 30)} level-2 taxonomy labels \\
Mean labels per benchmark unit & {samples_full["labels"].map(len).mean():.2f} \\
Prospective modeling window & 2020--2025 \\
Prospective units & {len(samples_main):,} \\
Main forward-test units & {len(main_test):,} across 2023 and 2024 \\
Freshness audit units & {len(freshness):,} in 2025 \\
Main target labels & {sum(1 for value in labels_main.values() if value >= 30)} level-2 taxonomy labels \\
Historical robustness & 2008--2019 fitting with 2020--2025 forward tests; rolling tests from 2015 to 2025 \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    write(ARTICLE / "tables" / "tab1_data_summary.tex", text)


def build_data_completeness_table() -> None:
    audit = pd.read_csv(QUALITY / "ntsb_stage_completeness_by_year.csv")
    audit = audit[audit["year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR])]
    body = []
    for _, r in audit.iterrows():
        role = "main" if int(r["year"]) in MAIN_TEST_YEARS else "freshness"
        body.append(
            f"{int(r['year'])} & {int(r['event_aircraft_units']):,} & "
            f"{f3(r['final_finding_rate'])} & {f3(r['preliminary_narrative_rate'])} & "
            f"{f3(r['mature_factual_narrative_rate'])} & {role} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Forward-year evidence completeness audit.}\n\\label{tab:data_completeness}\n\\papertablesize\n\\begin{tabular}{rrrrrl}\n\\toprule\nYear & Units & Final labels & Prelim. narrative & Factual narrative & Role \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Rates are unit-level coverage shares. The 2025 records are retained for freshness auditing because preliminary narrative coverage is sparse in the public export.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab_data_completeness.tex", text)


def build_stage_metrics_table() -> None:
    rows = pd.read_csv(EVIDENCE / "stage_metrics_by_year_full.csv")
    rows = rows[rows["test_year"].isin(MAIN_TEST_YEARS)]
    stage = (
        rows.groupby("stage", as_index=False)[
            ["micro_f1", "macro_f1", "samples_f1", "jaccard_samples", "mean_binary_entropy"]
        ].mean()
    )
    stage = stage.set_index("stage").loc[STAGE_ORDER].reset_index()
    best_micro = stage["micro_f1"].max()
    best_macro = stage["macro_f1"].max()
    best_samples = stage["samples_f1"].max()
    best_jaccard = stage["jaccard_samples"].max()
    best_entropy = stage["mean_binary_entropy"].min()
    body = []
    for _, r in stage.iterrows():
        values = {
            "micro": f3(r["micro_f1"]),
            "macro": f3(r["macro_f1"]),
            "samples": f3(r["samples_f1"]),
            "jaccard": f3(r["jaccard_samples"]),
            "entropy": f3(r["mean_binary_entropy"]),
        }
        if abs(r["micro_f1"] - best_micro) < 1e-12:
            values["micro"] = rf"\textbf{{{values['micro']}}}"
        if abs(r["macro_f1"] - best_macro) < 1e-12:
            values["macro"] = rf"\textbf{{{values['macro']}}}"
        if abs(r["samples_f1"] - best_samples) < 1e-12:
            values["samples"] = rf"\textbf{{{values['samples']}}}"
        if abs(r["jaccard_samples"] - best_jaccard) < 1e-12:
            values["jaccard"] = rf"\textbf{{{values['jaccard']}}}"
        if abs(r["mean_binary_entropy"] - best_entropy) < 1e-12:
            values["entropy"] = rf"\textbf{{{values['entropy']}}}"
        body.append(
            f"{STAGE_LABELS[r['stage']]} & {values['micro']} & {values['macro']} & "
            f"{values['samples']} & {values['jaccard']} & {values['entropy']} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Evidence-stage performance under 2023--2024 forward validation.}\n\\label{tab:stage_metrics}\n\\papertablesize\n\\begin{tabularx}{\\linewidth}{Yrrrrr}\n\\toprule\nEvidence stage & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard & Entropy \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabularx}\n\\begin{flushleft}\\footnotesize Higher F1 and Jaccard values indicate better final finding closure; lower entropy indicates lower final-finding uncertainty.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab2_stage_metrics.tex", text)


def build_faithful_baseline_table() -> None:
    metrics = pd.read_csv(RESULTS / "exact_baseline_metrics_rolling_2023_2025_faithful.csv")
    semantic_path = SEMANTIC / "semantic_text_baseline_metrics_aligned_main_2023_2024.csv"
    if semantic_path.exists():
        semantic = pd.read_csv(semantic_path)
        semantic = semantic[semantic["method"].eq("Frozen semantic-fusion BR w=1 C=1")].copy()
        semantic["method"] = "Frozen semantic-fusion BR"
        for col in ["split", "test_year", "stage", "method", "micro_f1", "macro_f1", "samples_f1", "jaccard", "mean_binary_entropy"]:
            if col not in semantic.columns:
                semantic[col] = math.nan
        metrics = pd.concat([metrics, semantic[metrics.columns.intersection(semantic.columns)]], ignore_index=True, sort=False)
    rows = aggregate(metrics, MAIN_TEST_YEARS)
    name_map = {"Hierarchy-gated BR": "H-gated BR", "Label-enhanced forest": "LE forest", "Frozen semantic-fusion BR": "Semantic-fusion BR"}
    body = []
    for stage in ["preliminary", "mature_factual"]:
        stage_rows = rows[(rows["stage"] == stage) & (rows["method"].isin(KEEP_METHODS))]
        best = stage_rows["micro_f1"].max()
        for _, r in stage_rows.sort_values("micro_f1", ascending=False).iterrows():
            method = name_map.get(r["method"], r["method"])
            micro = f3(r["micro_f1"])
            if abs(r["micro_f1"] - best) < 1e-12:
                micro = rf"\textbf{{{micro}}}"
            stage_label = "Preliminary" if stage == "preliminary" else "Mature factual"
            body.append(
                f"{stage_label} & {tex_escape(method)} & {micro} & {f3(r['macro_f1'])} & {f3(r['samples_f1'])} & {f3(r['jaccard'])} \\\\"
            )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Stage-aligned standard baselines on 2023--2024 tests.}\n\\label{tab:baseline_comparison}\n\\papertablesize\n\\begin{tabular}{llrrrr}\n\\toprule\nStage & Method & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Bold values mark the largest Micro-F1 within each evidence stage. H-gated BR denotes hierarchy-gated binary relevance; LE forest denotes label-enhanced forest. Semantic-fusion BR uses frozen sentence-encoder embeddings concatenated with lexical evidence features.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab3_baseline_comparison.tex", text)


def build_recent_mechanism_table() -> None:
    target = ARTICLE / "tables" / "tab_recent_mechanisms.tex"
    if target.exists():
        target.unlink()


def build_benchmark_robustness_table() -> None:
    fixed = pd.read_csv(RESULTS / "exact_baseline_summary_fixed_hist_2020_2025_prelim.csv")
    rolling = pd.read_csv(RESULTS / "exact_baseline_summary_rolling_2015_2025_core.csv")
    modern_metrics = pd.read_csv(RESULTS / "exact_baseline_metrics_rolling_2023_2025_faithful.csv")
    modern = aggregate(modern_metrics, MAIN_TEST_YEARS)
    freshness = aggregate(modern_metrics, [FRESHNESS_YEAR])
    rows = []
    best_fixed = fixed.sort_values("micro_f1", ascending=False).iloc[0]
    rows.append(["Hist.", "2020--2025", "Prelim.", best_fixed["method"], best_fixed["micro_f1"], best_fixed["macro_f1"], "fixed"])
    for stage, label in [("preliminary", "Prelim."), ("mature_factual", "Mature")]:
        best = rolling[rolling["stage"] == stage].sort_values("micro_f1", ascending=False).iloc[0]
        rows.append(["Rolling", "2015--2025", label, best["method"], best["micro_f1"], best["macro_f1"], "rolling"])
    for stage, label in [("preliminary", "Prelim."), ("mature_factual", "Mature")]:
        best = modern[modern["stage"] == stage].sort_values("micro_f1", ascending=False).iloc[0]
        rows.append(["Main", "2023--2024", label, best["method"], best["micro_f1"], best["macro_f1"], "rolling"])
    for stage, label in [("preliminary", "Prelim."), ("mature_factual", "Mature")]:
        best = freshness[freshness["stage"] == stage].sort_values("micro_f1", ascending=False).iloc[0]
        rows.append(["Fresh", "2025", label, best["method"], best["micro_f1"], best["macro_f1"], "audit"])
    best_micro = max(float(r[4]) for r in rows)
    body = []
    for setting, years, stage, method, micro, macro, protocol in rows:
        micro_s = f3(micro)
        if abs(float(micro) - best_micro) < 1e-12:
            micro_s = rf"\textbf{{{micro_s}}}"
        method_short = str(method).replace("Hierarchy-gated BR", "H-gated BR")
        body.append(f"{setting} & {years} & {stage} & {tex_escape(method_short)} & {micro_s} & {f3(macro)} & {protocol} \\\\")
    text = "\\begin{table}[t]\n\\centering\n\\caption{Historical, rolling, and freshness robustness.}\n\\label{tab:benchmark_robustness}\n\\papertablesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llllrrl}\n\\toprule\nSet & Years & Stage & Best & Micro-F1 & Macro-F1 & Protocol \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Fresh denotes the 2025 audit split. Higher Micro-F1 and Macro-F1 indicate better forward closure.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab9_benchmark_robustness.tex", text)


def summarize_fdfhc(years: list[int]) -> pd.DataFrame:
    df = pd.read_csv(FDFHC / "f_dfhc_u_metrics_full.csv")
    df = df[df["test_year"].isin(years)].copy()
    close_negative = df["closed_decisions"] - df["close_positive_decisions"]
    false_positive = df["close_positive_decisions"] - df["true_positive_closed"]
    false_negative = df["wrong_closed_decisions"] - false_positive
    df["close_negative_accuracy"] = (close_negative - false_negative) / close_negative
    rate_cols = [
        "close_positive_precision",
        "close_negative_accuracy",
        "positive_coverage",
        "event_positive_coverage",
        "workload_reduction",
        "overall_coverage",
        "empirical_all_risk",
        "certified_all_bound",
        "late_premature_closure_risk",
        "parent_close_positive_precision",
        "child_close_positive_precision",
        "parent_positive_coverage",
        "child_positive_coverage",
        "parent_f1",
        "child_f1",
    ]
    out = df.groupby("method", as_index=False)[rate_cols].mean()
    sums = df.groupby("method", as_index=False)[["projected_hierarchy_violations", "raw_hierarchy_violations"]].sum()
    return out.merge(sums, on="method", how="left")


def build_fdfhc_event_table() -> None:
    rows = summarize_fdfhc(MAIN_TEST_YEARS)
    order = [
        "BR-TFIDF threshold",
        "ES3D",
        "Instancewise RCPS",
        "LTT without hierarchy",
        "F-DFHC without utility",
        "F-DFHC-U w/o projection",
        "F-DFHC-U full",
    ]
    name_map = {
        "BR-TFIDF threshold": "BR",
        "ES3D": "ES3D",
        "Instancewise RCPS": "Inst. RCPS",
        "LTT without hierarchy": "LTT",
        "F-DFHC without utility": "F-DFHC",
        "F-DFHC-U w/o projection": "F-DFHC-U $-\\Pi_H$",
        "F-DFHC-U full": "F-DFHC-U",
    }
    body = []
    for method in order:
        r = rows[rows["method"] == method].iloc[0]
        cplus = f3(r["close_positive_precision"])
        pcov = f3(r["positive_coverage"])
        evcov = f3(r["event_positive_coverage"])
        cneg = f3(r["close_negative_accuracy"])
        work = f3(r["workload_reduction"])
        risk = f3(r["empirical_all_risk"])
        cert = f3(r["certified_all_bound"])
        h = str(int(round(r["projected_hierarchy_violations"])))
        if method == "F-DFHC-U full":
            h = r"\textbf{0}"
            cert = rf"\textbf{{{cert}}}"
        body.append(f"{name_map[method]} & {cplus} & {cneg} & {pcov} & {evcov} & {work} & {risk} & {h} & {cert} \\\\")
    text = "\\begin{table}[t]\n\\centering\n\\caption{Selective closure utility on 2023--2024 tests.}\n\\label{tab:fdfhc_decision}\n\\papertablesize\n\\setlength{\\tabcolsep}{1.8pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\nMethod & C+ P & C- Acc. & P cov. & Event+ & Work & Risk & H viol. & Cert. \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize BR is binary relevance thresholding; ES3D is Evidence-Sufficiency Three-Way Decision; Inst. RCPS is instancewise Risk-Controlling Prediction Sets. C+ P is close-positive precision; C- Acc. is accuracy among close-negative decisions; P cov. is positive-label coverage; Event+ is event-level positive closure; Work is workload reduction; H viol. is hierarchy violations; Cert. is the certified all-closure risk bound. Good operating points have high C+ P and C- Acc., nonzero P cov., zero H viol., and empirical Risk below Cert. High-precision selective closure emphasizes precision and certified risk control under preliminary evidence. Bold values mark the projected hierarchy and risk-certificate properties of F-DFHC-U.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab7_dfhc_decision.tex", text)


def build_hierarchy_table() -> None:
    rows = summarize_fdfhc(MAIN_TEST_YEARS)
    order = ["BR-TFIDF threshold", "ES3D", "Instancewise RCPS", "LTT without hierarchy", "F-DFHC without utility", "F-DFHC-U w/o projection", "F-DFHC-U full"]
    name_map = {
        "BR-TFIDF threshold": "BR",
        "ES3D": "ES3D",
        "Instancewise RCPS": "Inst. RCPS",
        "LTT without hierarchy": "LTT",
        "F-DFHC without utility": "F-DFHC",
        "F-DFHC-U w/o projection": "F-DFHC-U $-\\Pi_H$",
        "F-DFHC-U full": "F-DFHC-U",
    }
    body = []
    for method in order:
        r = rows[rows["method"] == method].iloc[0]
        raw_h = int(round(r["raw_hierarchy_violations"]))
        proj_h = int(round(r["projected_hierarchy_violations"]))
        par_p = f3(r["parent_close_positive_precision"])
        ch_p = f3(r["child_close_positive_precision"])
        par_cov = f3(r["parent_positive_coverage"])
        ch_cov = f3(r["child_positive_coverage"])
        if method == "F-DFHC-U full":
            ch_p = rf"\textbf{{{ch_p}}}"
            proj_h_s = r"\textbf{0}"
        else:
            proj_h_s = str(proj_h)
        body.append(f"{name_map[method]} & {par_p} & {ch_p} & {par_cov} & {ch_cov} & {raw_h} & {proj_h_s} \\\\")
    text = "\\begin{table}[t]\n\\centering\n\\caption{Parent-child hierarchy consistency on 2023--2024 tests.}\n\\label{tab:hierarchy_projection}\n\\papertablesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & Par. C+ P & Ch. C+ P & Par. P cov. & Ch. P cov. & Raw H & Proj. H \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize ES3D is Evidence-Sufficiency Three-Way Decision; Inst. RCPS is instancewise Risk-Controlling Prediction Sets. Par. and Ch. denote parent and child nodes. C+ P is close-positive precision; P cov. is positive-label coverage. Raw H and Proj. H are hierarchy violations before and after projection.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab8_hierarchy_projection.tex", text)


def build_frontier_table() -> None:
    df = pd.read_csv(FDFHC / "f_dfhc_u_frontier_full.csv")
    df = df[(df["method"] == "F-DFHC-U") & (df["test_year"].isin(MAIN_TEST_YEARS))].copy()
    params = [
        "child_positive_alpha",
        "parent_positive_alpha",
        "personnel_parent_positive_alpha",
        "parent_positive_floor",
        "negative_alpha",
        "reliability_floor",
        "child_reliability_lcb_floor",
        "aircraft_child_floor",
        "child_margin",
    ]
    body = []
    for target in [0.80, 0.85, 0.90, 0.95]:
        feasible = df[df["close_positive_precision"] >= target]
        grouped = feasible.groupby(params, dropna=False)[
            ["positive_coverage", "close_positive_precision", "overall_coverage", "empirical_all_risk", "certified_all_bound"]
        ].mean()
        best = grouped.sort_values(["positive_coverage", "close_positive_precision"], ascending=False).iloc[0]
        body.append(
            f"{target:.2f} & {f3(best['close_positive_precision'])} & {f3(best['positive_coverage'])} & "
            f"{f3(best['overall_coverage'])} & {f3(best['empirical_all_risk'])} & {f3(best['certified_all_bound'])} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{F-DFHC-U precision--coverage operating points.}\n\\label{tab:frontier_points}\n\\papertablesize\n\\begin{tabular}{rrrrrr}\n\\toprule\nC+ P floor & C+ P & P cov. & Overall cov. & Risk & Cert. \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize C+ P is close-positive precision; P cov. is positive-label coverage; Cert. is the certified all-closure risk bound. For each close-positive precision floor, the row reports the feasible F-DFHC-U candidate with the largest positive-label coverage on the 2023--2024 folds.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab10_frontier_points.tex", text)


def build_budgeted_closure_table() -> None:
    budget = pd.read_csv(FDFHC / "f_dfhc_u_budgeted_closure_full.csv")
    budget = budget[(budget["method"] == "F-DFHC-U full") & (budget["test_year"].isin(MAIN_TEST_YEARS))].copy()
    order = ["top-1", "top-2", "all"]
    grouped = budget.groupby("positive_budget", as_index=False)[
        [
            "close_positive_precision",
            "positive_coverage",
            "event_positive_coverage",
            "true_positive_yield_per_event",
            "positive_actions_per_event",
            "net_utility_per_event",
        ]
    ].mean()
    grouped["positive_budget"] = pd.Categorical(grouped["positive_budget"], order, ordered=True)
    grouped = grouped.sort_values("positive_budget")
    body = []
    for _, r in grouped.iterrows():
        label = {"top-1": "Top-1", "top-2": "Top-2", "all": "All C+"}[str(r["positive_budget"])]
        body.append(
            f"{label} & {f3(r['close_positive_precision'])} & {f3(r['positive_coverage'])} & "
            f"{f3(r['event_positive_coverage'])} & {f3(r['true_positive_yield_per_event'])} & "
            f"{f3(r['positive_actions_per_event'])} & {f3(r['net_utility_per_event'])} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Budgeted F-DFHC-U positive closure on 2023--2024 tests.}\n\\label{tab:budgeted_closure}\n\\papertablesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrr}\n\\toprule\nBudget & C+ P & P cov. & Event+ & TP yield & C+ actions & Net utility \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize The budget caps close-positive findings per event. TP denotes true positive. TP yield, C+ actions, and net utility are per-event averages, with net utility computed as true positive closures minus false positive closures. Higher values are better for C+ P, P cov., Event+, TP yield, and net utility.\\end{flushleft}\n\\end{table}\n"
    write(ARTICLE / "tables" / "tab11_budgeted_closure.tex", text)


def build_supplementary_tables() -> None:
    recent = pd.read_csv(RECENT_MECH / "baseline_metrics_by_year_full.csv")
    recent = recent[recent["test_year"].isin(MAIN_TEST_YEARS)].copy()
    recent = (
        recent[recent["method"].isin(RECENT_METHODS)]
        .groupby(["stage", "method"], as_index=False)[["micro_f1", "macro_f1", "samples_f1", "jaccard_samples"]]
        .mean()
    )
    recent["stage"] = pd.Categorical(recent["stage"], STAGE_ORDER, ordered=True)
    recent["method"] = pd.Categorical(recent["method"], RECENT_METHODS, ordered=True)
    recent = recent.sort_values(["stage", "micro_f1"], ascending=[True, False])
    body = []
    for _, r in recent.iterrows():
        body.append(
            f"{STAGE_LABELS[r['stage']]} & {tex_escape(RECENT_DISPLAY[r['method']])} & {f3(r['micro_f1'])} & "
            f"{f3(r['macro_f1'])} & {f3(r['samples_f1'])} & {f3(r['jaccard_samples'])} \\\\"
        )
    text = "\\begin{longtable}{llrrrr}\n\\caption{Stage-view mechanism audit on 2023--2024 tests.}\\label{tab:s_complete_baselines}\\\\\n\\toprule\nStage & Mechanism & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nStage & Mechanism & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard \\\\\n\\midrule\n\\endhead\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{longtable}\n"
    write(SUPP / "tables" / "stab1_complete_baselines.tex", text)

    audit_rows = [
        ("BR-TFIDF", "standard implementation", "stage-specific evidence text", "level-2 labels", "validation-year F1 thresholds", "M"),
        ("Classifier Chains", "standard implementation", "same evidence text", "support-ordered label chain", "validation-year F1 thresholds", "M"),
        ("ML-KNN", "standard implementation", "reduced sparse text features", "level-2 labels", "validation-year F1 thresholds", "M"),
        ("RAkEL", "standard implementation", "same evidence text", "random label subsets", "validation-year F1 thresholds", "M"),
        ("H-gated BR", "structural extension", "same evidence text", "parent labels induced from children", "parent-gated thresholds", "M"),
        ("LE forest", "structural extension", "same evidence text", "label-enhanced feature-label correlations", "validation-year thresholds", "A"),
        ("Semantic-fusion BR", "semantic text baseline", "same evidence text", "frozen sentence-encoder and lexical features", "validation-year thresholds", "M"),
        ("VCLDL", "mechanism-aligned audit", "same stage views", "continuous label-distribution direction", "validation-year thresholds", "S"),
        ("HALB", "mechanism-aligned audit", "same stage views", "hierarchy-aware and label-balance direction", "validation-year thresholds", "S"),
        ("MatchXML", "mechanism-aligned audit", "same stage views", "text-label matching direction", "validation-year thresholds", "S"),
        ("LSPCL", "mechanism-aligned audit", "same stage views", "label-specific prototype direction", "validation-year thresholds", "S"),
        ("DyLas", "mechanism-aligned audit", "same stage views", "dynamic label-alignment direction", "validation-year thresholds", "S"),
        ("SIHTC", "mechanism-aligned audit", "same stage views", "structural smoothing direction", "validation-year thresholds", "S"),
    ]
    body = []
    for method, role, evidence, labels_used, thresholding, use in audit_rows:
        body.append(
            f"{tex_escape(method)} & {tex_escape(role)} & {tex_escape(evidence)} & "
            f"{tex_escape(labels_used)} & {tex_escape(thresholding)} & {tex_escape(use)} \\\\"
        )
    text = "\\begin{longtable}{p{0.14\\linewidth}p{0.16\\linewidth}p{0.17\\linewidth}p{0.19\\linewidth}p{0.15\\linewidth}p{0.05\\linewidth}}\n\\caption{Baseline and mechanism audit protocol.}\\label{tab:s_baseline_audit}\\\\\n\\toprule\nMethod & Role & Evidence input & Label use & Thresholding & Use \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nMethod & Role & Evidence input & Label use & Thresholding & Use \\\\\n\\midrule\n\\endhead\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{longtable}\n\\begin{flushleft}\\footnotesize Use codes: M indicates a main-table comparison, A indicates a main audit entry, and S indicates a supplementary stage-pattern audit.\\end{flushleft}\n"
    write(SUPP / "tables" / "stab9_baseline_audit.tex", text)

    ci = pd.read_csv(RESULTS / "exact_baseline_ci_rolling_2023_2025_faithful.csv")
    ci = ci.sort_values(["stage", "test_year", "micro_f1"], ascending=[True, True, False])
    body = []
    for _, r in ci.iterrows():
        body.append(
            f"{int(r['test_year'])} & {tex_escape(r['stage'])} & {tex_escape(r['method'])} & {f3(r['micro_f1'])} & [{f3(r['micro_f1_ci_low'])}, {f3(r['micro_f1_ci_high'])}] & {f3(r['paired_bootstrap_p'])} \\\\"
        )
    text = "\\begin{longtable}{rllrrr}\n\\caption{Standard baseline year-level intervals and paired tests.}\\\\\n\\toprule\nYear & Stage & Method & Micro-F1 & 95\\% CI & Paired p \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nYear & Stage & Method & Micro-F1 & 95\\% CI & Paired p \\\\\n\\midrule\n\\endhead\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{longtable}\n"
    write(SUPP / "tables" / "stab10_baseline_ci.tex", text)

    ext = read_csv(EXTERNAL / "external_source_feasibility.csv")
    faa = read_csv(EXTERNAL / "faa_aids_file_summary.csv")
    faa_field = read_csv(EXTERNAL / "faa_external_field_audit.csv")
    text_audit = read_csv(EXTERNAL / "external_text_distribution_audit.csv")
    body = []
    for r in ext:
        paper_use = r["paper_use"]
        if paper_use.endswith(" source only"):
            paper_use = paper_use[: -len(" source only")]
        if paper_use == "supplementary source-feasibility and matching audit":
            paper_use = "source-feasibility audit"
        body.append(f"{tex_escape(r['source'])} & {tex_escape(r['role'])} & {tex_escape(r['downloadable'])} & {tex_escape(r['local_check'])} & {tex_escape(paper_use)} \\\\")
    text = "\\begin{table}[t]\n\\centering\n\\caption{External source feasibility.}\n\\begin{tabularx}{\\linewidth}{lYYYY}\n\\toprule\nSource & Role & Access & Source audit & Use \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    body2 = [f"{tex_escape(r['filename'])} & {int(r['rows']):,} & {tex_escape(r['year_min'])}--{tex_escape(r['year_max'])} & {tex_escape(r['columns'])} \\\\" for r in faa]
    text += "\n\\begin{table}[t]\n\\centering\n\\caption{FAA ASIAS AIDS file inspection.}\n\\begin{tabular}{lrrr}\n\\toprule\nFile & Rows & Years & Columns \\\\\n\\midrule\n"
    text += "\n".join(body2)
    text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    body3 = []
    for r in faa_field:
        years = f"{tex_escape(r['year_min'])}--{tex_escape(r['year_max'])}"
        units = f"{int(r['records']):,} records"
        fields = f"{int(r['columns'])} fields"
        body3.append(
            f"{tex_escape(r['archive'])} & {units} & {fields} & {years} & "
            f"stage-field group coverage & {f3(r['mean_common_group_coverage'])} \\\\"
        )
        body3.append(
            f"{tex_escape(r['archive'])} & {units} & {fields} & {years} & "
            f"narrative coverage & {f3(r['narrative_coverage'])} \\\\"
        )
    text += "\n\\begin{table}[t]\n\\centering\n\\caption{FAA ASIAS AIDS external field audit.}\n\\label{tab:s_external_field_audit}\n\\begin{tabularx}{\\linewidth}{lYYYYY}\n\\toprule\nSource file & Units & Fields & Years & Measured quantity & Result \\\\\n\\midrule\n"
    text += "\n".join(body3)
    text += "\n\\bottomrule\n\\end{tabularx}\n\\begin{flushleft}\\footnotesize Stage-field group coverage averages the availability of event date, location, aircraft identity, injury and damage, phase or operation, and narrative groups that correspond to early evidence fields in the NTSB benchmark.\\end{flushleft}\n\\end{table}\n"
    body4 = []
    for r in text_audit:
        result = r["result"]
        if "records" in str(r["measured_quantity"]):
            result_s = f"{int(float(result)):,}"
        else:
            try:
                result_s = f3(float(result))
            except ValueError:
                result_s = tex_escape(result)
        body4.append(
            f"{tex_escape(r['source'])} & {tex_escape(r['units'])} & {tex_escape(r['years'])} & "
            f"{tex_escape(r['measured_quantity'])} & {result_s} & {tex_escape(r['reference'])} \\\\"
        )
    text += "\n\\begin{table}[t]\n\\centering\n\\caption{External narrative and score-distribution audit.}\n\\label{tab:s_external_text_distribution}\n\\begin{tabularx}{\\linewidth}{lrlYYY}\n\\toprule\nSource & Units & Years & Measured quantity & Result & Reference \\\\\n\\midrule\n"
    text += "\n".join(body4)
    text += "\n\\bottomrule\n\\end{tabularx}\n\\begin{flushleft}\\footnotesize The audit compares FAA AIDS narratives with NTSB preliminary evidence text. Score quantities are computed by applying the NTSB preliminary-stage scoring model to FAA AIDS narratives without using final finding labels.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab11_external_sources.tex", text)

    manifest = read_csv(QUALITY / "source_hash_manifest.csv")
    artifact_names = [
        ("NTSB downloaded archive", "avall.zip"),
        ("NTSB extracted database", "avall.mdb"),
        ("Final finding benchmark rows", "finding_rows_2008_2025.csv"),
        ("Event-aircraft benchmark rows", "case_aircraft_rows_2008_2025.csv"),
    ]
    body = []
    for label, suffix in artifact_names:
        found = next((r for r in manifest if str(r["path"]).endswith(suffix)), None)
        if not found:
            continue
        size_mb = int(found["bytes"]) / (1024 * 1024)
        body.append(f"{label} & {size_mb:.1f} & {tex_escape(found['sha256'][:16])} \\\\")
    text = "\\begin{table}[t]\n\\centering\n\\caption{Source traceability manifest.}\n\\label{tab:s_source_trace}\n\\begin{tabular}{lrl}\n\\toprule\nArtifact & Size (MB) & SHA-256 prefix \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize The manifest is generated after data acquisition and processing. The hash prefix identifies the public archive, extracted database, and derived benchmark files used to reproduce the tables.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab12_source_trace.tex", text)

    metrics = pd.read_csv(FDFHC / "f_dfhc_u_metrics_full.csv")
    full = metrics[(metrics["method"] == "F-DFHC-U full") & (metrics["test_year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR]))].copy()
    close_negative = full["closed_decisions"] - full["close_positive_decisions"]
    false_positive = full["close_positive_decisions"] - full["true_positive_closed"]
    false_negative = full["wrong_closed_decisions"] - false_positive
    full["close_negative_accuracy"] = (close_negative - false_negative) / close_negative
    full = full.sort_values("test_year")
    main_mean = full[full["test_year"].isin(MAIN_TEST_YEARS)][
        [
            "close_positive_precision",
            "close_negative_accuracy",
            "positive_coverage",
            "overall_coverage",
            "empirical_all_risk",
            "certified_all_bound",
            "projected_hierarchy_violations",
            "late_premature_closure_risk",
        ]
    ].mean()
    body = []
    for _, r in full.iterrows():
        role = "main" if int(r["test_year"]) in MAIN_TEST_YEARS else "freshness"
        body.append(
            f"{int(r['test_year'])} & {role} & {f3(r['close_positive_precision'])} & {f3(r['close_negative_accuracy'])} & {f3(r['positive_coverage'])} & "
            f"{f3(r['overall_coverage'])} & {f3(r['empirical_all_risk'])} & {f3(r['certified_all_bound'])} & "
            f"{int(round(r['projected_hierarchy_violations']))} & {f4(r['late_premature_closure_risk'])} \\\\"
        )
    body.append(
        f"2023--2024 & main mean & \\textbf{{{f3(main_mean['close_positive_precision'])}}} & "
        f"\\textbf{{{f3(main_mean['close_negative_accuracy'])}}} & \\textbf{{{f3(main_mean['positive_coverage'])}}} & \\textbf{{{f3(main_mean['overall_coverage'])}}} & "
        f"\\textbf{{{f3(main_mean['empirical_all_risk'])}}} & \\textbf{{{f3(main_mean['certified_all_bound'])}}} & "
        f"\\textbf{{{int(round(main_mean['projected_hierarchy_violations']))}}} & \\textbf{{{f4(main_mean['late_premature_closure_risk'])}}} \\\\"
    )
    text = "\\begin{table}[t]\n\\centering\n\\caption{F-DFHC-U yearly closure metrics.}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llrrrrrrrr}\n\\toprule\nTest year & Role & C+ P & C- Acc. & P cov. & Cov. & Risk & Cert. & H viol. & Late \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize C+ P is close-positive precision; C- Acc. is accuracy among close-negative decisions; P cov. is positive-label coverage; Cov. is overall closure coverage; Cert. is the all-risk certificate; Late is late-emerging premature closure risk. Bold values summarize the 2023--2024 main folds; 2025 is a freshness audit.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab5_fdfhc_metrics.tex", text)

    forward = pd.read_csv(FDFHC / "f_dfhc_u_forward_certificate_full.csv")
    if "method" in forward.columns:
        forward = forward[forward["method"] == "F-DFHC-U full"].copy()
    forward = forward[forward["test_year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR])].copy()
    forward = forward.sort_values("test_year")
    body = []
    for _, r in forward.iterrows():
        role = "main" if int(r["test_year"]) in MAIN_TEST_YEARS else "freshness"
        valid_col = "certificate_valid" if "certificate_valid" in r.index else "bound_valid"
        risk_col = "empirical_all_risk" if "empirical_all_risk" in r.index else "empirical_risk"
        cert_col = "certified_all_bound" if "certified_all_bound" in r.index else "certified_bound"
        valid = "Yes" if bool(r[valid_col]) else "No"
        body.append(
            f"{int(r['test_year'])} & {role} & {int(r['calibration_year'])} & {f3(r[risk_col])} & "
            f"{f3(r[cert_col])} & {valid} & {f3(r['close_positive_precision'])} & {f3(r['positive_coverage'])} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Forward-year F-DFHC-U certificates.}\n\\begin{tabular}{llrrrrrr}\n\\toprule\nTest year & Role & Calib. year & Emp. risk & Cert. & Valid & C+ P & P cov. \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Calib. year is the calibration year; Emp. risk is empirical all-closure risk; Cert. is the certified all-risk bound. Empirical all-closure risk remains below the calibration-year certificate in every prospective split.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab6_fdfhc_u_forward.tex", text)

    drift = pd.read_csv(FDFHC / "f_dfhc_u_exchangeability_diagnostics_full.csv")
    drift = drift[drift["test_year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR])].sort_values("test_year")
    forward_cert = pd.read_csv(FDFHC / "f_dfhc_u_forward_certificate_full.csv")
    forward_cert = forward_cert[forward_cert["test_year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR])]
    valid_by_year = {
        int(r["test_year"]): bool(r["bound_valid"]) if "bound_valid" in r.index else bool(r.get("certificate_valid", False))
        for _, r in forward_cert.iterrows()
    }
    body = []
    for _, r in drift.iterrows():
        role = "main" if int(r["test_year"]) in MAIN_TEST_YEARS else "freshness"
        valid = "Yes" if valid_by_year.get(int(r["test_year"]), False) else "No"
        body.append(
            f"{int(r['test_year'])} & {role} & {int(r['calibration_year'])} & "
            f"{f3(r['label_prevalence_l1'])} & {f3(r['score_mean_l1'])} & {f3(r['score_ks_mean'])} & "
            f"{f3(r['risk_drift'])} & {f3(r['coverage_drift'])} & {f3(r['positive_coverage_drift'])} & {valid} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{Calibration-to-test drift diagnostics for F-DFHC-U.}\n\\setlength{\\tabcolsep}{1.5pt}\n\\begin{tabular}{llrrrrrrrl}\n\\toprule\nYear & Role & Calib. & Prev. & Score & KS & Risk & Cov. & P-cov. & Risk $\\leq$ Cert. \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Prev. is mean absolute label-prevalence shift; Score is mean absolute score-mean shift; KS is the mean Kolmogorov--Smirnov score statistic across labels. Risk, Cov., and P-cov. are test-year values minus calibration-year values under the selected closure rule. Risk $\\leq$ Cert. records whether empirical all-closure risk remains below the yearly certificate.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab13_exchangeability_diagnostics.tex", text)

    projection = pd.read_csv(FDFHC / "f_dfhc_u_projection_stats_full.csv")
    projection = projection[(projection["method"] == "F-DFHC-U full") & (projection["test_year"].isin(MAIN_TEST_YEARS + [FRESHNESS_YEAR]))].copy()
    projection = projection.sort_values("test_year")
    body = []
    for _, r in projection.iterrows():
        role = "main" if int(r["test_year"]) in MAIN_TEST_YEARS else "freshness"
        body.append(
            f"{int(r['test_year'])} & {role} & {int(r['raw_hierarchy_violations'])} & {int(r['projected_hierarchy_violations'])} & "
            f"{int(r['projection_action_changes'])} & {f4(r['projection_change_rate'])} \\\\"
        )
    text = "\\begin{table}[t]\n\\centering\n\\caption{F-DFHC-U projection change statistics.}\n\\begin{tabular}{llrrrr}\n\\toprule\nTest year & Role & Raw H & Proj. H & Action changes & Change rate \\\\\n\\midrule\n"
    text += "\n".join(body)
    text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Raw H denotes hierarchy violations before minimum-disturbance projection; Proj. H denotes hierarchy violations after projection.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab7_fdfhc_u_projection.tex", text)

    parent = (
        pd.read_csv(FDFHC / "f_dfhc_u_label_details_full.csv")
        if (FDFHC / "f_dfhc_u_label_details_full.csv").exists()
        else pd.DataFrame()
    )
    if parent.empty or not {"method", "test_year", "node_type", "label", "support", "close_positive_decisions", "close_positive_precision", "positive_coverage"}.issubset(parent.columns):
        parent_text = (SUPP / "tables" / "stab8_parent_closure.tex").read_text(encoding="utf-8")
    else:
        parent = parent[
            (parent["method"] == "F-DFHC-U full")
            & (parent["test_year"].isin(MAIN_TEST_YEARS))
            & (parent["node_type"] == "parent")
        ].copy()
        parent["true_positive_closed"] = parent["close_positive_precision"] * parent["close_positive_decisions"]
        grouped = parent.groupby("label", as_index=False)[["support", "close_positive_decisions", "true_positive_closed"]].sum()
        grouped = grouped.sort_values("support", ascending=False)
        body = []
        for _, r in grouped.iterrows():
            precision = float("nan")
            if float(r["close_positive_decisions"]) > 0:
                precision = float(r["true_positive_closed"]) / float(r["close_positive_decisions"])
            coverage = 0.0
            if float(r["support"]) > 0:
                coverage = float(r["true_positive_closed"]) / float(r["support"])
            cplus = f3(precision)
            pcov = f3(coverage)
            if str(r["label"]) == "Aircraft":
                cplus = rf"\textbf{{{cplus}}}"
                pcov = rf"\textbf{{{pcov}}}"
            elif str(r["label"]) == "Personnel issues":
                cplus = rf"\textbf{{{cplus}}}"
            body.append(
                f"{tex_escape(r['label'])} & {int(r['support'])} & {int(r['close_positive_decisions'])} & "
                f"{int(round(r['true_positive_closed']))} & {cplus} & {pcov} \\\\"
            )
        parent_text = "\\begin{table}[t]\n\\centering\n\\caption{Parent-node close-positive results of F-DFHC-U.}\n\\begin{tabular}{lrrrrr}\n\\toprule\nParent node & Support & C+ decisions & True C+ & C+ P & P cov. \\\\\n\\midrule\n"
        parent_text += "\n".join(body)
        parent_text += "\n\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\footnotesize Support, C+ decisions, and True C+ are summed over the 2023--2024 main forward folds. C+ P is close-positive precision; P cov. is positive-label coverage. Dashes indicate no close-positive decisions for that parent node.\\end{flushleft}\n\\end{table}\n"
    write(SUPP / "tables" / "stab8_parent_closure.tex", parent_text)


def main() -> None:
    build_data_summary()
    build_data_completeness_table()
    build_stage_metrics_table()
    build_faithful_baseline_table()
    build_recent_mechanism_table()
    build_benchmark_robustness_table()
    build_fdfhc_event_table()
    build_hierarchy_table()
    build_frontier_table()
    build_budgeted_closure_table()
    build_supplementary_tables()


if __name__ == "__main__":
    main()
