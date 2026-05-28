from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SUPP = ROOT / "article" / "supplementary"
TABLE_DIR = SUPP / "tables"


STAGE_LABELS = {
    "event_aircraft_core": "Core",
    "event_aircraft_weather": "Core + weather",
    "preliminary": "Preliminary",
    "mature_factual": "Mature factual",
}


def tex_escape(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tex_breakable_terms(text: str) -> str:
    text = tex_escape(text)
    text = text.replace(r"\_", r"\_{}\allowbreak ")
    text = text.replace(", ", r",\allowbreak ")
    text = text.replace("-", r"-\allowbreak ")
    text = text.replace("/", r"/\allowbreak ")
    return text


def fmt(value: object) -> str:
    return f"{float(value):.3f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def strong_baselines() -> None:
    df = pd.read_csv(RESULTS / "tables" / "tab_strong_baseline_comparison.csv")
    order = ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]
    df["stage"] = pd.Categorical(df["stage"], order, ordered=True)
    df = df.sort_values(["stage", "micro_f1"], ascending=[True, False])
    lines = [
        r"\begin{longtable}{llrrrr}",
        r"\caption{Complete strong baseline comparison.}\label{tab:s_complete_baselines}\\",
        r"\toprule",
        r"Stage & Method & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Stage & Method & Micro-F1 & Macro-F1 & Samples-F1 & Jaccard \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{STAGE_LABELS[str(row['stage'])]} & {tex_escape(str(row['method']))} & {fmt(row['micro_f1'])} & "
            f"{fmt(row['macro_f1'])} & {fmt(row['samples_f1'])} & {fmt(row['jaccard_samples'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write(TABLE_DIR / "stab1_complete_strong_baselines.tex", "\n".join(lines))


def yearly_stage_metrics() -> None:
    df = pd.read_csv(RESULTS / "tables" / "tab_evidence_stage_metrics_by_year.csv")
    order = ["event_aircraft_core", "event_aircraft_weather", "preliminary", "mature_factual"]
    df["stage"] = pd.Categorical(df["stage"], order, ordered=True)
    df = df.sort_values(["test_year", "stage"])
    lines = [
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Year-by-year evidence-stage metrics.}\label{tab:s_year_stage}\\",
        r"\toprule",
        r"Year & Stage & $n$ & Micro-F1 & Macro-F1 & Samples-F1 & Entropy \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Year & Stage & $n$ & Micro-F1 & Macro-F1 & Samples-F1 & Entropy \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['test_year'])} & {STAGE_LABELS[str(row['stage'])]} & {int(row['n_samples'])} & "
            f"{fmt(row['micro_f1'])} & {fmt(row['macro_f1'])} & {fmt(row['samples_f1'])} & "
            f"{fmt(row['mean_binary_entropy'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write(TABLE_DIR / "stab2_yearly_stage_metrics.tex", "\n".join(lines))


def label_taxonomy() -> None:
    df = pd.read_csv(RESULTS / "tables" / "tab_label_maturation_taxonomy.csv")
    df = df.sort_values(["maturation_class", "support"], ascending=[True, False])
    lines = [
        r"\begin{longtable}{@{}p{0.34\linewidth}rrrrp{0.15\linewidth}@{}}",
        r"\caption{Complete label-level maturation taxonomy.}\label{tab:s_label_taxonomy}\\",
        r"\toprule",
        r"Label & Support & Prelim. F1 & Mature F1 & Gain & Class \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Label & Support & Prelim. F1 & Mature F1 & Gain & Class \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{tex_breakable_terms(row['label'])} & {int(row['support'])} & {fmt(row['preliminary_f1'])} & "
            f"{fmt(row['mature_factual_f1'])} & {fmt(row['mature_minus_prelim_f1'])} & "
            f"{tex_breakable_terms(row['maturation_class'].replace('_', ' '))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write(TABLE_DIR / "stab3_full_label_taxonomy.tex", "\n".join(lines))


def top_terms() -> None:
    df = pd.read_csv(RESULTS / "experiments" / "evidence_maturation_learning" / "interpretable_top_terms_full.csv")
    selected_labels = [
        "Personnel issues-Task performance",
        "Aircraft-Aircraft oper/perf/capability",
        "Aircraft-Fluids/misc hardware",
        "Environmental issues-Conditions/weather/phenomena",
        "Aircraft-Aircraft systems",
        "Environmental issues-Physical environment",
    ]
    selected = df[(df["stage"].isin(["preliminary", "mature_factual"])) & (df["label"].isin(selected_labels)) & (df["rank"] <= 3)]
    selected = selected.sort_values(["stage", "label", "rank"])
    grouped = (
        selected.groupby(["stage", "label"], observed=True)["term"]
        .apply(lambda terms: ", ".join(dict.fromkeys(str(t) for t in terms)))
        .reset_index()
    )
    lines = [
        r"\begin{longtable}{@{}p{0.16\linewidth}p{0.28\linewidth}p{0.48\linewidth}@{}}",
        r"\caption{Representative leakage-safe high-weight terms.}\label{tab:s_top_terms}\\",
        r"\toprule",
        r"Stage & Label & Terms \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Stage & Label & Terms \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"{STAGE_LABELS[str(row['stage'])]} & {tex_breakable_terms(row['label'])} & {tex_breakable_terms(row['term'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write(TABLE_DIR / "stab4_interpretable_terms.tex", "\n".join(lines))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    strong_baselines()
    yearly_stage_metrics()
    label_taxonomy()
    top_terms()


if __name__ == "__main__":
    main()
