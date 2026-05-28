from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "experiments" / "f_dfhc_u_closure" / "f_dfhc_u_frontier_full.csv"
OUT = ROOT / "article" / "elsarticle_manuscript" / "figures" / "fig5_risk_utility_frontier.pdf"
MAIN_TEST_YEARS = [2023, 2024]


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["test_year"].isin(MAIN_TEST_YEARS)].copy()
    methods = ["BR-TFIDF threshold", "ES3D", "LTT without hierarchy", "F-DFHC-U"]
    fig, ax = plt.subplots(figsize=(6.4, 4.15))
    styles = {
        "BR-TFIDF threshold": {"marker": "x", "linestyle": "None", "label": "BR-TFIDF (violating)"},
        "ES3D": {"marker": "^", "linestyle": "None", "label": "ES3D (violating)"},
        "LTT without hierarchy": {
            "marker": "x",
            "linestyle": "--",
            "label": "LTT (violating)",
            "linewidth": 0.65,
            "markersize": 2.0,
            "alpha": 0.45,
        },
        "F-DFHC-U": {
            "marker": "o",
            "linestyle": "-",
            "label": "F-DFHC-U (feasible)",
            "linewidth": 0.80,
            "markersize": 2.2,
            "alpha": 0.70,
        },
    }
    for method in methods:
        part = df[df["method"] == method].copy()
        if part.empty:
            continue
        part = (
            part.groupby("positive_coverage", as_index=False)
            .agg(close_positive_precision=("close_positive_precision", "mean"))
            .sort_values("positive_coverage")
        )
        style = styles[method]
        if method in {"BR-TFIDF threshold", "ES3D"}:
            ax.scatter(
                part["positive_coverage"],
                part["close_positive_precision"],
                s=42,
                marker=style["marker"],
                label=style["label"],
            )
        else:
            ax.plot(
                part["positive_coverage"],
                part["close_positive_precision"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                markeredgewidth=0.65,
                label=style["label"],
            )
    ax.axhline(0.90, color="0.40", linestyle="--", linewidth=0.75)
    ax.axhline(0.85, color="0.68", linestyle=":", linewidth=0.65)
    ax.axhline(0.95, color="0.68", linestyle=":", linewidth=0.65)
    ax.text(
        0.835,
        0.918,
        "0.90 precision floor",
        fontsize=8,
        ha="right",
        va="bottom",
        color="0.25",
    )
    ax.set_xlabel("Positive coverage")
    ax.set_ylabel("Close-positive precision")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        columnspacing=1.5,
        handlelength=2.0,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
