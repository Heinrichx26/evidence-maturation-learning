# Strong baseline result assessment

## Verification status

All six recent baseline sources were verified through DOI metadata and written to `results/references/verified_strong_baseline_references.csv`.

## Main usable comparisons

The strongest CPU-feasible baselines are VCLDL-adapted, SIHTC-adapted, and DyLas-adapted. They improve over BR-TFIDF on micro-F1 in the preliminary and mature factual stages while preserving the same forward-year validation protocol.

| Stage | Best method | Best micro-F1 | BR-TFIDF micro-F1 | Delta |
|---|---:|---:|---:|---:|
| Event-aircraft core | VCLDL-adapted | 0.423 | 0.405 | 0.018 |
| Event-aircraft plus weather | SIHTC-adapted | 0.431 | 0.406 | 0.025 |
| Preliminary evidence | VCLDL-adapted | 0.544 | 0.534 | 0.010 |
| Mature factual evidence | SIHTC-adapted | 0.681 | 0.675 | 0.006 |

## Paper-use judgment

The strong-baseline results are usable for the paper. The main conclusion remains stable under recent label-distribution and hierarchy-aware baselines: evidence-stage maturation produces a larger performance gap than the choice among lightweight strong baselines. The best preliminary method reaches 0.544 micro-F1, and the best mature factual method reaches 0.681 micro-F1. This keeps the core evidence maturation gap at about 0.137 micro-F1 after strong-baseline comparison.

MatchXML-adapted and LSPCL-adapted are weaker in this dataset. Their pure label-profile signal is limited because the NTSB level-2 taxonomy labels are short and broad. They can remain in the complete result table or appendix-level robustness table, while the main table should focus on BR-TFIDF, VCLDL-adapted, SIHTC-adapted, DyLas-adapted, and HALB-adapted.

## Suggested manuscript handling

Use `tab_strong_baseline_preliminary_mature.csv` as the compact main baseline table. Use `tab_best_strong_baseline_by_stage.csv` to support the statement that the staged evidence effect remains visible under strong baselines. Keep `tab_strong_baseline_comparison.csv` as the complete result table for supplementary material.
