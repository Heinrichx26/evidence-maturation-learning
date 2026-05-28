# Evidence Maturation Learning

This repository contains code and derived results for the study on preliminary-to-final evidence maturation learning in aviation investigation records.

The repository is organized for reproducibility:

- `src/`: data assembly, data-quality audit, external source audit, stage-aligned learning, calibrated hierarchical closure, and plotting code.
- `scripts/`: commands for running the main experiment suite and preliminary checks.
- `results/`: derived result tables used to report evidence-stage performance, calibrated closure, robustness, and external source audits.
- `data/`: source descriptions and reconstruction notes for public aviation data.

The manuscript source, supplementary manuscript, submission forms, and generated article PDFs are excluded. Record-level narrative tables and raw agency archives are also excluded from this public code repository. The data notes identify the public sources and the scripts used to reconstruct the analytical tables.

## Installation

```powershell
pip install -r requirements.txt
```

## Running the analyses

Run the preliminary checks first:

```powershell
.\scripts\run_smoke_tests.ps1
```

Run the main experiment suite:

```powershell
.\scripts\run_full_experiments.ps1
```

The scripts write outputs under `results/`. Existing CSV, JSON, and Markdown files under `results/` record the derived outputs used for the manuscript tables and figures.

## Data sources

The benchmark is reconstructed from public aviation safety sources:

- National Transportation Safety Board (NTSB) Case Analysis and Reporting Online (CAROL): https://data.ntsb.gov/carol-main-public/basic-search
- NTSB Aviation Accident Database and Synopses: https://www.ntsb.gov/Pages/AviationQuery.aspx
- Federal Aviation Administration (FAA) Accident and Incident Data System (AIDS): https://www.asias.faa.gov/apex/f?p=100:189:::NO
- National Aeronautics and Space Administration (NASA) Aviation Safety Reporting System (ASRS) Database Online: https://asrs.arc.nasa.gov/search/database.html

The closed-label benchmark uses NTSB investigation records because they link preliminary evidence, mature factual evidence, and final findings. FAA AIDS and NASA ASRS are used for external source auditing.

## Result files

The main result files include:

- `results/tables/tab_evidence_stage_metrics.csv`
- `results/tables/tab_strong_baseline_comparison.csv`
- `results/tables/tab_label_maturation_taxonomy.csv`
- `results/experiments/f_dfhc_u_closure/f_dfhc_u_metrics_full.csv`
- `results/experiments/f_dfhc_u_closure/f_dfhc_u_frontier_full.csv`
- `results/external_robustness/external_text_distribution_audit.csv`
- `results/data_quality/ntsb_stage_completeness_by_year.csv`

These files contain aggregate metrics and derived summaries. They do not contain the manuscript text.

