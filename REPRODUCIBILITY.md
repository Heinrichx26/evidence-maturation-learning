# Reproducibility notes

This project keeps paper assets in four directories:

- `data`: raw downloads, processed NTSB records, and data-source manifests.
- `src`: data extraction, data-quality audit, experiments, closure calibration, table builders, and plotting.
- `results`: experiment outputs, external-source checks, logs, summaries, and source hashes.
- `article`: LaTeX manuscript, supplementary material, figures, and tables.

Install the Python dependencies with:

```powershell
pip install -r .\requirements.txt
```

## Data-quality audit

Run the audit before any experiment:

```powershell
python .\src\data\audit_ntsb_completeness.py
```

The audit writes yearly evidence-completeness rates to `results\data_quality`. The 2023--2024 folds are used as the main forward tests. The 2025 fold is retained as a freshness audit because the public export has sparse preliminary narrative coverage for that year.

## Smoke tests

Run the lightweight checks first:

```powershell
.\scripts\run_smoke_tests.ps1
```

The script runs the data audit, one stage-aligned baseline smoke test, and a one-year F-DFHC-U smoke test.

## Full experiments

Run the full experiment set with:

```powershell
.\scripts\run_full_experiments.ps1
```

The full run covers stage-aligned standard baselines, rolling robustness, fixed historical stress testing, and calibrated hierarchical closure. Main manuscript claims use the 2023--2024 forward folds. The 2025 outputs support the freshness audit and supplementary checks.

## Manuscript tables and compilation

Regenerate tables, the F-DFHC-U frontier figure, and PDFs with:

```powershell
.\scripts\build_article.ps1
```

The script reads saved results and does not rerun the full experiments. Conceptual workflow figures are maintained as PowerPoint files and exported as PDF. Data figures read saved CSV outputs from `results`.

## Review package

For anonymous review, include:

- the manuscript PDF and supplementary PDF;
- `REPRODUCIBILITY.md`, `requirements.txt`, and `scripts`;
- `src` excluding cache folders;
- `results\data_quality`, selected `results\experiments` CSV summaries, and generated tables;
- public-source provenance and hashes from `results\data_quality\source_hash_manifest.csv`.

Processed analysis tables and code are available in the public repository at https://github.com/Heinrichx26/evidence-maturation-learning.
