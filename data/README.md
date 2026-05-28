# Data reconstruction notes

The analysis uses public aviation safety records. Raw source archives and record-level narrative tables are not stored in this repository. They can be reconstructed from the public sources listed below with the scripts under `src/data`.

## Public sources

1. National Transportation Safety Board (NTSB) Case Analysis and Reporting Online (CAROL)  
   https://data.ntsb.gov/carol-main-public/basic-search

2. NTSB Aviation Accident Database and Synopses  
   https://www.ntsb.gov/Pages/AviationQuery.aspx

3. Federal Aviation Administration (FAA) Accident and Incident Data System (AIDS)  
   https://www.asias.faa.gov/apex/f?p=100:189:::NO

4. National Aeronautics and Space Administration (NASA) Aviation Safety Reporting System (ASRS) Database Online  
   https://asrs.arc.nasa.gov/search/database.html

## Reconstruction workflow

- `src/data/export_ntsb_evidence_maturation.py` exports NTSB event-aircraft and finding records.
- `src/data/merge_ntsb_year_range.py` builds historical year-range tables.
- `src/data/audit_ntsb_completeness.py` computes yearly evidence-completeness summaries.
- `src/data/download_faa_aids_feasibility.py`, `src/data/audit_faa_external_fields.py`, and `src/data/audit_external_text_distribution.py` support the external FAA source audit.

The files under `results/data_quality` and `results/external_robustness` provide aggregate source checks and audit outputs.

