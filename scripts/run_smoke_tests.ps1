Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
  python .\src\data\audit_ntsb_completeness.py
  python .\src\experiments\benchmark_extensions.py --smoke --suffix smoke_is_revision_2023 --test-years 2023 --splits rolling --stages preliminary --max-labels 12 --max-features 12000 --svd-dims 48 --rakel-subsets 8 --bootstrap 80 --jobs 4
  python .\src\experiments\f_dfhc_u_closure.py --test-years 2023 --max-features 12000 --jobs 4
}
finally {
  Pop-Location
}
