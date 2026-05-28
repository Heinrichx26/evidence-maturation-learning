Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
  python .\src\data\audit_ntsb_completeness.py
  python .\src\experiments\benchmark_extensions.py --suffix rolling_2023_2025_faithful --test-years 2023-2025 --splits rolling --stages preliminary,mature_factual --methods "BR-TFIDF,Classifier Chains,ML-KNN,RAkEL,Hierarchy-gated BR,Label-enhanced forest" --max-labels 16 --max-features 18000 --svd-dims 64 --rakel-subsets 12 --bootstrap 120 --jobs 4
  python .\src\experiments\benchmark_extensions.py --suffix rolling_2015_2025_core --test-years 2015-2025 --splits rolling --stages preliminary,mature_factual --methods "BR-TFIDF,ML-KNN,Hierarchy-gated BR" --max-labels 16 --max-features 18000 --svd-dims 64 --bootstrap 60 --jobs 4
  python .\src\experiments\benchmark_extensions.py --suffix fixed_hist_2020_2025_prelim --test-years 2020-2025 --splits fixed_historical --stages preliminary --max-labels 16 --max-features 18000 --svd-dims 64 --rakel-subsets 12 --bootstrap 120 --jobs 4
  python .\src\experiments\f_dfhc_u_closure.py --full --test-years 2023-2025 --jobs 4 --alpha-all 0.075
}
finally {
  Pop-Location
}
