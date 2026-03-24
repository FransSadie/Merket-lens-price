Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .\.venv. Create it first: python -m venv .venv"
}

@'
import json
from app.data_quality.checks import data_status_snapshot, run_data_quality_checks

print("DATA STATUS")
print(json.dumps(data_status_snapshot(), indent=2))
print("")
print("DATA QUALITY")
print(json.dumps(run_data_quality_checks(), indent=2))
'@ | .\.venv\Scripts\python.exe -

