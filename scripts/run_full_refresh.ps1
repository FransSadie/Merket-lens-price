Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .\.venv. Create it first: python -m venv .venv"
}

& .\.venv\Scripts\python.exe -m app.ingestion.run_once
& .\.venv\Scripts\python.exe -m app.features.run_once
& .\.venv\Scripts\python.exe -m app.models.train_baseline

