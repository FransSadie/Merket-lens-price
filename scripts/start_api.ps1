Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .\.venv. Create it first: python -m venv .venv"
}

& .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

