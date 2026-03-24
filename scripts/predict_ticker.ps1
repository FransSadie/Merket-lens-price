param(
    [Parameter(Mandatory = $true)]
    [string]$Ticker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .\.venv. Create it first: python -m venv .venv"
}

@"
import json
from app.models.inference import predict_for_ticker

print(json.dumps(predict_for_ticker("$Ticker"), indent=2))
"@ | .\.venv\Scripts\python.exe -

