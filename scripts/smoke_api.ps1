Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$baseUrl = "http://127.0.0.1:8000"

Write-Host "GET /health"
$health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
$health | ConvertTo-Json -Depth 6

Write-Host "POST /ingest/run"
$ingest = Invoke-RestMethod -Uri "$baseUrl/ingest/run" -Method Post
$ingest | ConvertTo-Json -Depth 6

Write-Host "GET /ingest/status"
$status = Invoke-RestMethod -Uri "$baseUrl/ingest/status" -Method Get
$status | ConvertTo-Json -Depth 6

