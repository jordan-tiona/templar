# Templar - weekly retrain pipeline
# Runs: fetch --refresh, tune (monthly), train
# Schedule: Sunday at 10:00 PM local time

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "train_$(Get-Date -Format 'yyyy-MM-dd').log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

Log "=== Templar weekly retrain starting ==="

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Log "ERROR: venv not found at $Python"
    exit 1
}

# 1. Refresh all market data and sentiment
Log "--- Phase 1: Refreshing market data ---"
& $Python (Join-Path $Root "main.py") fetch --refresh 2>&1 | Tee-Object -Append -FilePath $LogFile -Encoding utf8
if ($LASTEXITCODE -ne 0) { Log "ERROR: fetch failed"; exit 1 }

# 2. Run Optuna tuning once a month (first Sunday of the month)
$dayOfMonth = (Get-Date).Day
if ($dayOfMonth -le 7) {
    Log "--- Phase 2: Monthly Optuna tuning ---"
    & $Python (Join-Path $Root "main.py") tune --trials 50 2>&1 | Tee-Object -Append -FilePath $LogFile -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Log "WARNING: tune failed, continuing with existing params" }
} else {
    Log "--- Phase 2: Skipping Optuna (not first Sunday of month) ---"
}

# 3. Retrain both models
Log "--- Phase 3: Retraining models ---"
& $Python (Join-Path $Root "main.py") train 2>&1 | Tee-Object -Append -FilePath $LogFile -Encoding utf8
if ($LASTEXITCODE -ne 0) { Log "ERROR: train failed"; exit 1 }

Log "=== Templar weekly retrain complete ==="
