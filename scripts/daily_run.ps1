# Templar - daily trading pipeline
# Runs fetch + signal generation + order execution before market open
# Schedule: Mon-Fri at 9:00 AM local time (adjust to your timezone vs ET)

param(
    [switch]$Dry  # pass -Dry to log orders without submitting
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "daily_$(Get-Date -Format 'yyyy-MM-dd').log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== Templar daily run starting ==="

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Log "ERROR: venv not found at $Python"
    exit 1
}

# Check if today is a market day (Mon-Fri); skip weekends
$day = (Get-Date).DayOfWeek
if ($day -eq "Saturday" -or $day -eq "Sunday") {
    Log "Weekend - skipping."
    exit 0
}

# 1. Fetch latest bars and sentiment
Log "--- Phase 1: Fetching market data ---"
& $Python (Join-Path $Root "main.py") fetch 2>&1 | Tee-Object -Append -FilePath $LogFile
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: fetch failed (exit $LASTEXITCODE)"
    exit 1
}

# 2. Generate signals and execute orders
Log "--- Phase 2: Generating signals and executing orders ---"
$MainPy = Join-Path $Root "main.py"
$runArgs = if ($Dry) { @("run", "--dry") } else { @("run") }
& $Python $MainPy $runArgs 2>&1 | Tee-Object -Append -FilePath $LogFile
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: run failed (exit $LASTEXITCODE)"
    exit 1
}

Log "=== Templar daily run complete ==="
