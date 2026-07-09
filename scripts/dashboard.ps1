# Templar - launch the dashboard on demand
# Starts the API (port 8000) and frontend dev server (port 5173), opens the
# browser, and shuts both down when you press Ctrl+C in this window.

$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root "venv\Scripts\python.exe"

$api = Start-Process -FilePath $Python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--port", "8000" `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden

$frontend = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory (Join-Path $Root "frontend") -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host "Dashboard running: http://localhost:5173  (API on :8000)"
Write-Host "Press Ctrl+C to stop both servers."

try {
    Wait-Process -Id $api.Id
} finally {
    foreach ($proc in @($api, $frontend)) {
        if ($proc -and -not $proc.HasExited) {
            # Kill the process tree (npm spawns node children)
            & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
        }
    }
    Write-Host "Dashboard stopped."
}
