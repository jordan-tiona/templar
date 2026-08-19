# Templar - daily trading pipeline
# Runs fetch + signal generation + order execution shortly before market open;
# orders queue with Alpaca and fill at the opening auction.
# Schedule: Mon-Fri at 7:15 AM Mountain (9:15 AM ET)

param(
    [switch]$Dry  # pass -Dry to log orders without submitting
)

# Disable console QuickEdit mode. With it on, a stray click into this
# (intentionally visible) window selects text and freezes all console I/O
# until someone clicks back in and presses Enter/Esc -- which silently
# stalled the run on 2026-08-13. Turning it off keeps the window visible
# and readable but stops clicks from pausing execution.
try {
    $sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern IntPtr GetStdHandle(int nStdHandle);
'@
    Add-Type -MemberDefinition $sig -Name NativeConsole -Namespace Win32Console -ErrorAction Stop

    $STD_INPUT_HANDLE = -10
    $ENABLE_QUICK_EDIT_MODE = 0x0040
    $ENABLE_EXTENDED_FLAGS = 0x0080

    $handle = [Win32Console.NativeConsole]::GetStdHandle($STD_INPUT_HANDLE)
    $mode = 0
    if ([Win32Console.NativeConsole]::GetConsoleMode($handle, [ref]$mode)) {
        $newMode = ($mode -band (-bnot $ENABLE_QUICK_EDIT_MODE)) -bor $ENABLE_EXTENDED_FLAGS
        [Win32Console.NativeConsole]::SetConsoleMode($handle, $newMode) | Out-Null
    }
} catch {
    # Non-fatal: worst case QuickEdit stays on and we're back to the old behavior.
}

$Root = Split-Path $PSScriptRoot -Parent
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "scheduler_$(Get-Date -Format 'yyyy-MM-dd').log"

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

# 2. Start Ollama for the plain-English summaries if it isn't already running.
# Track the PID so we only stop an instance we started, never the user's own.
# Detection is a raw TCP probe: PS 5.1 Invoke-WebRequest can time out against
# an Ollama that is up but busy with inference.
$OllamaProc = $null
$OllamaUp = $false
$tcp = New-Object Net.Sockets.TcpClient
try {
    $ar = $tcp.BeginConnect("127.0.0.1", 11434, $null, $null)
    $OllamaUp = $ar.AsyncWaitHandle.WaitOne(2000) -and $tcp.Connected
} catch {} finally { $tcp.Close() }

if ($OllamaUp) {
    Log "Ollama already running -- will leave it running afterwards"
} else {
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCmd) {
        $OllamaProc = Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden -PassThru
        Log "Started Ollama (PID $($OllamaProc.Id)) for summary generation"
        Start-Sleep -Seconds 3
    } else {
        Log "Ollama not found -- plain-English summaries will be skipped"
    }
}

# 3. Generate signals and execute orders
Log "--- Phase 2: Generating signals and executing orders ---"
$MainPy = Join-Path $Root "main.py"
$runArgs = if ($Dry) { @("run", "--dry") } else { @("run") }
& $Python $MainPy $runArgs 2>&1 | Tee-Object -Append -FilePath $LogFile
$runExit = $LASTEXITCODE

# Stop Ollama if this run started it (taskkill /T catches the model
# runner children that 'ollama serve' spawns; frees the ~2GB model RAM)
if ($OllamaProc -and -not $OllamaProc.HasExited) {
    & taskkill /PID $OllamaProc.Id /T /F 2>$null | Out-Null
    Log "Stopped Ollama (PID $($OllamaProc.Id))"
}

if ($runExit -ne 0) {
    Log "ERROR: run failed (exit $runExit)"
    exit 1
}

Log "=== Templar daily run complete ==="
