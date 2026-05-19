# Templar — Task Scheduler setup
# Run this once as Administrator to register both scheduled tasks.
# Adjust $MarketOpenTime if you are not in Eastern Time.

param(
    [string]$DailyTime  = "09:00",   # time to run daily pipeline (local time)
    [string]$WeeklyTime = "22:00",   # time to run weekly retrain (Sunday)
    [switch]$Dry                      # register tasks in dry-run mode
)

$Root = Split-Path $PSScriptRoot -Parent
$DailyScript  = Join-Path $PSScriptRoot "daily_run.ps1"
$WeeklyScript = Join-Path $PSScriptRoot "weekly_train.ps1"

$DailyArgs  = "-NonInteractive -ExecutionPolicy Bypass -File `"$DailyScript`""
$WeeklyArgs = "-NonInteractive -ExecutionPolicy Bypass -File `"$WeeklyScript`""
if ($Dry) {
    $DailyArgs += " -Dry"
}

# --- Daily task (Mon–Fri) ---
$dailyAction  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $DailyArgs -WorkingDirectory $Root
$dailyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $DailyTime
$dailySettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable

Register-ScheduledTask `
    -TaskName   "Templar - Daily Run" `
    -Action     $dailyAction `
    -Trigger    $dailyTrigger `
    -Settings   $dailySettings `
    -RunLevel   Highest `
    -Force

Write-Host "Registered: Templar - Daily Run (Mon-Fri at $DailyTime)"

# --- Weekly retrain task (Sunday) ---
$weeklyAction  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $WeeklyArgs -WorkingDirectory $Root
$weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $WeeklyTime
$weeklySettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 10) -StartWhenAvailable

Register-ScheduledTask `
    -TaskName   "Templar - Weekly Retrain" `
    -Action     $weeklyAction `
    -Trigger    $weeklyTrigger `
    -Settings   $weeklySettings `
    -RunLevel   Highest `
    -Force

Write-Host "Registered: Templar - Weekly Retrain (Sunday at $WeeklyTime)"
Write-Host ""
Write-Host "Done. View tasks in Task Scheduler under 'Task Scheduler Library'."
Write-Host "To remove: Unregister-ScheduledTask -TaskName 'Templar - Daily Run' -Confirm:`$false"
