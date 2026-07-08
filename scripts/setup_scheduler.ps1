# Templar - Task Scheduler setup
# Run this once as Administrator to register both scheduled tasks.
# The daily run happens shortly before market open (9:30 AM ET); orders queue
# with Alpaca and fill at the opening auction. main.py run skips execution only
# if the next open is more than an hour away. Adjust for your timezone.

param(
    [string]$DailyTime  = "07:15",   # time to run daily pipeline (local time; 7:15 MT = 9:15 ET)
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

# Interactive: shows a console window with live output in the user's session.
# Caveats: the task only runs while the user is logged on, and closing the
# window kills the run mid-flight. Output is also captured in logs\ either way.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# --- Daily task (Mon-Fri) ---
$dailyAction  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $DailyArgs -WorkingDirectory $Root
$dailyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $DailyTime
$dailySettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable

Register-ScheduledTask `
    -TaskName   "Templar - Daily Run" `
    -Action     $dailyAction `
    -Trigger    $dailyTrigger `
    -Settings   $dailySettings `
    -Principal  $principal `
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
    -Principal  $principal `
    -Force

Write-Host "Registered: Templar - Weekly Retrain (Sunday at $WeeklyTime)"
Write-Host ""
Write-Host "Done. View tasks in Task Scheduler under 'Task Scheduler Library'."
Write-Host "To remove: Unregister-ScheduledTask -TaskName 'Templar - Daily Run' -Confirm:`$false"
