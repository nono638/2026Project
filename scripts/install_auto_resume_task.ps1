# Install a Windows scheduled task that auto-resumes Experiment 1 at logon
# after a BSOD/reboot.
#
# Why this exists: the 5090 BSODs ~once every 12-24 h of sustained load. The
# row-level CSV checkpoint in run_experiment_1.py makes resume cheap, but
# *someone* still has to log in and re-launch the script. This task closes
# that gap so a BSOD costs only the reboot time (~1 minute) plus whatever
# row was in flight, not the hours between you noticing and restarting.
#
# Run from PowerShell (admin not strictly required — the task runs as the
# current user). Re-running this script is idempotent; it overwrites any
# existing task with the same name.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_auto_resume_task.ps1
#
# To remove: scripts\uninstall_auto_resume_task.ps1 (or
#   Unregister-ScheduledTask -TaskName 'RagBench-Experiment1-AutoResume' -Confirm:$false)
#
# Pre-req:
#   - .venv exists at D:\Projects\2026Project\.venv
#   - scripts\start_ollama_safe.ps1 should be run AFTER logon manually OR
#     extended into this task — currently NOT auto-started because the
#     Ollama server may already be alive at logon if the tray app launches.

$ErrorActionPreference = 'Stop'

$taskName = 'RagBench-Experiment1-AutoResume'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$watchdog = Join-Path $projectRoot 'scripts\run_experiment_1_watchdog.py'

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python venv not found at $pythonExe. Activate / create the venv first."
    exit 1
}
if (-not (Test-Path $watchdog)) {
    Write-Error "Watchdog script not found at $watchdog."
    exit 1
}

# Action: run the watchdog from the project root.
$action = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument "`"$watchdog`"" `
    -WorkingDirectory $projectRoot

# Trigger: at logon for the current user. AtStartup would require admin
# AND wouldn't have HKEY_CURRENT_USER env vars; AtLogon is the right call
# for a single-user development laptop.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# 30 s delay to let Ollama / driver settle before we slam them with work.
$trigger.Delay = 'PT30S'

# Settings: no time limit, allow on battery, single-instance.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 7) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Principal: run interactively as the current user (HighestAvailable allows
# elevation if the user is admin, but does not require it).
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Idempotent register: remove old version first.
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing scheduled task '$taskName'..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description ("Auto-resume RagBench Experiment 1 via watchdog after a " +
                  "BSOD/reboot. Installed by install_auto_resume_task.ps1.") | Out-Null

Write-Host ""
Write-Host "Scheduled task '$taskName' installed." -ForegroundColor Green
Write-Host "  Trigger: at logon ($env:USERNAME), 30 s delay"
Write-Host "  Command: $pythonExe `"$watchdog`""
Write-Host "  Working dir: $projectRoot"
Write-Host ""
Write-Host "After a BSOD: log back in, wait ~30 s, the task fires and resumes."
Write-Host "To disable temporarily: Disable-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
