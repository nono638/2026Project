@echo off
setlocal enabledelayedexpansion

:: repairrun.bat — Resets the project to a known-good nighttime-safe state
::
:: Usage:
::   repairrun.bat [project_dir]
::
:: What it fixes:
::   1. settings.json  → restores nighttime settings (in case dayrun.bat left daytime settings)
::   2. active_mode.md → restores nighttime rules (same reason)
::   3. .claude\hooks\ → re-copies all hook scripts from templates (missing or stale files)
::   4. tracker.json   → resets any "in_progress" tasks back to "todo" (crashed mid-task)
::
:: Safe to run any time. Does not touch your code, git history, spec files, or result files.

set "PROJECT_DIR=%~1"
if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%CD%"

cd /d "%PROJECT_DIR%" || (
    echo ERROR: Cannot cd to %PROJECT_DIR%
    exit /b 1
)

set "NIGHTTIME_SETTINGS=_claude_sandbox_setup\templates\nighttime_settings.json"
set "NIGHTTIME_SUPPLEMENT=_claude_sandbox_setup\templates\nighttime_supplement.md"
set "HOOKS_SRC=_claude_sandbox_setup\hooks"
set "TRACKER=DaytimeNighttimeHandOff\tracker.json"

echo [%DATE% %TIME%] repairrun.bat started
echo [%DATE% %TIME%] Project: %PROJECT_DIR%
echo.

if not exist "%NIGHTTIME_SETTINGS%" (
    echo ERROR: %NIGHTTIME_SETTINGS% not found.
    echo        Make sure _claude_sandbox_setup\ is present in this project.
    exit /b 1
)

:: 1. Restore nighttime settings
echo [ 1/4 ] Restoring nighttime settings.json...
if not exist ".claude" mkdir ".claude"
copy /y "%NIGHTTIME_SETTINGS%" ".claude\settings.json" >nul || (
    echo         ERROR: Failed to copy settings.json.
    exit /b 1
)
echo         Done.

:: 2. Restore nighttime active_mode.md
echo [ 2/4 ] Restoring nighttime active_mode.md...
copy /y "%NIGHTTIME_SUPPLEMENT%" ".claude\active_mode.md" >nul || (
    echo         ERROR: Failed to copy active_mode.md.
    exit /b 1
)
echo         Done.

:: 3. Re-copy hook scripts
echo [ 3/4 ] Restoring hook scripts...
if not exist ".claude\hooks" mkdir ".claude\hooks"
copy /y "%HOOKS_SRC%\*.py" ".claude\hooks\" >nul || (
    echo         ERROR: Failed to copy hook scripts.
    exit /b 1
)
echo         Done.

:: 4. Unstick in_progress tasks in tracker.json
echo [ 4/4 ] Checking tracker.json for stuck in_progress tasks...
if not exist "%TRACKER%" (
    echo         No tracker.json found -- skipping.
    goto :done
)

python -c "
import json, sys
path = r'%TRACKER%'
with open(path, encoding='utf-8') as f:
    tasks = json.load(f)
stuck = [t for t in tasks if t.get('status') == 'in_progress']
if not stuck:
    print('        No stuck tasks found.')
    sys.exit(0)
for t in stuck:
    print(f'        Resetting {t.get(\"task_id\", \"unknown\")} from in_progress to todo')
    t['status'] = 'todo'
    t['nighttime_started'] = None
with open(path, 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2)
    f.write('\n')
print(f'        Reset {len(stuck)} task(s).')
"

:done
echo.
echo [%DATE% %TIME%] Repair complete. Project is in nighttime-safe mode.
echo         Run dayrun.bat to start a daytime session.
echo         Run nightrun.bat to start a nighttime session.
endlocal
