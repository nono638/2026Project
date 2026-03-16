@echo off
setlocal enabledelayedexpansion

:: nightrun.bat — Launches Claude nighttime sessions with auto-resume on usage cap
::
:: Usage:
::   nightrun.bat [project_dir] [cooldown_seconds] [max_turns]
::
:: Examples:
::   nightrun.bat                          -- Run in current directory, defaults
::   nightrun.bat C:\Projects\myapp        -- Run in specific directory
::   nightrun.bat . 600 2000               -- 10 min cooldown, 2000 max turns
::
:: Environment variable overrides:
::   NIGHTRUN_COOLDOWN=300       Seconds to wait between relaunches (default: 300)
::   NIGHTRUN_MAX_TURNS=2000     Max turns per Claude session (default: 2000)
::   NIGHTRUN_MAX_RELAUNCHES=10  Max relaunches before giving up (default: 10)
::
:: What it does:
::   1. Pre-flight checks (claude on PATH, python, setup files present)
::   2. Installs nighttime settings, hooks, and active_mode.md
::   3. Launches Claude with --dangerously-skip-permissions and the nighttime prompt
::   4. When Claude exits, checks tracker.json for remaining tasks
::   5. If tasks remain, waits COOLDOWN seconds then relaunches
::   6. Gives up after MAX_RELAUNCHES with no progress
::   7. Prints a morning summary on exit
::
:: Requirements:
::   - claude CLI on PATH
::   - python on PATH
::   - Run from the target project directory (must have DaytimeNighttimeHandOff/)

set "PROJECT_DIR=%~1"
if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%CD%"

set "COOLDOWN=%~2"
if "%COOLDOWN%"=="" (
    if defined NIGHTRUN_COOLDOWN (set "COOLDOWN=%NIGHTRUN_COOLDOWN%") else (set "COOLDOWN=300")
)

set "MAX_TURNS=%~3"
if "%MAX_TURNS%"=="" (
    if defined NIGHTRUN_MAX_TURNS (set "MAX_TURNS=%NIGHTRUN_MAX_TURNS%") else (set "MAX_TURNS=2000")
)

set "MAX_RELAUNCHES=10"
if defined NIGHTRUN_MAX_RELAUNCHES set "MAX_RELAUNCHES=%NIGHTRUN_MAX_RELAUNCHES%"

set "TRACKER=%PROJECT_DIR%\DaytimeNighttimeHandOff\tracker.json"
set "NIGHTTIME_PROMPT=Begin nighttime work session. Check DaytimeNighttimeHandOff/tracker.json for in_progress tasks to resume and todo tasks to start. Follow the nighttime workflow defined in CLAUDE.md."

for /f "delims=" %%i in ('python -c "from datetime import date; print(date.today().strftime('nightrun-%%Y%%m%%d'))"') do set "SESSION_NAME=%%i"

cd /d "%PROJECT_DIR%" || (
    echo ERROR: Cannot cd to %PROJECT_DIR%
    exit /b 1
)

echo [%DATE% %TIME%] nightrun.bat started
echo [%DATE% %TIME%] Project: %PROJECT_DIR%
echo [%DATE% %TIME%] Cooldown: %COOLDOWN%s ^| Max turns: %MAX_TURNS% ^| Max relaunches: %MAX_RELAUNCHES%
echo.

:: --- Pre-flight checks ---
set "PREFLIGHT_OK=1"

where claude >nul 2>&1 || (
    echo ERROR: 'claude' not found on PATH. Install Claude Code and ensure it's on your PATH.
    set "PREFLIGHT_OK=0"
)

where python >nul 2>&1 || (
    echo ERROR: 'python' not found on PATH. Python 3 is required.
    set "PREFLIGHT_OK=0"
)

if not exist "_claude_sandbox_setup" (
    echo ERROR: '_claude_sandbox_setup\' not found in %PROJECT_DIR%.
    echo        Run this script from the project root.
    set "PREFLIGHT_OK=0"
)

if not exist "_claude_sandbox_setup\templates\nighttime_settings.json" (
    echo ERROR: nighttime_settings.json template missing. Setup may be incomplete.
    set "PREFLIGHT_OK=0"
)

git rev-parse --is-inside-work-tree >nul 2>&1 || (
    echo ERROR: Not a git repository. The nighttime workflow requires git.
    echo        Run 'git init' and make an initial commit first.
    set "PREFLIGHT_OK=0"
)

if not exist "DaytimeNighttimeHandOff" (
    echo WARNING: DaytimeNighttimeHandOff\ not found. Run setup first (see _claude_sandbox_setup\SETUP.md^).
    echo          Continuing anyway -- Claude will handle this.
)

if "%PREFLIGHT_OK%"=="0" (
    echo.
    echo Pre-flight checks failed. Fix the above errors and retry.
    exit /b 1
)

echo [%DATE% %TIME%] Pre-flight checks passed.
echo.

:: --- Main loop ---
set "RELAUNCH_COUNT=0"
set "LAST_PENDING=-1"

:loop
:: Guard: stop if we've hit the relaunch limit
if %RELAUNCH_COUNT% GEQ %MAX_RELAUNCHES% (
    echo [%DATE% %TIME%] ERROR: Reached max relaunches (%MAX_RELAUNCHES%^) without completing all tasks.
    echo [%DATE% %TIME%] Check tracker.json and nighttime.log. Run repairrun.bat if tasks are stuck.
    goto :summary
)

:: Install nighttime settings, mode rules, and hooks before each launch.
set /a RELAUNCH_DISPLAY=%RELAUNCH_COUNT%+1
echo [%DATE% %TIME%] Installing nighttime settings (relaunch %RELAUNCH_DISPLAY%/%MAX_RELAUNCHES%)...

if not exist ".claude\hooks" mkdir ".claude\hooks"

copy /y "_claude_sandbox_setup\templates\nighttime_settings.json" ".claude\settings.json" >nul 2>&1 || (
    echo ERROR: Failed to copy settings.json. Aborting.
    goto :summary
)
copy /y "_claude_sandbox_setup\templates\nighttime_supplement.md" ".claude\active_mode.md" >nul 2>&1 || (
    echo ERROR: Failed to copy active_mode.md. Aborting.
    goto :summary
)
copy /y "_claude_sandbox_setup\hooks\*.py" ".claude\hooks\" >nul 2>&1 || (
    echo ERROR: Failed to copy hook scripts. Aborting.
    goto :summary
)

:: Back up tracker.json before each launch — protects against mid-write corruption
if exist "%TRACKER%" copy /y "%TRACKER%" "%TRACKER%.bak" >nul 2>&1

echo [%DATE% %TIME%] Starting Claude nighttime session...

:: Launch Claude. Non-zero exit is expected on usage cap -- don't treat it as fatal.
claude --dangerously-skip-permissions --max-turns %MAX_TURNS% --print -n "%SESSION_NAME%" "%NIGHTTIME_PROMPT%"
if errorlevel 1 (
    echo [%DATE% %TIME%] Claude exited with non-zero status (expected on usage cap^).
)

echo [%DATE% %TIME%] Claude session ended. Checking tracker...
set /a RELAUNCH_COUNT=%RELAUNCH_COUNT%+1

:: Count pending tasks
set "PENDING=0"
if exist "%TRACKER%" (
    python -c "
import json, sys
try:
    with open(r'%TRACKER%') as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get('status') in ('todo', 'in_progress')]
    print(len(pending))
except Exception as e:
    sys.stderr.write('tracker read error: ' + str(e) + '\n')
    print(0)
" > "%TEMP%\nightrun_count.txt" 2>nul
    set /p PENDING=<"%TEMP%\nightrun_count.txt"
    del "%TEMP%\nightrun_count.txt" 2>nul
)

if "%PENDING%"=="0" (
    echo [%DATE% %TIME%] All tasks complete or no tracker found. Exiting.
    goto :summary
)

:: Detect no-progress: same count two sessions in a row
if "%PENDING%"=="%LAST_PENDING%" (
    echo [%DATE% %TIME%] WARNING: No progress since last session (%PENDING% task(s^) still pending^).
    echo [%DATE% %TIME%] Claude may be crashing on startup. Will abort after %MAX_RELAUNCHES% total relaunches.
)

set "LAST_PENDING=%PENDING%"
echo [%DATE% %TIME%] %PENDING% task(s) remaining. Waiting %COOLDOWN% seconds before resuming...
timeout /t %COOLDOWN% /nobreak >nul

goto :loop

:summary
echo.
echo [%DATE% %TIME%] nightrun.bat finished.
echo.
echo ========================================
echo   NIGHTRUN SUMMARY
echo   %DATE% %TIME%
echo   Total relaunches: %RELAUNCH_COUNT%
echo ========================================

if exist "%TRACKER%" (
    python -c "
import json
try:
    with open(r'%TRACKER%') as f:
        tasks = json.load(f)
    done    = [t for t in tasks if t.get('status') == 'done']
    skipped = [t for t in tasks if t.get('status') == 'skipped']
    blocked = [t for t in tasks if t.get('status') == 'blocked']
    todo    = [t for t in tasks if t.get('status') == 'todo']
    print(f'  Done:    {len(done)}')
    print(f'  Skipped: {len(skipped)}')
    print(f'  Blocked: {len(blocked)}  <- needs your input')
    print(f'  Todo:    {len(todo)}     <- not started')
    print()
    if done:
        print('  Completed tasks:')
        for t in done:
            branch = t.get('branch', 'no branch')
            flags  = t.get('flags', [])
            flag_str = f'  [{len(flags)} flag(s)]' if flags else ''
            print(f'    {t[\"task_id\"]}: {t.get(\"description\", \"\")} -- branch: {branch}{flag_str}')
    if skipped:
        print('  Skipped tasks:')
        for t in skipped:
            print(f'    {t[\"task_id\"]}: {t.get(\"nighttime_comments\", \"see result.md\")}')
    if blocked:
        print('  Blocked tasks (need your input before next run):')
        for t in blocked:
            print(f'    {t[\"task_id\"]}: {t.get(\"blocked_reason\", \"see tracker.json\")}')
except Exception as e:
    print(f'  (Could not read tracker: {e})')
" 2>nul || echo   (Could not read tracker.json)
) else (
    echo   No tracker.json found.
)

echo ========================================
endlocal
