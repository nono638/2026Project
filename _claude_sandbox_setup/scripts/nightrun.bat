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
::   NIGHTRUN_MODEL=claude-sonnet-4-6  Override the Claude model (default: claude-opus-4-6)
::   NIGHTRUN_EFFORT=max               Override effort level (default: high)
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
set "HELPER=%PROJECT_DIR%\_claude_sandbox_setup\scripts\nightrun_helper.py"
set "SETUP_DIR=%PROJECT_DIR%\_claude_sandbox_setup"
set "NIGHTTIME_PROMPT=Begin nighttime work session. Check DaytimeNighttimeHandOff/tracker.json for in_progress tasks to resume and todo tasks to start. Follow the nighttime workflow defined in CLAUDE.md."

:: Generate session name — use a simple date format to avoid quote-nesting issues
for /f "delims=" %%i in ('python -c "from datetime import date; print(date.today().strftime('nightrun-%%Y%%m%%d'))"') do set "SESSION_NAME=%%i"
if not defined SESSION_NAME set "SESSION_NAME=nightrun-session"

:: Capture session start time so the summary can distinguish tonight's work from previous
for /f "delims=" %%i in ('python "%HELPER%" timestamp') do set "SESSION_START=%%i"

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

:: --- Resolve model from config file → fallback file → env var → hardcoded ---
set "MODEL="
set "EFFORT="
set "MODEL_SOURCE="

:: Pass env vars as fallback args so the helper can use them if config/fallback missing
set "ENV_MODEL="
set "ENV_EFFORT="
if defined NIGHTRUN_MODEL set "ENV_MODEL=%NIGHTRUN_MODEL%"
if defined NIGHTRUN_EFFORT set "ENV_EFFORT=%NIGHTRUN_EFFORT%"

for /f "tokens=1,* delims==" %%a in ('python "%HELPER%" model "%SETUP_DIR%" night "%ENV_MODEL%" "%ENV_EFFORT%" 2^>nul') do (
    if "%%a"=="MODEL" set "MODEL=%%b"
    if "%%a"=="EFFORT" set "EFFORT=%%b"
    if "%%a"=="SOURCE" set "MODEL_SOURCE=%%b"
)

:: Fallback if helper failed entirely
if not defined MODEL set "MODEL=claude-opus-4-6"
if not defined EFFORT set "EFFORT=medium"
if not defined MODEL_SOURCE set "MODEL_SOURCE=hardcoded"

set "MODEL_FLAG="

:: Count pending tasks using helper script (avoids inline Python in if-blocks)
set "PREFLIGHT_PENDING=0"
if exist "%TRACKER%" (
    python "%HELPER%" count "%TRACKER%" > "%TEMP%\nightrun_preflight.txt" 2>nul
    set /p PREFLIGHT_PENDING=<"%TEMP%\nightrun_preflight.txt"
    del "%TEMP%\nightrun_preflight.txt" 2>nul
)

echo ========================================
echo   Good evening. Ready to begin.
echo.
echo   Pending tasks: %PREFLIGHT_PENDING%
echo   Max turns:     %MAX_TURNS%
echo   Cooldown:      %COOLDOWN%s
echo   Model:         %MODEL% [%MODEL_SOURCE%]
echo   Effort:        %EFFORT%
echo ========================================

:: Show task summary if there are pending tasks
if exist "%TRACKER%" if not "%PREFLIGHT_PENDING%"=="0" (
    echo.
    echo   Tonight's work:
    python "%HELPER%" show "%TRACKER%" 2>nul
    echo.
)
echo.
echo Type a model name to override (e.g. claude-sonnet-4-6), or press Enter to proceed.
echo Ctrl+C to cancel.
echo.
set "USER_MODEL="
set /p USER_MODEL="> "

if defined USER_MODEL (
    set "MODEL=!USER_MODEL!"
    echo [%DATE% %TIME%] Model set to: !MODEL!
)

set "MODEL_FLAG=--model !MODEL! --effort !EFFORT!"

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
claude --dangerously-skip-permissions --max-turns %MAX_TURNS% %MODEL_FLAG% --print -n "%SESSION_NAME%" "%NIGHTTIME_PROMPT%"
if errorlevel 1 (
    echo [%DATE% %TIME%] Claude exited with non-zero status (expected on usage cap^).
)

echo [%DATE% %TIME%] Claude session ended. Checking tracker...
set /a RELAUNCH_COUNT=%RELAUNCH_COUNT%+1

:: Count pending tasks using helper script
set "PENDING=0"
if exist "%TRACKER%" (
    python "%HELPER%" count "%TRACKER%" > "%TEMP%\nightrun_count.txt" 2>nul
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
    python "%HELPER%" summary "%TRACKER%" "%SESSION_START%" 2>nul
) else (
    echo   No tracker.json found.
)

:: Auto-promote: if session completed successfully, update the fallback file
:: so the environment tracks the last-known-good model.
if "%PENDING%"=="0" (
    echo.
    python "%HELPER%" promote "%SETUP_DIR%" "%MODEL%" "%EFFORT%" 2>nul
)

echo ========================================
endlocal
