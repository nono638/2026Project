@echo off
setlocal enabledelayedexpansion

:: dayrun.bat — Launches Claude in daytime/collaborative mode
::
:: Usage:
::   dayrun.bat [project_dir]
::
:: Examples:
::   dayrun.bat                           -- Run in current directory
::   dayrun.bat C:\Projects\myapp        -- Run in specific directory
::
:: What it does:
::   1. Installs daytime settings (.claude/settings.json)
::      -> WebSearch, WebFetch, AskUserQuestion allowed
::      -> no_ask_human hook NOT active (you're present)
::      -> directory guard and all other safety rules still apply
::   2. Copies latest hook scripts to .claude\hooks\
::   3. Launches Claude interactively with daytime rules pre-loaded
::      (no need to type /day -- the supplement is injected automatically)
::   4. On exit: restores nighttime settings so the project defaults
::      back to safe/locked mode
::
:: Requirements:
::   - claude CLI on PATH
::   - python on PATH
::   - Run from or pass the target project directory
::   - _claude_sandbox_setup\ must exist in the project

set "PROJECT_DIR=%~1"
if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%CD%"

set "DAYTIME_SETTINGS=%PROJECT_DIR%\_claude_sandbox_setup\templates\daytime_settings.json"
set "NIGHTTIME_SETTINGS=%PROJECT_DIR%\_claude_sandbox_setup\templates\nighttime_settings.json"
set "DAYTIME_SUPPLEMENT=%PROJECT_DIR%\_claude_sandbox_setup\templates\daytime_supplement.md"
set "NIGHTTIME_SUPPLEMENT=%PROJECT_DIR%\_claude_sandbox_setup\templates\nighttime_supplement.md"
set "HOOKS_SRC=%PROJECT_DIR%\_claude_sandbox_setup\hooks"

cd /d "%PROJECT_DIR%" || (
    echo ERROR: Cannot cd to %PROJECT_DIR%
    exit /b 1
)

if not exist "%DAYTIME_SETTINGS%" (
    echo ERROR: %DAYTIME_SETTINGS% not found.
    echo Make sure _claude_sandbox_setup\ is present in this project.
    exit /b 1
)

echo [%DATE% %TIME%] dayrun.bat started
echo [%DATE% %TIME%] Project: %PROJECT_DIR%
echo [%DATE% %TIME%] Installing daytime settings...

if not exist ".claude\hooks" mkdir ".claude\hooks"
copy /y "%DAYTIME_SETTINGS%" ".claude\settings.json" > nul
copy /y "%DAYTIME_SUPPLEMENT%" ".claude\active_mode.md" > nul
copy /y "%HOOKS_SRC%\*.py" ".claude\hooks\" > nul

echo [%DATE% %TIME%] Daytime mode active: WebSearch, WebFetch, questions allowed.
echo.

:: Launch Claude interactively. CLAUDE.md imports .claude\active_mode.md which
:: now contains daytime rules -- no need to type /day.
claude

:: Restore nighttime settings and active_mode.md after Claude exits
echo.
echo [%DATE% %TIME%] Restoring nighttime settings...
copy /y "%NIGHTTIME_SETTINGS%" ".claude\settings.json" > nul
copy /y "%NIGHTTIME_SUPPLEMENT%" ".claude\active_mode.md" > nul
echo [%DATE% %TIME%] Done. Project is back in nighttime-safe mode.

endlocal
