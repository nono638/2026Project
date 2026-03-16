#!/usr/bin/env bash
# dayrun.sh — Launches Claude in daytime/collaborative mode
#
# Usage:
#   bash _claude_sandbox_setup/scripts/dayrun.sh [project_dir]
#
# Examples:
#   bash _claude_sandbox_setup/scripts/dayrun.sh          # Run in current directory
#   bash _claude_sandbox_setup/scripts/dayrun.sh ~/myapp  # Run in specific directory
#
# What it does:
#   1. Installs daytime settings (.claude/settings.json)
#      → WebSearch, WebFetch, AskUserQuestion allowed
#      → no_ask_human hook NOT active (you're present)
#      → directory guard and all other safety rules still apply
#   2. Copies latest hook scripts to .claude/hooks/
#   3. Writes daytime rules to .claude/active_mode.md (CLAUDE.md imports this)
#   4. Launches Claude interactively — daytime mode is active from the first message
#   5. On exit: restores nighttime settings and active_mode.md so the project
#      defaults back to safe/locked mode even if you walk away
#
# Requirements:
#   - claude CLI on PATH
#   - Run from or pass the target project directory
#   - _claude_sandbox_setup/ must exist in the project

set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

DAYTIME_SETTINGS="_claude_sandbox_setup/templates/daytime_settings.json"
NIGHTTIME_SETTINGS="_claude_sandbox_setup/templates/nighttime_settings.json"
DAYTIME_SUPPLEMENT="_claude_sandbox_setup/templates/daytime_supplement.md"
NIGHTTIME_SUPPLEMENT="_claude_sandbox_setup/templates/nighttime_supplement.md"
HOOKS_SRC="_claude_sandbox_setup/hooks"

if [ ! -f "$DAYTIME_SETTINGS" ]; then
    echo "ERROR: $DAYTIME_SETTINGS not found."
    echo "Make sure _claude_sandbox_setup/ is present in this project."
    exit 1
fi

# On exit (including Ctrl-C), restore nighttime settings.
# This ensures the project is always left in nighttime-safe mode.
restore_nighttime() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restoring nighttime settings..."
    cp "$NIGHTTIME_SETTINGS" ".claude/settings.json"
    cp "$NIGHTTIME_SUPPLEMENT" ".claude/active_mode.md"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Project is back in nighttime-safe mode."
}
trap restore_nighttime EXIT HUP TERM INT

# Install daytime settings, mode rules, and hooks
echo "[$(date '+%Y-%m-%d %H:%M:%S')] dayrun.sh started"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Project: $PROJECT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing daytime settings..."
mkdir -p .claude/hooks
cp "$DAYTIME_SETTINGS" ".claude/settings.json"
cp "$DAYTIME_SUPPLEMENT" ".claude/active_mode.md"
cp "$HOOKS_SRC/"*.py ".claude/hooks/"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daytime mode active: WebSearch, WebFetch, questions allowed."
echo ""

# Launch Claude interactively with 1M context and high effort.
# CLAUDE.md imports .claude/active_mode.md which now contains daytime rules.
claude --model claude-opus-4-6[1m] --effort high

# trap handles nighttime restore on EXIT
