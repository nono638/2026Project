#!/usr/bin/env bash
# repairrun.sh — Resets the project to a known-good nighttime-safe state
#
# Usage:
#   bash _claude_sandbox_setup/scripts/repairrun.sh [project_dir]
#
# What it fixes:
#   1. settings.json → restores nighttime settings (in case dayrun.sh left daytime settings)
#   2. active_mode.md → restores nighttime rules (same reason)
#   3. .claude/hooks/ → re-copies all hook scripts from templates (in case files are missing/stale)
#   4. tracker.json → resets any "in_progress" tasks back to "todo" (in case nighttime crashed
#      mid-task and left a task stuck)
#
# Safe to run any time. Does not touch your code, git history, spec files, or result files.

set -euo pipefail

# Use python3 if available, fall back to python (Windows/Git Bash compatibility)
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Neither 'python3' nor 'python' found on PATH."
    exit 1
fi

PROJECT_DIR="${1:-$(pwd)}"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

NIGHTTIME_SETTINGS="_claude_sandbox_setup/templates/nighttime_settings.json"
NIGHTTIME_SUPPLEMENT="_claude_sandbox_setup/templates/nighttime_supplement.md"
HOOKS_SRC="_claude_sandbox_setup/hooks"
TRACKER="DaytimeNighttimeHandOff/tracker.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] repairrun.sh started"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Project: $PROJECT_DIR"
echo ""

if [ ! -f "$NIGHTTIME_SETTINGS" ]; then
    echo "ERROR: $NIGHTTIME_SETTINGS not found."
    echo "Make sure _claude_sandbox_setup/ is present in this project."
    exit 1
fi

# 1. Restore nighttime settings
echo "[ 1/4 ] Restoring nighttime settings.json..."
mkdir -p .claude
cp "$NIGHTTIME_SETTINGS" ".claude/settings.json"
echo "        Done."

# 2. Restore nighttime active_mode.md
echo "[ 2/4 ] Restoring nighttime active_mode.md..."
cp "$NIGHTTIME_SUPPLEMENT" ".claude/active_mode.md"
echo "        Done."

# 3. Re-copy hook scripts
echo "[ 3/4 ] Restoring hook scripts..."
mkdir -p .claude/hooks
cp "$HOOKS_SRC/"*.py ".claude/hooks/"
echo "        Done."

# 4. Unstick in_progress tasks in tracker.json
echo "[ 4/4 ] Checking tracker.json for stuck in_progress tasks..."
if [ ! -f "$TRACKER" ]; then
    echo "        No tracker.json found — skipping."
else
    $PYTHON - "$TRACKER" <<'PYEOF'
import json, sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    tasks = json.load(f)

stuck = [t for t in tasks if t.get("status") == "in_progress"]

if not stuck:
    print("        No stuck tasks found.")
    sys.exit(0)

for t in stuck:
    print(f"        Resetting {t.get('task_id', 'unknown')} from in_progress → todo")
    t["status"] = "todo"
    t["nighttime_started"] = None

with open(path, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)
    f.write("\n")

print(f"        Reset {len(stuck)} task(s).")
PYEOF
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repair complete. Project is in nighttime-safe mode."
echo "        Run dayrun.sh to start a daytime session."
echo "        Run nightrun.sh to start a nighttime session."
