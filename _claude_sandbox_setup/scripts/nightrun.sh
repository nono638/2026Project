#!/usr/bin/env bash
# nightrun.sh — Launches Claude nighttime sessions with auto-resume on usage cap
#
# Usage:
#   ./nightrun.sh [project_dir] [cooldown_seconds] [max_turns]
#
# Examples:
#   ./nightrun.sh                          # Run in current directory, defaults
#   ./nightrun.sh /home/user/myapp         # Run in specific directory
#   ./nightrun.sh . 600 2000               # 10 min cooldown, 2000 max turns
#
# Environment variable overrides:
#   NIGHTRUN_COOLDOWN=300     Seconds to wait between relaunches (default: 300)
#   NIGHTRUN_MAX_TURNS=2000   Max turns per Claude session (default: 2000)
#   NIGHTRUN_MAX_RELAUNCHES=10  Max times to relaunch before giving up (default: 10)
#   NIGHTRUN_MODEL=claude-sonnet-4-6  Override the Claude model (default: claude-opus-4-6[1m])
#   NIGHTRUN_EFFORT=high              Override effort level (default: medium)
#
# What it does:
#   1. Installs nighttime settings, hooks, and active_mode.md
#   2. Launches Claude with --dangerously-skip-permissions and the nighttime prompt
#   3. When Claude exits (usage cap, completion, or error), checks tracker.json
#   4. If tasks remain (todo or in_progress), waits COOLDOWN seconds then relaunches
#   5. Gives up after MAX_RELAUNCHES consecutive relaunches with no progress
#   6. If all tasks are done/skipped/blocked, exits cleanly
#
# Requirements:
#   - claude CLI on PATH
#   - python3 on PATH
#   - Run from or pass the target project directory (must have DaytimeNighttimeHandOff/)

set -uo pipefail
# Note: -e is intentionally omitted. We handle errors explicitly so a failed
# command inside the loop doesn't silently kill the entire script.

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
COOLDOWN="${2:-${NIGHTRUN_COOLDOWN:-300}}"
MAX_TURNS="${3:-${NIGHTRUN_MAX_TURNS:-2000}}"
MAX_RELAUNCHES="${NIGHTRUN_MAX_RELAUNCHES:-10}"

TRACKER="${PROJECT_DIR}/DaytimeNighttimeHandOff/tracker.json"
NIGHTTIME_PROMPT="Begin nighttime work session. Check DaytimeNighttimeHandOff/tracker.json for in_progress tasks to resume and todo tasks to start. Follow the nighttime workflow defined in CLAUDE.md."

# Session name is fixed for the day — all relaunches within the same nightrun
# share a name for traceability, but each relaunch starts fresh (no --resume)
# so tracker.json remains the authoritative state, not conversation history.
SESSION_NAME="nightrun-$(date +%Y%m%d)"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

echo "[$(date '+%Y-%m-%d %H:%M:%S')] nightrun.sh started"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Project: $PROJECT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooldown: ${COOLDOWN}s | Max turns: $MAX_TURNS | Max relaunches: $MAX_RELAUNCHES"
echo ""

# --- Pre-flight checks ---
PREFLIGHT_OK=1

if ! command -v claude &>/dev/null; then
    echo "ERROR: 'claude' not found on PATH. Install Claude Code and ensure it's on your PATH."
    PREFLIGHT_OK=0
fi

# Python already validated above (PYTHON variable set)

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not a git repository. The nighttime workflow requires git."
    echo "       Run 'git init && git add -A && git commit -m \"initial commit\"' first."
    PREFLIGHT_OK=0
fi

if [ ! -d "_claude_sandbox_setup" ]; then
    echo "ERROR: '_claude_sandbox_setup/' not found in $PROJECT_DIR."
    echo "       Run this script from the project root."
    PREFLIGHT_OK=0
fi

if [ ! -f "_claude_sandbox_setup/templates/nighttime_settings.json" ]; then
    echo "ERROR: nighttime_settings.json template missing. Setup may be incomplete."
    PREFLIGHT_OK=0
fi

if [ ! -d "DaytimeNighttimeHandOff" ]; then
    echo "WARNING: DaytimeNighttimeHandOff/ not found. Run setup first (see _claude_sandbox_setup/SETUP.md)."
    echo "         Continuing anyway — Claude will handle this."
fi

if [ "$PREFLIGHT_OK" -eq 0 ]; then
    echo ""
    echo "Pre-flight checks failed. Fix the above errors and retry."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pre-flight checks passed."
echo ""

# --- Interactive confirmation ---
MODEL="${NIGHTRUN_MODEL:-claude-opus-4-6[1m]}"
EFFORT="${NIGHTRUN_EFFORT:-medium}"
MODEL_FLAG=""

# Count pending tasks for the pre-launch summary
PREFLIGHT_PENDING=0
if [ -f "$TRACKER" ]; then
    PREFLIGHT_PENDING=$($PYTHON -c "
import json, sys
try:
    with open('${TRACKER}') as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get('status') in ('todo', 'in_progress')]
    print(len(pending))
except:
    print(0)
" 2>/dev/null || echo "0")
fi

echo "========================================"
echo "  Good evening. Ready to begin."
echo ""
echo "  Pending tasks: $PREFLIGHT_PENDING"
echo "  Max turns:     $MAX_TURNS"
echo "  Cooldown:      ${COOLDOWN}s"
echo "  Model:         $MODEL"
echo "  Effort:        $EFFORT"
echo "========================================"

# Show task summary if there are pending tasks
if [ -f "$TRACKER" ] && [ "$PREFLIGHT_PENDING" -gt 0 ]; then
    echo ""
    echo "  Tonight's work:"
    $PYTHON -c "
import json
try:
    with open('${TRACKER}') as f:
        tasks = json.load(f)
    for t in tasks:
        s = t.get('status', '')
        if s in ('in_progress', 'todo'):
            tid = t.get('task_id', '???')
            desc = t.get('description', 'no description')
            tag = ' (resuming)' if s == 'in_progress' else ''
            print(f'    {tid}: {desc}{tag}')
except:
    print('    (could not read tracker)')
" 2>/dev/null
    echo ""
fi
echo ""
echo "Type a model name to override (e.g. claude-sonnet-4-6), or press Enter to proceed."
echo "Ctrl+C to cancel."
echo ""
read -r -p "> " USER_MODEL

if [ -n "$USER_MODEL" ]; then
    MODEL="$USER_MODEL"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model set to: $MODEL"
fi

MODEL_FLAG="--model $MODEL --effort $EFFORT"

echo ""

# Returns the count of tasks with status todo or in_progress.
# Prints 0 if tracker doesn't exist or can't be read.
count_pending() {
    if [ ! -f "$TRACKER" ]; then
        echo "0"
        return
    fi
    $PYTHON -c "
import json, sys
try:
    with open('${TRACKER}') as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get('status') in ('todo', 'in_progress')]
    print(len(pending))
except Exception as e:
    sys.stderr.write('tracker read error: ' + str(e) + '\n')
    print(0)
" 2>/dev/null || echo "0"
}

install_nighttime_files() {
    local settings="_claude_sandbox_setup/templates/nighttime_settings.json"
    local supplement="_claude_sandbox_setup/templates/nighttime_supplement.md"
    local hooks_src="_claude_sandbox_setup/hooks"

    if [ ! -f "$settings" ]; then
        echo "ERROR: $settings not found. Is _claude_sandbox_setup/ present?"
        return 1
    fi

    mkdir -p .claude/hooks
    cp "$settings"      ".claude/settings.json"   || { echo "ERROR: Failed to copy settings.json";      return 1; }
    cp "$supplement"    ".claude/active_mode.md"   || { echo "ERROR: Failed to copy active_mode.md";    return 1; }
    cp "$hooks_src/"*.py ".claude/hooks/"          || { echo "ERROR: Failed to copy hook scripts";      return 1; }
    return 0
}

RELAUNCH_COUNT=0
LAST_PENDING=-1

while true; do
    # Guard: stop if we've hit the relaunch limit
    if [ "$RELAUNCH_COUNT" -ge "$MAX_RELAUNCHES" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Reached max relaunches ($MAX_RELAUNCHES) without completing all tasks."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check tracker.json and nighttime.log. Run repairrun.sh if tasks are stuck."
        exit 1
    fi

    # Install nighttime settings, mode rules, and hooks before each launch.
    # This ensures correct settings even if a prior dayrun session left them in daytime state.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing nighttime settings (relaunch $((RELAUNCH_COUNT + 1))/$MAX_RELAUNCHES)..."
    if ! install_nighttime_files; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FATAL: Could not install nighttime files. Aborting."
        exit 1
    fi

    # Back up tracker.json before each launch — protects against mid-write corruption
    if [ -f "$TRACKER" ]; then
        cp "$TRACKER" "${TRACKER}.bak" 2>/dev/null
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Claude nighttime session..."

    # Launch Claude non-interactively with the nighttime prompt.
    # --print runs a single non-interactive session with the given message as the first user turn.
    # Non-zero exit is expected (usage cap) — log it and continue.
    claude --dangerously-skip-permissions --max-turns "$MAX_TURNS" $MODEL_FLAG --print -n "$SESSION_NAME" "$NIGHTTIME_PROMPT" \
        || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude exited with non-zero status (expected on usage cap)."

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude session ended. Checking tracker..."

    PENDING=$(count_pending)
    RELAUNCH_COUNT=$((RELAUNCH_COUNT + 1))

    if [ "$PENDING" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks complete (or no tracker found). Exiting."
        break
    fi

    # Detect if we're making no progress — same pending count two sessions in a row.
    # This catches Claude crashing immediately every time without doing any work.
    if [ "$PENDING" -eq "$LAST_PENDING" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: No progress since last session ($PENDING task(s) still pending)."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude may be crashing on startup. Continuing — will abort after $MAX_RELAUNCHES total relaunches."
    fi

    LAST_PENDING=$PENDING
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${PENDING} task(s) remaining. Waiting ${COOLDOWN}s before resuming..."
    sleep "$COOLDOWN"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] nightrun.sh finished."
echo ""
echo "========================================"
echo "  NIGHTRUN SUMMARY"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total relaunches: $RELAUNCH_COUNT"
echo "========================================"

if [ -f "$TRACKER" ]; then
    $PYTHON -c "
import json
try:
    with open('${TRACKER}') as f:
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
            print(f'    {t[\"task_id\"]}: {t.get(\"description\", \"\")} — branch: {branch}{flag_str}')
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
" 2>/dev/null || echo "  (Could not read tracker.json)"
else
    echo "  No tracker.json found."
fi

echo "========================================"
