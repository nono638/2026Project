---
name: protocol-fix
description: Log and fix systemic issues with the day/night workflow — updates both active and template copies of affected files
---

# Protocol Fix

Use this skill when a nighttime flag, observation, or user report reveals a systemic issue
with the day/night workflow itself (not a project bug). Examples: missing process steps,
confusing output, protocol gaps, file sync issues.

This skill is **daytime only** — it requires judgment about root cause and fix design.

---

## Step 1 — Confirm it's a protocol issue

Not every nighttime flag is a protocol issue. Ask:
- Did the problem come from a gap in the day/night process? (yes = protocol issue)
- Or was it a bug in project code, a missing dep, or a one-off mistake? (no = not this skill)

If it's not a protocol issue, handle it normally and stop here.

---

## Step 2 — Diagnose the root cause

Don't just fix the symptom. Trace backward:
- What did the night agent actually do?
- What should it have done instead?
- Which instruction (or missing instruction) caused the wrong behavior?
- Would this happen again on a different task?

Write down the root cause in one sentence before proceeding.

---

## Step 3 — Log the issue

Add an entry to `DaytimeNighttimeHandOff/DaytimeOnly/reference/day-night-system-issues.md`
under the "Open / Watch List" section:

```markdown
### Short descriptive title
**Found:** YYYY-MM-DD
**Severity:** blocks workflow | causes confusion | minor
**Description:** What happens and why (root cause from Step 2)
**Fix:** What was changed and where
**Transposed to main repo:** no
```

---

## Step 4 — Identify affected files

The day/night system has paired files: active copies (used at runtime) and template copies
(the canonical source in `_claude_sandbox_setup/`). **Both must be updated.**

Here is the complete map of file pairs:

### Supplements (mode instructions)
| Active copy | Template copy |
|---|---|
| `.claude/active_mode.md` (when in day mode) | `_claude_sandbox_setup/templates/daytime_supplement.md` |
| `.claude/active_mode.md` (when in night mode) | `_claude_sandbox_setup/templates/nighttime_supplement.md` |

### Skills
| Active copy | Template copy |
|---|---|
| `.claude/skills/<name>/SKILL.md` | `_claude_sandbox_setup/templates/skills/<name>/SKILL.md` |

### Settings
| Active copy | Template copy |
|---|---|
| `.claude/settings.json` (when in night mode) | `_claude_sandbox_setup/templates/nighttime_settings.json` |

### Scripts (no active copy — only one location)
| File | Location |
|---|---|
| `nightrun.bat` | `_claude_sandbox_setup/scripts/nightrun.bat` |
| `nightrun.sh` | `_claude_sandbox_setup/scripts/nightrun.sh` |
| `nightrun_helper.py` | `_claude_sandbox_setup/scripts/nightrun_helper.py` |
| `dayrun.bat` | `_claude_sandbox_setup/scripts/dayrun.bat` |
| `dayrun.sh` | `_claude_sandbox_setup/scripts/dayrun.sh` |

### Hooks (active copies are NOT committed — gitignored)
| Active copy | Template copy |
|---|---|
| `.claude/hooks/<name>.py` | `_claude_sandbox_setup/hooks/<name>.py` |

For each file you need to change, check this table and update **both copies**.
If you update a template but forget the active copy (or vice versa), the fix won't
take effect until the next mode switch.

---

## Step 5 — Make the fix

Edit the affected files. For each change:
1. Make the change in the **template copy** first (canonical source)
2. Make the identical change in the **active copy**
3. If you're changing a `.sh` or `.bat` script, check whether both the `.sh` and `.bat`
   versions need the same change

---

## Step 6 — Verify consistency

After all edits, spot-check that the active and template copies are in sync for any
file you touched. A quick diff or read of the changed sections is sufficient.

---

## Step 7 — Notify user about transposing

Tell the user:
> "Protocol fix applied to [N] files. This needs to be transposed to the main
> `_claude_sandbox_setup` repo — it's logged in `day-night-system-issues.md`."

---

## Done

Return to the calling context.
