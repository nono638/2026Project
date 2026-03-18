---
name: spec-writer
description: Guided spec creation for nighttime tasks — requirements, file identification, judgment calls, tests, tracker entry
---

# Spec Writer

Walk through a structured process to create a complete, unambiguous spec for a nighttime
task. This skill is for **daytime use only** — it requires user input to resolve ambiguities.

If running in nighttime mode, skip this skill and log a warning to nighttime.log.

---

## Step 1 — Determine the task ID

Read `DaytimeNighttimeHandOff/tracker.json`.

Find the highest existing `task_id` number (parse `task-NNN` from each entry). The new task
is `task-<N+1>`. If tracker is empty, start at `task-001`.

Generate a short kebab-case name from the description (e.g., `add-caching-layer`,
`fix-login-redirect`). Keep it under 30 characters.

The task directory: `DaytimeNighttimeHandOff/WrittenByDaytime/task-NNN-short-name/`

---

## Step 2 — Gather requirements

**If promoting from incubating.md:** Read the incubating entry for context and trigger
condition. Use that as the starting point.

**If from conversation:** Work with the user to define:
- What exactly should be built, changed, or fixed?
- What is the expected behavior when it works?
- What are the edge cases?

Ask structured questions **one at a time** to fill gaps:
- "What should happen when [edge case]?"
- "Should this handle [scenario] or is that out of scope?"
- "Any constraints on [approach/library/pattern]?"

Don't over-ask — if the answer is obvious from context, decide and note it.

---

## Step 3 — Identify exact files

Explore the codebase to determine which files need modification:
- List every file that will be **created** (new files)
- List every file that will be **modified** (with the specific functions/classes/sections)
- List files that will be **read** for context but not changed

For modifications, note the specific section: "In `src/auth.py`, modify the `login()`
function to add..." — not just "modify src/auth.py."

If research is needed (which library, which API, which pattern), use WebSearch/WebFetch
now and record the URLs for the spec's rationale section.

---

## Step 4 — Make judgment calls

Identify every ambiguous decision the night instance would face. Resolve each one
explicitly with the user.

Common decisions to resolve:
- Library choice (why this one over alternatives?)
- Data structure or schema design
- Error handling strategy (fail fast? retry? fallback?)
- Naming conventions
- API shape (endpoints, function signatures)
- What's in scope vs. out of scope

For each decision, write a one-line rationale: why this choice over the alternatives.

---

## Step 5 — Write the spec

Create `DaytimeNighttimeHandOff/WrittenByDaytime/task-NNN-short-name/spec.md`:

```markdown
# task-NNN: <Short Description>

## Summary
[One paragraph — what this task does and why it exists]

## Requirements
1. [Must be true when done — specific, testable]
2. [...]

## Files to Modify
- `path/to/file.py` — [what changes, which functions/classes]
- `path/to/new_file.py` — [create, purpose]

## New Dependencies
- `package-name==x.y.z` — [why needed, what it provides]
- (or "None — all required packages are already installed")

## Edge Cases
- [Explicit scenario → expected behavior]
- [...]

## Decisions Made
- [Decision]: [choice]. **Why:** [rationale]. [URL if researched]
- [...]

## What NOT to Touch
- [File or area that should be left alone, and why]

## Testing Approach
- [What the pre-written tests cover]
- [How to run them]
```

**Quality gate — check each before proceeding:**
- [ ] Exact files to modify are listed with specific sections
- [ ] All edge cases are explicit (not "handle appropriately")
- [ ] All judgment calls are made — no ambiguity for the night instance
- [ ] "Why" is answered for every non-obvious decision
- [ ] Research URLs included where research was done
- [ ] New dependencies identified and installed in venv (see Step 5b)
- [ ] Scoped to one focused nighttime session

If any item fails, go back and fix it before continuing.

---

## Step 5b — Install new dependencies

If the spec lists new dependencies in the "New Dependencies" section, install them now
using the install-package skill: read `.claude/skills/install-package/SKILL.md` and follow
all instructions.

This must happen during the daytime session — before the task is queued for nighttime.
The night agent can install packages, but doing it during the day lets you troubleshoot
pip errors immediately and avoids wasting unattended time.

If the spec says "None", skip this step.

---

## Step 6 — Write pre-written tests

Create `DaytimeNighttimeHandOff/WrittenByDaytime/task-NNN-short-name/tests/` directory.

Write test files that:
- Cover the key behaviors from the requirements (not just "does it run")
- Are runnable with `pytest` (or the project's test runner)
- Will **fail** before implementation (they test the expected outcome)
- Are named to match source files: `test_<source_file>.py`

Include at least:
- One test per requirement
- Edge case tests for the scenarios listed in the spec
- A regression test if this is a bug fix

---

## Step 7 — Create tracker.json entry

Add to `DaytimeNighttimeHandOff/tracker.json`:

```json
{
  "task_id": "task-NNN",
  "description": "<brief description>",
  "daytime_created": "<actual wall-clock time — run: python -c \"from datetime import datetime; print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))\" — do NOT estimate>",
  "daytime_comments": "<any extra context not in the spec>",
  "depends_on": null,
  "status": "todo"
}
```

If this task depends on another, set `depends_on` to an array: `["task-001"]`.

---

## Step 8 — Handle incubating.md (if promoting)

If this task was promoted from incubating.md:
1. In `incubating.md`, mark the entry: `**Promoted to:** task-NNN`
2. Move the entry to `DaytimeNighttimeHandOff/DaytimeOnly/archive/`

---

## Step 9 — Confirm to user

Summarize:
> "task-NNN queued: <description>. Spec has N requirements, K test files, covers M files.
> Ready for tonight's run."

---

## Done

Return to the calling supplement.
