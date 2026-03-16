---
name: pre-commit-review
description: Self-review code changes before committing — checks for quality issues, spec compliance, and common mistakes
---

# Pre-Commit Review

Review all pending changes against the task spec before committing. Catches spec violations,
quality issues, and common mistakes that would otherwise require morning cleanup.

This skill is usable by both day and night agents but is **primarily designed for nighttime
use** — it runs between implementation (Step 7) and commit (Step 8) of the task loop.

---

## Step 1 — Gather changes

Run `git diff` to see unstaged changes and `git diff --cached` to see staged changes.
Combine both into the full picture of what is about to be committed.

If there are no changes at all, log a note and return — nothing to review.

---

## Step 2 — Read the spec

Read `spec.md` for the current task (located in the task's spec directory under
`DaytimeNighttimeHandOff/WrittenByDaytime/`). Parse out:

- **Requirements:** What the spec explicitly asked for
- **Acceptance criteria:** Any specific conditions, edge cases, or constraints mentioned
- **Out of scope:** Anything the spec explicitly excluded or deferred

Keep these in mind for every check that follows.

---

## Step 3 — Spec compliance check

Compare the implementation against the spec:

- **Missing requirements:** Is anything the spec asked for not implemented? Check every
  requirement individually — partial implementations count as missing.
- **Extra work:** Are there changes that go beyond what the spec requested? Unrequested
  features, refactors of unrelated code, or scope creep.
- **Misinterpretations:** Does the implementation do what the spec meant, not just what
  it literally said? Watch for subtle misreadings.

Severity:
- Missing requirement → BLOCKER
- Significant extra work → WARNING
- Minor extra cleanup (e.g., fixing a typo nearby) → NOTE

---

## Step 4 — Code quality check

Review all changed code for:

- **DRY violations:** Duplicated logic across the changed files, or new code that
  duplicates existing code elsewhere in the project.
- **Overly complex functions:** Functions longer than ~30 lines or with deep nesting
  (3+ levels). Consider whether they should be split.
- **Magic numbers/strings:** Unnamed constants or repeated string literals that should
  be extracted into named constants.
- **Missing type hints:** All new functions should have parameter and return type hints.
- **Missing docstrings:** All new public functions should have docstrings.
- **Poor abstractions:** Classes or functions that do too much or too little.

Severity:
- Major DRY violation (whole blocks copied) → WARNING
- Missing type hints or docstrings → WARNING
- Minor style issues → NOTE

---

## Step 5 — Common mistakes check

Scan for patterns that frequently cause bugs:

- **Off-by-one errors:** Loop bounds, slice indices, range() calls, fence-post conditions.
- **Unclosed resources:** Files, database connections, network sockets opened without
  `with` statements or explicit close in a `finally` block.
- **Missing error handling:** I/O operations, network calls, file operations, or
  external command invocations without try/except at the boundary.
- **Mutable default arguments:** `def foo(items=[])` or `def foo(config={})` — these
  share state across calls.
- **Incorrect None checks:** `if x` when `if x is not None` is needed (0, empty string,
  and empty list are falsy but not None).
- **Silent failures:** Bare `except:` or `except Exception: pass` that swallow errors
  without logging.

Severity:
- Unclosed resources, mutable defaults, silent failures → BLOCKER
- Missing error handling at boundaries → WARNING
- Potential off-by-one (uncertain) → WARNING

---

## Step 6 — Test quality check

Review all new or modified tests:

- **Tautological tests:** Tests that would pass even if the code under test was deleted
  or returned a wrong value. Every assertion should fail if the behavior breaks.
- **Missing edge cases:** Check the spec for boundary conditions, empty inputs, error
  paths, and large inputs. Are they tested?
- **Missing assertions:** Tests that call functions but don't assert anything meaningful
  about the result.
- **Brittle tests:** Tests that depend on exact formatting, timestamps, or other
  non-deterministic values without tolerance.
- **Test isolation:** Tests that depend on execution order or shared mutable state.

Severity:
- Tautological test (always passes) → BLOCKER
- Missing edge case from spec → WARNING
- Brittle or poorly isolated test → WARNING
- Minor assertion gaps → NOTE

---

## Step 7 — Security check

Scan all changed code for:

- **SQL injection:** String concatenation or f-strings used to build SQL queries instead
  of parameterized queries.
- **Command injection:** User-controlled or external input passed to `os.system`,
  `subprocess.run(shell=True)`, or `eval`/`exec`.
- **Path traversal:** User input used in file paths without sanitization — check for
  `..` traversal, absolute path injection.
- **Hardcoded secrets:** API keys, passwords, tokens, or credentials in source code.
- **Unsafe deserialization:** `pickle.loads`, `yaml.load` (without SafeLoader), `eval`
  on untrusted data.

Severity:
- Any security issue → BLOCKER

---

## Step 8 — Naming check

Review variable, function, class, and file names in the changed code:

- Are names descriptive and unambiguous?
- Do they follow the existing codebase conventions (snake_case, camelCase, etc.)?
- Are abbreviations consistent with what the rest of the project uses?
- Do similar concepts use similar naming patterns?

Severity:
- Misleading name (says one thing, does another) → WARNING
- Inconsistent convention → NOTE
- Could be slightly clearer → NOTE

---

## Step 9 — Produce findings summary

Compile all findings into a structured summary:

```
## Pre-Commit Review: <task-id>

### BLOCKERs (<N>)
- [BLOCKER] <category>: <description>
  File: <path>:<line>
  Fix: <what to do>

### WARNINGs (<N>)
- [WARNING] <category>: <description>
  File: <path>:<line>

### NOTEs (<N>)
- [NOTE] <category>: <description>
```

If there are no findings in a severity level, omit that section entirely.

---

## Step 10 — Act on findings

Based on the highest severity found:

**If BLOCKERs exist:**
1. Fix every BLOCKER issue
2. Re-run the full test suite to confirm fixes don't break anything
3. Return to Step 1 and re-review the updated changes
4. Repeat until no BLOCKERs remain (max 3 cycles — if BLOCKERs persist after 3 rounds,
   log them as flags and proceed with a warning in `result.md`)

**If only WARNINGs exist (no BLOCKERs):**
1. Fix any WARNING that takes less than ~2 minutes
2. For remaining WARNINGs, add each to the `## Flags` section of `result.md`:
   ```
   - [WARNING] <category>: <description> — deferred to morning review
   ```
3. Proceed to commit

**If only NOTEs exist (no BLOCKERs or WARNINGs):**
1. Commit as-is — no changes needed
2. Optionally mention NOTEs in `result.md` under a `## Notes` section

---

## Step 11 — Log findings

Append the review summary to `nighttime.log`:

```
[<ISO timestamp>] REVIEW <task-id> — <N> blockers, <N> warnings, <N> notes
```

If BLOCKERs were found and fixed, also log:
```
[<ISO timestamp>] REVIEW <task-id> — blockers resolved after <N> fix cycles
```

---

## Done

Return to the calling workflow. The changes are now reviewed and ready to commit
(or have been flagged appropriately in `result.md` for morning review).
