---
name: end-of-night-sweeps
description: Codebase-wide quality sweeps run after all nighttime tasks are complete — tests, bugs, DRY, type hints, dead code, security
---

# End-of-Night Sweeps

Run these codebase-wide sweeps before ending the nighttime session. Each sweep gets its own
branch and follows the same commit discipline as regular tasks.

**Skip sweeps entirely if:**
- There are no source files in the project yet (nothing to sweep)
- Every queued task was `skipped` or `blocked` (the codebase didn't change tonight)
- The context monitor hook has fired a WARNING or CRITICAL alert (preserve remaining context)

**Sweep rules:**
- Create a branch for each sweep: `night/sweep-<name>`
- Run the full test suite after each sweep to confirm nothing broke
- If a sweep's changes cause test failures, revert the failing changes and commit only what's safe
- Log each sweep to `nighttime.log` like a regular task
- Do NOT add tracker.json entries for sweeps — they are automatic, not queued tasks
- Follow the 3-attempt rule: if a sweep keeps breaking things, skip it and log why
- Apply the same code quality standards (type hints, docstrings, comments) to all code you touch
- Use subagents for investigation when scanning many files — keep the main context clean for fixes

## Sweep 1 — Full test suite run and fix

Run every test in the project. For each failure:
1. Read the test to understand what it expects
2. Read the source code it tests
3. Fix the source code (not the test) unless the test itself is clearly wrong
4. Re-run to confirm the fix
5. Commit each fix individually with a clear message: `sweep: fix <what failed>`

If a test failure is ambiguous or the fix would be a large change, log it in
`nighttime.log` as a flag for morning review instead of attempting a risky fix.

## Sweep 2 — Bug sweep

Scan every source file in the project for common bugs:
- Unhandled exceptions in I/O, network, or file operations
- Off-by-one errors in loops and slicing
- Incorrect None/null checks (e.g., `if x` vs `if x is not None`)
- Resource leaks (files, connections, sockets not closed)
- Race conditions or unsafe shared state
- Hardcoded values that should be constants or config
- Logic errors (wrong operator, inverted condition, unreachable code)
- Missing return statements or inconsistent return types

Fix what you find. Commit each logically grouped fix separately. If a fix is uncertain or
could change behavior, log it as a flag rather than applying it.

## Sweep 3 — DRY violations and class extraction

Look for duplicated or near-duplicated code across the project:
- Repeated logic blocks that appear 3+ times → extract into a function
- Related functions operating on the same data → consider grouping into a class
- Copy-pasted code with minor variations → parameterize into a single function
- Magic strings or numbers repeated across files → extract into named constants

When extracting, ensure:
- The new abstraction has a clear name and docstring
- All call sites are updated
- Tests still pass after refactoring
- The abstraction is genuinely simpler than the duplication (don't over-abstract)

## Sweep 4 — Type hints and docstrings

Scan all source files for functions missing type hints or docstrings:
- Add type hints (parameters + return type) to every function that lacks them
- Add docstrings to every function that lacks them
- For existing docstrings that are incomplete, add missing Args/Returns/Raises sections
- Add `from __future__ import annotations` at the top of files if needed for forward refs
- Do NOT modify test files unless their helper functions are missing annotations

## Sweep 5 — Dead code and import cleanup

Remove code that serves no purpose:
- Unused imports
- Unused variables and functions (verify with grep before removing)
- Commented-out code blocks with no explanation (if a comment says why it's kept, leave it)
- Unreachable code after return/raise/break/continue
- Empty except blocks with no logging or comment

## Sweep 6 — Security scan

Check for common security issues:
- SQL injection (string concatenation in queries)
- Command injection (unsanitized input in shell commands)
- Path traversal (user input used in file paths without validation)
- Hardcoded secrets, API keys, or passwords
- Insecure defaults (debug mode on, CORS *, no auth)
- Unsafe deserialization (pickle.loads, eval, exec on untrusted input)

Flag anything found in `nighttime.log`. Fix what's clearly fixable. For anything that
needs human judgment (e.g., "is this debug mode intentional?"), log it as a flag.

## After all sweeps

1. Checkout `main`
2. Append a sweep summary to `nighttime.log`:
   ```
   [<timestamp>] SWEEPS COMPLETE: <N> sweeps run, <M> fixes applied, <K> flags for review
   ```
3. Then end the session normally.
