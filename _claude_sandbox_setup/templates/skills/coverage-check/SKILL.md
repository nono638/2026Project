---
name: coverage-check
description: Analyze test coverage using pytest-cov — identify untested critical paths and flag coverage gaps for morning review
---

# Coverage Check

Analyze test coverage, identify untested critical paths, and produce a report for morning review. This skill is **diagnostic only** — it does not fix coverage gaps.

---

## Steps

### 1. Ensure pytest-cov Is Installed

Run:

```bash
pip show pytest-cov
```

- If installed, proceed to Step 2.
- If **not** installed, read `.claude/skills/install-package/SKILL.md` and use the install-package skill to add `pytest-cov`.
- Do **NOT** install globally. All installs must target the project's virtualenv.

### 2. Run Coverage Analysis

```bash
python -m pytest --cov=src --cov-report=term-missing --cov-report=html -q
```

Before running, determine the correct `--cov` target:

- Check for `src/`, `app/`, or a project-name-as-package directory.
- If the source layout is unclear, fall back to `--cov=.` and exclude test files (`--cov-config=.coveragerc` or `--ignore=tests/`).
- Adjust the command accordingly — the goal is to cover all production source code and nothing else.

### 3. Parse Results

From the `term-missing` output, extract for each source file:

- **Coverage percentage**
- **Missing line numbers** (the `Missing` column)

Categorize every file into one of three buckets:

| Category | Threshold | Action |
|---|---|---|
| **Critical gaps** | < 50% coverage | List specific uncovered functions/methods by reading the source at the missing line ranges |
| **Moderate gaps** | 50–80% coverage | Note what categories of code are missing |
| **Well covered** | > 80% coverage | No action needed |

### 4. Identify High-Risk Uncovered Code

Among the missing lines, flag anything that falls into these categories:

- **Error handling / exception paths** — try/except blocks, error-return branches
- **Security-related code** — authentication, authorization, input validation, sanitization
- **Data mutation** — writes, deletes, updates to databases or files
- **External integrations** — API calls, file I/O, database queries, network requests

These are the most dangerous things to leave untested. Read the source files at the missing line ranges to determine which category each gap falls into.

### 5. Write a Coverage Report

**If running during nighttime**, write the report to `WrittenByNighttime/coverage-report.md`:

```markdown
# Test Coverage Report — YYYY-MM-DD

## Summary
- Overall coverage: XX%
- Files analyzed: N
- Critical gaps: N files below 50%

## Critical Gaps
- `path/to/file.py` (XX%) — missing: [function names or line ranges]

## High-Risk Uncovered Code
- `path/to/file.py:42-58` — error handling for [what]
- `path/to/file.py:100-115` — input validation for [what]

## Recommendations
- [Specific tests that should be written, prioritized by risk]
```

**If running during daytime**, show the summary to the user and ask if they want to create specs for the gaps.

### 6. Log Results

**Nighttime only** — append a single line to `nighttime.log`:

```
[<ISO timestamp>] COVERAGE — overall XX%, <N> critical gaps, <N> high-risk uncovered paths
```

### 7. Do Not Fix Coverage

This skill is diagnostic. Do **not** write tests or modify source code to improve coverage unless that work is explicitly part of a separate task spec. Coverage gaps should flow through the normal pipeline:

> spec → tracker → night implementation

---

## Done

This skill is complete when:

- [ ] pytest-cov ran successfully against the project's source
- [ ] Every source file is categorized (critical / moderate / well covered)
- [ ] High-risk uncovered code is identified and labeled by category
- [ ] Report is written (nighttime) or shown to user (daytime)
- [ ] Results are logged to nighttime.log (nighttime only)
