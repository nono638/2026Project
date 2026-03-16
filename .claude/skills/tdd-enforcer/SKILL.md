---
name: tdd-enforcer
description: Enforce strict red-green-refactor TDD — one failing test at a time, minimal implementation, then next test. Prevents bulk test writing.
---

# TDD Enforcer

This skill enforces strict test-driven development using a red-green-refactor vertical slicing approach during the nighttime implementation phase. Every behavior is proven by a failing test before implementation code is written.

---

## 1. Planning Phase (Before Any Code)

Before writing a single line of test or implementation code:

1. **Read the spec's Acceptance Tests section** in full. Understand every expected behavior.
2. **Order the tests** from simplest/most fundamental to most complex. foundational behaviors first (e.g., "constructor creates an instance") before dependent behaviors (e.g., "processes a batch of items").
3. **For each test, identify:**
   - What interface (function, method, class) it requires
   - What single behavior it validates
   - What inputs and expected outputs define that behavior
4. **Write a brief test plan** to `plan.md` in the task working directory, listing the tests in order. Example:

```
## Test Plan — <task-id>

1. creates instance with default config
2. accepts custom config overrides
3. rejects invalid config keys
4. processes single item
5. processes batch of items
6. returns error for malformed input
7. respects rate limit
...
```

---

## 2. The Cycle (Repeat for Each Test)

Work through the test plan one test at a time. Do not skip ahead.

### 2a. RED — Write One Failing Test

1. Write **exactly one test function** that tests **one behavior**.
2. Run it. It **must fail**.
3. If it passes unexpectedly:
   - The behavior already exists — verify, then mark the test as covered and move to the next.
   - Or the test is wrong — fix the assertion so it actually checks the intended behavior.
4. If it fails for the **wrong reason** (import error, syntax error, missing file):
   - Fix only the test scaffolding (imports, file creation, boilerplate).
   - Do **not** write any implementation logic.
   - Re-run. It should now fail for the **right reason** (assertion failure on the expected behavior).

### 2b. GREEN — Minimal Implementation

1. Write the **minimum code** to make that one test pass.
2. Do **not** write code for future tests. Only what is needed right now.
3. Hardcoded return values are acceptable if they satisfy the current test — the next test will force generalization.
4. Run **all tests written so far**. Every one must pass.
5. If any previous test breaks, fix the regression **before** moving on.

### 2c. REFACTOR — Clean Up Periodically

After every 3-5 GREEN passes, or whenever code smells accumulate:

1. Look for duplication, unclear naming, overly long functions, tangled logic.
2. Refactor the implementation and/or tests.
3. Run **all tests** — they must all still pass after refactoring.
4. Commit the refactored state:

```bash
git add -A && git commit -m "night: <task-id> refactor after <N> tests"
```

---

## 3. What NOT To Do

- **Don't write all tests at once then implement** (horizontal slicing). This defeats the feedback loop.
- **Don't write a test that tests multiple behaviors.** One test, one assertion, one behavior.
- **Don't write implementation code before the test for it exists.** The test comes first, always.
- **Don't mock internal implementation details.** Test through public interfaces only. Mocks are for external dependencies (network, filesystem, third-party APIs).
- **Don't rewrite a failing test to make it pass.** Fix the implementation instead. The test defines the contract.

---

## 4. When To Skip TDD

TDD is not required for:

- **Pure configuration files** — no behavior to test (e.g., JSON config, YAML manifests).
- **Simple data classes with no logic** — plain structs/dataclasses with no methods beyond accessors.
- **When the spec explicitly says "no tests needed"** for a given component.

When skipping, add a comment in `plan.md`:

```
- (skipped) config.json — pure configuration, no behavior
```

---

## 5. Logging

After completing all tests and implementation for a task, append a summary line to the night log:

```
[<ISO timestamp>] TDD <task-id> — <N> tests written, <N> refactor cycles, all passing
```

---

## Done

The TDD enforcer cycle is complete when:

- Every test in the plan has been written and passes.
- All refactoring rounds are committed.
- No test is skipped without documented justification.
- The summary log line has been written.
