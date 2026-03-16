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

## Step 5 — Write the spec section by section

Build the spec incrementally. Present each section to the user for validation before
moving to the next. This catches misunderstandings early instead of at the end.

Create `DaytimeNighttimeHandOff/WrittenByDaytime/task-NNN-short-name/spec.md` and build
it section by section:

**Section 1 — Summary and Requirements:**
Write the summary paragraph and numbered requirements. Show the user:
> "Here's what I have for the core requirements:
> 1. [requirement]
> 2. [requirement]
> Anything missing or wrong?"

Wait for confirmation before proceeding.

**Section 2 — Files to Modify:**
List exact files with specific functions/classes/sections. Show the user:
> "These are the files I'd have the night agent touch:
> - `path/to/file.py` — modify `function_name()` to...
> - `path/to/new_file.py` — create, purpose...
> Look right?"

**Section 3 — Edge Cases and Decisions:**
List edge cases with explicit expected behaviors, and all decisions with rationale. Show:
> "Edge cases and decisions I've locked in:
> - When [scenario]: [behavior]
> - Decision: [choice] because [why]
> Any edge cases I'm missing?"

**Section 4 — Acceptance Tests:**
Write concrete test descriptions with specific inputs and expected outputs. Show:
> "Here's what the night agent will test against:
> - `function(input)` returns `expected_output`
> - `function(bad_input)` raises `SpecificError`
> These cover the requirements?"

**Section 5 — What NOT to Touch:**
List boundaries. Show briefly, confirm.

After all sections are validated, assemble the complete spec:

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

## Edge Cases
- [Explicit scenario → expected behavior]
- [...]

## Decisions Made
- [Decision]: [choice]. **Why:** [rationale]. [URL if researched]
- [...]

## What NOT to Touch
- [File or area that should be left alone, and why]

## Acceptance Tests
- [Concrete test with specific input → expected output]
- [...]

## Testing Approach
- [What the pre-written tests cover]
- [How to run them]
```

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

## Step 7 — Automated spec review

Before finalizing, review the spec as if you were the night agent receiving it. Ask yourself
these questions and fix any failures:

**Completeness check — could an unattended agent implement this without asking a single question?**

| Check | Pass? | Fix if failing |
|---|---|---|
| Every requirement is testable (has a concrete pass/fail condition) | | Rewrite vague requirements |
| Every file to modify lists the specific function/class/section | | Add specifics |
| No vague language: "handle appropriately", "as needed", "etc." | | Replace with explicit behavior |
| Every decision has a rationale (not just "use X" but "use X because Y") | | Add the why |
| Acceptance tests have concrete inputs and expected outputs | | Add specific examples |
| Edge cases have explicit expected behavior (not just "consider X") | | State what should happen |
| Scope is clear — what NOT to touch is stated | | Add boundaries |
| No unresolved questions or TODOs in the spec | | Resolve them now or ask the user |
| The task is completable in one nighttime session | | Split if too large |

**Ambiguity scan — search the spec text for red flags:**
- "should" (weak — change to "must" or remove)
- "might", "could", "possibly" (unresolved decisions — resolve them)
- "appropriate", "reasonable", "as needed" (vague — replace with specifics)
- "TBD", "TODO", "TBC" (unfinished — finish them)
- "similar to", "like the existing" (which existing? name it)

If you find issues, fix them in the spec and re-show the affected section to the user
for confirmation.

**Final quality gate:**
- [ ] Exact files listed with specific sections
- [ ] All edge cases explicit
- [ ] All judgment calls made — zero ambiguity
- [ ] "Why" answered for every non-obvious decision
- [ ] Research URLs included where research was done
- [ ] Acceptance tests have concrete inputs → expected outputs
- [ ] Scoped to one focused nighttime session
- [ ] No vague language found in ambiguity scan

If any item fails, go back and fix it before continuing.

---

## Step 8 — Create tracker.json entry

Add to `DaytimeNighttimeHandOff/tracker.json`:

```json
{
  "task_id": "task-NNN",
  "description": "<brief description>",
  "daytime_created": "<ISO timestamp>",
  "daytime_comments": "<any extra context not in the spec>",
  "depends_on": null,
  "status": "todo"
}
```

If this task depends on another, set `depends_on` to an array: `["task-001"]`.

---

## Step 9 — Handle incubating.md (if promoting)

If this task was promoted from incubating.md:
1. In `incubating.md`, mark the entry: `**Promoted to:** task-NNN`
2. Move the entry to `DaytimeNighttimeHandOff/DaytimeOnly/archive/`

---

## Step 10 — Confirm to user

Summarize:
> "task-NNN queued: <description>. Spec has N requirements, K test files, covers M files.
> Ready for tonight's run."

---

## Done

Return to the calling supplement.
