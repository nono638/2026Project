---
name: intake
description: Discovery and requirements gathering — asks pointed questions, routes information to the right files, pulls toward spec creation
---

# Intake — Discovery and Requirements Gathering

A structured conversation workflow for daytime sessions. Use this when the user shares a
new idea, feature request, bug report, or direction change — before jumping to implementation
or spec writing.

If running in nighttime mode, skip this skill entirely. Log to nighttime.log and continue.

---

## Your mindset

You are a technical PM having an efficient conversation. Your goals:
1. **Understand** what the user wants and why
2. **Route** each piece of information to the right file as you learn it
3. **Pull toward a spec** — always be thinking "do I have enough to write a spec yet?"
4. **Don't over-ask** — if the answer is obvious from context, decide and note it

---

## Step 1 — Orient

Before asking anything, silently read:
- `DaytimeNighttimeHandOff/DaytimeOnly/project_overview.md` — what exists already
- `DaytimeNighttimeHandOff/tracker.json` — what's already queued or done
- `DaytimeNighttimeHandOff/DaytimeOnly/reference/architecture-decisions.md` (if it exists)

Use this context to ask smarter questions. Don't ask things the project overview already answers.

---

## Step 2 — Explore the problem space

Start open-ended. The user may have a vague idea or a concrete request. Match their energy:

**If they're exploring (vague, high-level):**
- Ask about the *problem* before the solution: "What's the pain point here?"
- Ask about users/audience: "Who encounters this? How often?"
- Ask about constraints: "Anything we can't change or must work with?"
- Let the conversation breathe. Don't rush to specifics.

**If they're specific (concrete feature or bug):**
- Jump to implementation questions faster. They already know what they want.
- Use multiple choice to narrow decisions efficiently.

**Read the room.** If the user gives long, thoughtful answers, they want to think out loud —
listen and reflect back. If they give short answers, they want to move fast — match that pace.

---

## Step 3 — Narrow with pointed questions

As the conversation progresses, shift from open-ended to specific. Use multiple choice
for implementation decisions:

> **A** *(recommended)* — [option with brief rationale]
> **B** — [alternative with brief rationale]
> **C** — [if applicable]
> Or describe what you want.

Categories of questions to work through (not all will apply every time):

### What — behavior and scope
- What exactly should this do when it works?
- What's the expected input → output?
- What's explicitly OUT of scope?
- Any existing code this interacts with?

### Why — rationale and constraints
- Why now? What triggered this?
- Why this approach over simpler alternatives?
- Any prior art or industry patterns to follow?

### How — implementation details (use multiple choice here)
- Which library/framework/pattern?
- What data structures?
- Error handling strategy?
- API shape / function signatures?

### Tests — acceptance criteria
- How do we know it works? (concrete examples with inputs/outputs)
- What are the edge cases?
- What should it NOT do? (negative tests)
- Performance requirements? (if applicable)

**One question at a time.** Let the user answer before asking the next.

---

## Step 4 — Route information as you go

Don't accumulate context in your head. As you learn things, write them immediately:

| What you learned | Where it goes | When to write |
|---|---|---|
| Project goals, vision, scope changes | `project_overview.md` | Immediately — update the relevant section |
| Architectural decisions with rationale | `reference/architecture-decisions.md` | Immediately — append with date |
| Research findings, tool evaluations | `reference/research.md` | Immediately — append with date and URLs |
| Known bugs or limitations | `reference/known-issues.md` | Immediately — append |
| "Maybe later" ideas | `incubating.md` | Immediately — with a next trigger |
| Random asides | `inbox.md` | Immediately — with date and one line of context |

Don't wait until you have a complete picture. Route partial information now — you can
always update it later.

**Tell the user when you route something:**
> "Noted — I've added that to architecture-decisions.md as a constraint."
> "Good point — capturing that edge case. It'll go in the spec's test section."

This builds trust and keeps them in the loop without slowing down.

---

## Step 5 — Check: ready to spec?

After each answer, silently ask yourself: **Do I have enough to write a spec?**

A spec is ready when you can answer YES to all of these:
- [ ] I know exactly what to build (behavior, not just concept)
- [ ] I know which files to create or modify
- [ ] I've resolved all ambiguous decisions (or the user said "your call")
- [ ] I can write concrete acceptance tests (specific inputs → expected outputs)
- [ ] It's scoped to one focused nighttime session

**If YES:** Tell the user:
> "I think I have enough to write this up. Let me draft the spec — I'll show you
> before finalizing."
Then hand off to the spec-writer skill: read `.claude/skills/spec-writer/SKILL.md`.

**If NO:** Identify the specific gap and ask about it. Be transparent:
> "I think I'm close to speccing this, but I need to know [specific thing] first."

**If the conversation is still exploratory and a spec isn't close:**
Don't force it. Some conversations need more time. Capture what you've learned so far
(route to the appropriate files) and let the user know:
> "We've made good progress on the direction. I've updated project_overview.md with
> [what]. When you're ready to nail down specifics, we can spec it out."

---

## Step 6 — Write acceptance tests in the spec

When the spec-writer skill runs (Step 5 of that skill), include an **Acceptance Tests**
section with plain-language test descriptions. These should be concrete enough for the
night agent to translate into pytest code:

```markdown
## Acceptance Tests
- `search("machine learning", top_k=3)` returns exactly 3 results
- Each result has fields: `text`, `score`, `source`
- Scores are floats between 0.0 and 1.0, in descending order
- `search()` with an empty corpus raises `EmptyCorpusError`
- Results from a corpus of 1000 docs return in under 2 seconds
```

The night agent will implement these as real test functions **before** writing the
feature code (test-first).

---

## Signals to use this skill

The daytime supplement should invoke this skill when:
- The user shares a new idea ("what if we added...")
- The user describes a problem ("users are having trouble with...")
- The user wants to change direction ("actually, let's...")
- The user asks "what should we build next?"

The skill can also be invoked explicitly by the user asking for it.

---

## Done

Return to the calling supplement. The conversation continues — the user may have more
ideas, want to modify what was just specced, or move to a different topic.
