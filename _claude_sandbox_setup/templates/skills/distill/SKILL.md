---
name: distill
description: Synthesize unstructured conversation into structured artifacts — extract decisions, requirements, context, and open questions from rambling discussion
---

# Distill

Use this skill when a conversation has covered a lot of ground — the user has been thinking
out loud, exploring ideas, going on tangents — and it's time to make sense of it all. This
is the heavyweight version of the day agent's always-on "listen and reflect" behavior.

If running in nighttime mode, skip this skill entirely.

---

## When to invoke this skill

- The conversation has been going for a while with no concrete output yet
- The user explicitly asks to "make sense of this" or "what did we decide?"
- You realize you've accumulated a lot of context but haven't routed any of it
- The user says something like "OK, so where are we?" or "let me think out loud for a minute"

---

## Step 1 — Replay and categorize

Go back through the conversation so far. For every substantive thing the user said,
categorize it:

| Category | What it looks like | Where it goes |
|---|---|---|
| **Decision** | "Let's use SQLite" / "I don't want a React frontend" | architecture-decisions.md |
| **Requirement** | "It needs to handle 1000 docs" / "Users should be able to..." | Future spec |
| **Constraint** | "I only have Python" / "No paid APIs" / "Must work offline" | project_overview.md (constraints section) |
| **Goal** | "I want this to be the fastest RAG system" / "This is for my thesis" | project_overview.md (goals section) |
| **Concern** | "I'm worried about performance" / "What if the API changes?" | Open questions to resolve |
| **Tangent** | "Oh, that reminds me of..." / "Someday we could..." | incubating.md or drop |
| **Context** | "In my experience..." / "The way this field works is..." | reference/research.md |

Don't force-fit. If something doesn't clearly fit a category, mark it as "unclear — needs
follow-up."

---

## Step 2 — Present the synthesis

Show the user what you extracted, organized by category. Be concise — one line per item:

> **Here's what I'm taking away from our conversation:**
>
> **Decisions made:**
> - Use SQLite for storage (zero-config, dataset <1GB)
> - No React — keep the frontend simple, maybe just CLI
>
> **Requirements emerging:**
> - Must handle corpus of ~1000 documents
> - Search results need relevance scores
>
> **Constraints:**
> - Python only, no paid APIs
> - Must run on a laptop without GPU
>
> **Open questions (need your input):**
> - How should we handle documents longer than the context window?
> - Is real-time indexing needed or is batch OK?
>
> **Parked for later:**
> - Multi-language support (you mentioned it but didn't commit)
>
> **Did I miss anything or get anything wrong?**

---

## Step 3 — Correct and confirm

Let the user correct, add, or remove items. This is critical — the synthesis is only
valuable if it accurately reflects what they meant, not what they said.

Common corrections:
- "That was just me thinking out loud, not a decision"
- "Actually, that's more important than I made it sound"
- "You missed the part where I said..."

Update your categories based on their corrections.

---

## Step 4 — Route everything

Now write each piece to its destination:

1. **Decisions** → `DaytimeOnly/reference/architecture-decisions.md`
   Format: `### YYYY-MM-DD — <decision>` with `**Why:** <rationale>` and
   `**Alternatives considered:** <what was rejected and why>`

2. **Goals and constraints** → `DaytimeOnly/project_overview.md`
   Update the relevant sections. Add a change history entry if goals shifted.

3. **Requirements** → If enough for a spec, hand off to spec-writer.
   If not ready, capture in `DaytimeOnly/inbox.md` with context for later triage.

4. **Context / research** → `DaytimeOnly/reference/research.md`

5. **Tangents / future ideas** → `DaytimeOnly/incubating.md` with a next trigger

6. **Open questions** → These stay in the conversation. Ask the most important one now.

Tell the user what you wrote and where:
> "Logged 2 decisions to architecture-decisions.md, updated project goals, captured 3
> requirements in inbox for speccing later. One open question: [ask it]."

---

## Step 5 — Assess readiness

After routing, check: is there enough to write a spec for anything?

- **If yes:** "I think [topic] is ready to spec. Want me to draft it now?"
  → Hand off to spec-writer or intake for the remaining questions.
- **If no:** "We've made progress but [topic] still needs [specific gap]. Let's come back
  to it when [trigger]." → Make sure the gap is captured in incubating.md with the trigger.

---

## Done

Return to the calling supplement. The conversation continues naturally.
