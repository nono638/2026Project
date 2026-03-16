---
name: architecture-advisor
description: Evaluate tech stacks, design patterns, and architectural decisions with research and evidence — avoid dead ends before committing
---

# Architecture Advisor

Use this skill when making decisions that are expensive to reverse: tech stack choices,
library selection, data model design, API shape, abstraction boundaries. The goal is to
**avoid dead ends** by doing due diligence before committing.

This skill is for **daytime use only** — it requires user input, web research, and
deliberation. If running in nighttime mode, skip this skill and log a warning.

---

## Philosophy

The most expensive mistake in a spare-time project isn't picking the wrong library — it's
picking it hastily, writing 2000 lines of code overnight, then discovering it doesn't
support your actual use case. A 30-minute conversation now saves 3 hours of wasted night
cycles later.

**Research before opining.** Don't recommend from vibes. Use WebSearch to check real data:
GitHub activity, open issues, Stack Overflow questions, compatibility notes. Include URLs
in every recommendation so the user can verify.

**Probe before recommending.** The right answer depends on constraints you might not know
yet. Ask first, then advise.

---

## Step 1 — Understand the decision

What specifically needs to be decided? Common categories:

| Decision type | Examples |
|---|---|
| **Tech stack** | Which web framework? Which database? Which ORM? |
| **Library choice** | httpx vs requests? SQLAlchemy vs raw SQL? |
| **Design pattern** | Class hierarchy vs composition? Interface vs concrete? |
| **Abstraction level** | Should this be a reusable module or a one-off script? |
| **Data model** | Relational vs document? Normalized vs denormalized? |
| **API shape** | REST vs GraphQL? Function signatures? |

State the decision clearly:
> "We need to decide: [specific question]. This affects [what downstream work]."

---

## Step 2 — Probe for constraints

Before evaluating options, understand what you're optimizing for. Ask about these
(skip any that are obviously answered by project context):

### Hard constraints (non-negotiable)
- What language/runtime must we use? (Python only? Node OK?)
- What's the deployment environment? (Laptop? Cloud? Docker?)
- Any budget constraints? (No paid APIs? Free tier only?)
- What's already in the codebase that we must integrate with?
- Any dependencies that are already chosen and locked in?

### Soft constraints (preferences)
- How much complexity is acceptable? (Solo dev → simpler is better)
- How important is long-term maintenance vs. shipping fast?
- Learning curve tolerance? (Familiar tools vs. best-in-class?)
- Performance requirements? (Quantify if possible: "under 2 seconds for 1000 docs")

### Project context
- What's the expected lifespan? (Thesis project vs. long-term tool?)
- How many people will maintain this? (Just you? Team?)
- Is this exploratory (might pivot) or committed (building on it)?

Use multiple choice for quick constraint gathering:
> **A** — Performance is critical, I'll accept more complexity
> **B** *(likely)* — Keep it simple, performance is secondary
> **C** — Somewhere in between — let me explain

---

## Step 3 — Research the options

For each viable option, research using WebSearch/WebFetch:

### For libraries/frameworks, check:
- **Activity**: Last commit date, release frequency, open issues count
- **Community**: GitHub stars, Stack Overflow questions, Discord/forum activity
- **Compatibility**: Does it work with our Python version? Our OS? Our other deps?
- **Maturity**: How long has it existed? Any major breaking changes recently?
- **Gotchas**: Search for "<library> problems" / "<library> limitations" / "<library> vs <alternative>"
- **License**: Any restrictions? (GPL can be viral)

### For design patterns, check:
- **Current codebase**: What patterns are already in use? Consistency matters.
- **Scale**: How many places will this pattern be used? (1 place = keep it simple)
- **Flexibility needed**: Will requirements change? (If yes → interfaces. If stable → concrete.)

### For data models, check:
- **Query patterns**: What questions will the code ask of this data?
- **Volume**: How much data? What's the growth trajectory?
- **Relationships**: Are there natural joins? Or is it mostly key-value?

---

## Step 4 — Build a decision matrix

Present findings as a concrete comparison. Don't just list pros/cons — score them
against the constraints from Step 2:

```markdown
## <Decision>: Options Comparison

| Criteria | Weight | Option A: <name> | Option B: <name> | Option C: <name> |
|---|---|---|---|---|
| Simplicity | High | ✅ Minimal API | ⚠️ Config-heavy | ❌ Steep learning curve |
| Performance | Medium | ⚠️ Adequate | ✅ Fast | ✅ Fast |
| Community | Medium | ✅ Active | ✅ Active | ⚠️ Niche |
| Compatibility | High | ✅ Works | ✅ Works | ❌ Requires Node |
| Maintenance | High | ✅ Stable API | ⚠️ Breaking changes | ✅ Stable |

**Recommendation: Option A**
**Why:** [1-2 sentences linking the recommendation to the highest-weighted criteria]
**Risk:** [What could go wrong with this choice, and how would we recover]
```

Include source URLs for every factual claim.

---

## Step 5 — Dead-end detection

Before the user commits, explicitly ask:

> "Before we lock this in, let me stress-test it:
> - **What if [likely requirement change]?** Would this choice still work?
> - **What if [scale changes]?** Does it handle 10x the current need?
> - **What's the escape hatch?** If this doesn't work out, how hard is it to switch?
> - **What are we giving up?** Every choice closes doors — which doors are we OK closing?"

If the escape hatch is expensive (major rewrite), flag it explicitly:
> "⚠️ Switching away from [choice] later would require rewriting [X]. Make sure we're
> confident before the night agent builds on this."

---

## Step 6 — When to abstract vs. keep simple

For design pattern decisions specifically:

### Make it a class/interface when:
- It will be used in 3+ places with different implementations
- You need polymorphism (swap implementations at runtime)
- The spec explicitly calls for a plugin system or strategy pattern
- Testing requires mocking (interfaces make this clean)

### Keep it a function/one-off when:
- It's used in 1-2 places
- There's only one implementation and no foreseeable second one
- The logic is straightforward and self-contained
- Adding a class would just be wrapping a function for no benefit

### The "Rule of Three" heuristic:
- **First time**: Write it inline, keep it simple
- **Second time**: Notice the duplication, but don't abstract yet
- **Third time**: Now extract it — you have enough examples to know the right abstraction

> "Right now this is used in [N] places. I'd recommend [concrete/abstract] because
> [reason]. If it grows to [threshold], we should revisit."

---

## Step 7 — Record the decision

Write to `DaytimeOnly/reference/architecture-decisions.md`:

```markdown
### YYYY-MM-DD — <Decision title>

**Decision:** <What was chosen>
**Status:** Accepted
**Context:** <Why this decision was needed — 1-2 sentences>

**Options considered:**
1. **<Option A>** — <one line>. Rejected because: <reason>
2. **<Option B>** — <one line>. **Selected** because: <reason>
3. **<Option C>** — <one line>. Rejected because: <reason>

**Constraints that drove this decision:**
- <constraint 1>
- <constraint 2>

**Risks and mitigations:**
- Risk: <what could go wrong>. Mitigation: <how we'd handle it>

**Escape hatch:** <How hard is it to reverse this decision. Easy/Medium/Hard + what's involved>

**Research:**
- <URL 1> — <what it showed>
- <URL 2> — <what it showed>
```

This record means the night agent can follow the decision without relitigating it,
and future-you can understand why the choice was made without reconstructing the
conversation.

---

## Done

Return to the calling supplement. The decision is recorded and can now inform spec
writing — the spec-writer skill will reference architecture-decisions.md when creating
task specs.
