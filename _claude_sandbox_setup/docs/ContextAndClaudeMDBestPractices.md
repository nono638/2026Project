# CLAUDE.md and Context Management — Research Findings

Researched 2026-03-13. Sources: Anthropic official docs, HumanLayer blog, MCPcat,
ClaudeLog, eesel.ai, and others listed at the bottom.

These findings informed two specific changes to this sandbox:
- Added "Preserve state across context compaction" to `claudeMDsupplement.md`
- Added "Verify before declaring done" to `claudeMDsupplement.md`
- Rewrote SETUP.md step 1 to read and neutralize existing CLAUDE.md contradictions

---

## CLAUDE.md: What the Research Actually Says

### The core tension

There's a real debate between "minimal CLAUDE.md" and "comprehensive CLAUDE.md." The
research resolves it clearly: **the failure mode of a bloated CLAUDE.md is that Claude
ignores parts of it.** Rules literally get lost in the noise. So the goal isn't
minimalism for its own sake — it's maximizing signal density.

The practical test for every line: *"Would removing this cause Claude to make mistakes?"*
If not, cut it.

### What belongs in CLAUDE.md

| Include | Exclude |
|---|---|
| Bash commands Claude can't guess | Anything derivable from reading the code |
| Code style rules that differ from language defaults | Standard conventions Claude already knows |
| Testing instructions and preferred test runner | Detailed API docs (link to them instead) |
| Repo etiquette (branch naming, PR conventions) | Frequently changing information |
| Architectural decisions specific to your project | Long explanations or tutorials |
| Developer environment quirks / required env vars | File-by-file codebase descriptions |
| Common gotchas and non-obvious behaviors | Self-evident practices like "write clean code" |

### Size targets

- Official Anthropic guidance: ruthlessly prune, no hard limit but shorter is almost
  always better
- HumanLayer recommendation: under 60 lines for strict rules
- General consensus: under 300 lines maximum before degradation risk
- **For autonomous/unattended mode**: justified to go longer because there's no human
  to course-correct, but the modular approach (see below) handles this better than
  just making the file longer

### The modular approach (best of both worlds)

Don't put everything in one file. Use CLAUDE.md's `@import` syntax to reference
separate docs:

```markdown
# CLAUDE.md (keep short — ~60-80 lines of actual rules)
See @docs/architecture.md for system design.

## Critical Rules
- Never modify files in /migrations without explicit instruction
- Always run `npm test` before committing

## References
- Git workflow: @docs/git-conventions.md
- Testing standards: @docs/testing.md
```

This keeps CLAUDE.md short while making full reference docs available on-demand.
The imported files only consume context when Claude actually reads them.

### Skills for domain knowledge

Skills (`.claude/skills/`) are loaded on demand, not at session start. They're the
right place for domain-specific workflows that only apply sometimes. Using skills
instead of CLAUDE.md for occasional knowledge means that knowledge doesn't bloat
every session's context.

### Nested CLAUDE.md files

Claude Code pulls in CLAUDE.md files hierarchically:
- `~/.claude/CLAUDE.md` — global personal preferences (applies to all projects)
- `./CLAUDE.md` — project root (check into git, shared with team)
- `./subdir/CLAUDE.md` — loaded automatically when Claude works in that subdirectory
- Parent directories — useful for monorepos

This means a monorepo can have a lean root CLAUDE.md and more specific rules closer
to the code that needs them.

### For autonomous/unattended operation specifically

When there's no human to course-correct, CLAUDE.md needs:

1. **Explicit behavioral guardrails with emphasis markers.** Using `IMPORTANT:` and
   `YOU MUST` actually improves adherence — Claude treats these as signal weight.
   Don't overuse them or they lose meaning, but for hard constraints they work.

2. **Compaction preservation instructions.** Auto-compact can fire mid-task and wipe
   working memory. Adding a line like:
   > "When compacting, always preserve: current task status, list of modified files,
   > decisions made autonomously, and any errors encountered."
   directly influences what the compaction summary retains. This is now in our template.

3. **Verification criteria.** Anthropic's official docs call this "the single
   highest-leverage thing you can do." Without tests or a way to self-check, Claude
   produces plausible-looking output that may not actually work. The template now
   requires Claude to run tests/linter/sanity check before marking any task complete.

4. **Failure behavior.** What should Claude do when blocked? The template handles this
   with the 3-attempt rule and CLAUDE_LOG.md logging.

5. **Scope boundaries.** Explicit "do not touch" boundaries since there's no human to
   catch scope creep. The template handles this with the SCOPE AND AUTONOMY section.

### Diagnosing a broken CLAUDE.md

- Claude keeps doing something despite a rule against it → file is too long, rule is
  getting lost. Prune aggressively.
- Claude asks questions answered in CLAUDE.md → phrasing is ambiguous. Rewrite that
  rule more directly.
- Claude ignores emphasis → too many things are emphasized. Reserve `IMPORTANT:` /
  `YOU MUST` for the 2-3 genuinely critical constraints.

---

## Context Management: How It Actually Works

### The auto-compact misconception

Auto-compact does **not** wait until the context is full. It triggers at **64-75%
capacity**. But the problem is this still happens unpredictably mid-task, which is
disruptive in unattended sessions. Performance degrades before auto-compact fires —
the degradation starts earlier than most people expect.

Claude Code's 200,000 token context window sounds large, but system prompts, tool
definitions, MCP server schemas, and CLAUDE.md consume 30,000-40,000 tokens before
any work begins.

### The proactive strategy

Manual `/compact` at logical breakpoints is far better than letting auto-compact
interrupt mid-task:

```
/compact preserve current architecture decisions and list of modified files
/compact focus on the new feature requirements, discard debugging history
/compact keep the solution we found, remove all the failed debugging attempts
```

Best moments to compact:
- After a git commit
- After completing a distinct feature or subtask
- Before switching to a different area of the codebase
- When you notice context usage passing ~50%

### Context status line

Claude Code has a configurable status line that shows context usage in real time.
Enable it via `/config`. This lets you see degradation coming at 50-60% rather than
discovering it at 75% when auto-compact fires.

### Subagents for investigation

When Claude researches a codebase it reads many files, all consuming context. Routing
research to a subagent keeps the main context clean:

> "Use a subagent to investigate how our auth system handles token refresh."

The subagent reads hundreds of files and reports back a summary. The main context
only gets the summary, not all the file contents.

### `/btw` for quick side questions

Quick questions that don't need to persist in context. The answer appears in a
dismissible overlay and never enters conversation history. Underused feature.

### `/clear` vs `/compact`

- `/clear` — full reset, no preservation. Use between completely unrelated tasks.
- `/compact` — summarizes and compresses. Use at logical breakpoints within a task.
- `/compact <instructions>` — targeted summarization. Preserves what you specify.

After two failed corrections on the same issue: `/clear` and write a better initial
prompt. A clean session with a better prompt almost always beats a long session with
accumulated failed attempts.

### MCP token overhead

MCP server tool schemas consume significant context before any work starts. One
benchmark showed 51,000 tokens of MCP overhead reduced to 8,500 tokens (46.9%
reduction) after Claude Code's Tool Search feature was enabled — it auto-defers
MCP tool schema loading until a tool is actually needed.

If you have many MCP servers configured, disable unused ones before compacting:
`/mcp` or `@server-name disable`.

### MCP: dynamic context pruning

There's a dedicated MCP server for proactive context management:
`dxta-claude-dynamic-context-pruning`

It saves/loads structured checkpoints that survive compaction, tracks duplicate tool
usage, and marks subtask completion for safe context reduction.

```json
{
  "mcpServers": {
    "dxta-claude-dynamic-context-pruning": {
      "command": "npx",
      "args": ["-y", "dxta-claude-dynamic-context-pruning"]
    }
  }
}
```

---

## For This Sandbox Specifically

The `claudeMDsupplement.md` template is already purpose-built for unattended operation
and covers most of what the research recommends. The two genuine gaps that were added:

1. **Compaction preservation** (UNATTENDED OPERATION section) — writes task state to
   CLAUDE_LOG.md before long operations so it survives context resets.

2. **Verify before done** (PROGRESS REPORTING section) — run tests/linter before
   marking task complete. The single highest-leverage autonomous behavior per
   Anthropic's own docs.

The template is ~230 lines, above the "ideal under 80" threshold. This is justified
for unattended mode — more explicit direction is needed when there's no human to
correct mistakes. The modular @import approach could reduce this further if needed.

---

## Sources

- [Best Practices for Claude Code — Anthropic official docs](https://code.claude.com/docs/en/best-practices)
- [Writing a good CLAUDE.md — HumanLayer Blog](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Managing Claude Code Context — MCPcat](https://mcpcat.io/guides/managing-claude-code-context/)
- [What is Claude Code auto-compact — ClaudeLog](https://claudelog.com/faqs/what-is-claude-code-auto-compact/)
- [Claude Code Autonomous Mode guide — Pasquale Pillitteri](https://pasqualepillitteri.it/en/news/141/claude-code-dangerously-skip-permissions-guide-autonomous-mode)
- [context-manager MCP Plugin — LobeHub](https://lobehub.com/mcp/dxta-claude-dynamic-context-pruning)
- [7 Claude Code best practices for 2026 — eesel.ai](https://www.eesel.ai/blog/claude-code-best-practices)
- [How Claude Code Got Better by Protecting More Context — Hyperdev](https://hyperdev.matsuoka.com/p/how-claude-code-got-better-by-protecting)
