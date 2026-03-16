# Claude Code Sandbox Template for Windows

A set of configuration files that make it safer to run Claude Code with
`--dangerously-skip-permissions` on Windows, especially for unattended sessions
where no human is watching.

**Last updated**: March 2026

---

## Why this exists

Claude Code on Mac and Linux can use OS-level sandboxing (macOS `sandbox-exec`,
Linux namespaces/containers) that provides kernel-enforced process isolation.
Windows has no equivalent built-in mechanism that Claude Code can leverage
automatically.

This template fills that gap with a userspace sandboxing approach: command-string
inspection hooks, permission deny lists, and CLAUDE.md instructions that together
approximate the protection of a real sandbox — enough to catch the realistic
failure modes (accidental mistakes) even though it can't match kernel-level
isolation against adversarial escape attempts.

---

## Design decisions and rationales

### Why prepend, not replace, CLAUDE.md

**Decision**: The sandbox rules are shipped as `claudeMDsupplement.md` and prepended
to any existing CLAUDE.md rather than replacing it.

**Rationale**: Many projects already have a CLAUDE.md with project-specific instructions
(coding conventions, architecture notes, testing requirements). Replacing it would
destroy that context. Prepending ensures the sandbox rules are read first (highest
priority) while preserving everything the project already defined.

**Research basis**: Claude Code reads CLAUDE.md top-to-bottom. Instructions appearing
earlier in the file take implicit priority when there's a conflict. Prepending ensures
the security constraints are established before any project-specific instructions that
might contradict them.

### Why settings.json merges additively (never removes)

**Decision**: The installer script unions allow lists and deny lists. It never removes
an existing rule from either list.

**Rationale**: The sandbox should only add restrictions, never weaken existing ones.
If a project already denies `Bash(rm *)`, the sandbox installer shouldn't remove that
just because the sandbox's own deny list uses the more specific `Bash(rm -rf *)`.

Claude Code's own merge behavior (system + project settings) follows the same principle:
deny always wins, and rules are unioned across levels.

### Why allow/deny lists are duplicated across system and project levels

**Decision**: The project-level settings.json contains a full copy of deny rules, even
if they already exist in the user's `~/.claude/settings.json`.

**Rationale**: Robustness through redundancy. If the system settings file gets corrupted,
deleted, or the template is used on a different machine without system settings, the
project-level file still provides full protection. Since deny-always-wins means duplicate
deny rules have zero side effects, the cost of duplication is nothing.

**Research basis**: The eesel AI study found that 32% of developers using
`--dangerously-skip-permissions` encountered unintended file modifications and 9%
reported data loss. Defense in depth matters — a single-point-of-failure settings file
is not acceptable for a safety mechanism.

Source: https://www.eesel.ai/blog/settings-json-claude-code

### Why AskUserQuestion is blocked

**Decision**: A PreToolUse hook intercepts and blocks all `AskUserQuestion` tool calls.
The question is logged to `CLAUDE_LOG.md` and Claude is told to decide on its own.

**Rationale**: In unattended sessions, an unanswered question means the session stalls
indefinitely. Claude stops working and waits for a human who isn't there. This is the
single most common reason autonomous sessions fail to complete.

**Research basis**: The 108-hour unattended experiment by yurukusa found that Claude
stopping to ask "should I continue?" or "which approach?" was a primary cause of
wasted time in overnight sessions. Their `no-ask-human` hook, which this template
adapts, solved the problem.

Source: https://dev.to/yurukusa/4-hooks-that-let-claude-code-run-autonomously-with-zero-babysitting-1748

### Why curl, wget, and all network tools are blocked

**Decision**: All outbound network tools are denied — curl, wget, WebFetch, WebSearch,
and PowerShell Invoke-WebRequest/Invoke-RestMethod.

**Rationale**: With `--dangerously-skip-permissions`, a malicious CLAUDE.md in a cloned
repo could instruct Claude to exfiltrate credentials or source code to an external
server. Since all permission prompts are bypassed, Claude would comply without asking.
Blocking network tools at the deny-rule level prevents this attack vector.

Package managers (pip, npm, cargo) are intentionally NOT blocked — they need network
access to install dependencies, and they only download from their registries, not
arbitrary URLs.

**Research basis**: Multiple security guides recommend running `--dangerously-skip-permissions`
in a sandbox without internet access. Docker's `--network none` flag is the gold standard
for this. Since this template doesn't use Docker, blocking the tools Claude directly
controls is the next-best option.

Sources:
- https://www.ksred.com/claude-code-dangerously-skip-permissions-when-to-use-it-and-when-you-absolutely-shouldnt/
- https://www.backslash.security/blog/claude-code-security-best-practices

### Why secrets files (.env, .key, .pem, credentials) are blocked

**Decision**: Read permissions deny access to common secrets files.

**Rationale**: Even without network exfiltration, reading secrets into Claude's context
means those secrets are sent to Anthropic's API as part of the conversation. They could
appear in logs, be referenced in generated code, or persist in conversation history.
Blocking reads at the permission level prevents accidental exposure.

**Patterns blocked**: `.env`, `.env.*`, `*.key`, `*.pem`, `*.pfx`, `*.p12`,
`credentials*`, `*secret*`, `*password*`, `.aws/*`, `.ssh/*`, `.gnupg/*`, `.npmrc`,
`.pypirc`

Source: https://www.backslash.security/blog/claude-code-security-best-practices

### Why git push is blocked (not just force push)

**Decision**: All `git push` commands are denied, not just `git push --force`.

**Rationale**: In unattended mode, Claude should commit locally but never push. The
user needs to review changes before they reach a remote. A regular `git push` of
broken or incomplete code to a shared branch can be just as disruptive as a force push.

### Why max-turns is recommended

**Decision**: The launch command in the docs includes `--max-turns 200`.

**Rationale**: This is the hard guardrail against infinite loops. If Claude gets stuck
retrying a failing operation, `--max-turns` ensures it eventually stops rather than
running forever.

The CLAUDE.md "3 attempts then move on" rule is a soft constraint — Claude usually
follows it, but `--max-turns` is the hard backstop.

**Note on --max-budget-usd**: This flag caps API spending and is essential for API
users. If you're on Claude Max (flat-rate subscription), you don't need it — there's
no per-call billing to cap. It's still available if you switch to API usage later.

**Research basis**: Both flags are cited as "essential for unattended execution" in the
Claude Code documentation and community guides. `--max-budget-usd` applies only to
API billing, not subscription plans.

Source: https://code.claude.com/docs/en/best-practices

### Why the "3 attempts then move on" rule

**Decision**: The CLAUDE.md instructs Claude to try up to 3 meaningfully different
approaches for a failing operation, then log the failure and move on.

**Rationale**: The most common failure mode in long autonomous sessions is Claude
retrying the exact same failing command endlessly. This wastes turns, burns budget,
and produces no useful work.

**Research basis**: The 108-hour unattended experiment identified this as the #1
accident pattern. Three different approaches is enough to cover most failure modes
(syntax error → different syntax, library bug → different library, approach flaw →
different algorithm) without burning excessive turns.

Source: https://dev.to/yurukusa/i-let-claude-code-run-unattended-for-108-hours-heres-every-accident-that-happened-51cm

### Why git checkpoints before large changes

**Decision**: The CLAUDE.md instructs Claude to commit the current state before making
large-scale changes (refactoring, deleting code, architectural changes).

**Rationale**: In unattended mode, the user can't Ctrl+Z if Claude makes a wrong turn.
Git checkpoints provide a guaranteed rollback point. `git reset --hard HEAD~1` recovers
the state before the risky change.

**Research basis**: Experienced developers using Claude autonomously consistently
recommend `git add -A && git commit -m "checkpoint"` before every session. This template
applies the same principle within a session, at the granularity of individual risky
operations.

Source: https://blog.promptlayer.com/claude-dangerously-skip-permissions/

### Why "never ask — decide and log" (soft + hard enforcement)

**Decision**: Both the CLAUDE.md (soft) and no_ask_human.py hook (hard) enforce the
no-questions rule. The CLAUDE.md instruction tells Claude to log decisions in
CLAUDE_LOG.md. The hook blocks the tool call and also logs the blocked question.

**Rationale**: Belt and suspenders. The soft instruction teaches Claude the preferred
behavior (decide and log reasoning). The hard hook catches it if Claude tries to ask
anyway. Both write to CLAUDE_LOG.md so the user has a complete record of what Claude
decided autonomously and what it wanted to ask but couldn't.

### Why the directory guard uses a hardcoded path (not cwd)

**Decision**: On first run, Claude writes the absolute project path into
`directory_guard.py` as `HARDCODED_PROJECT_DIR`. The hook uses this instead of
the cwd from stdin.

**Rationale**: The cwd from stdin reflects where Claude thinks it is, which could
theoretically be manipulated. The hardcoded path is set once and becomes the
immutable source of truth. Even if Claude cd's elsewhere, the guard still checks
against the original project directory.

### Why the hook fails open (exit 0) on parse errors

**Decision**: If the directory guard can't parse its stdin JSON, it exits 0 (allow).

**Rationale**: Fail-closed (exit 2 on errors) would make Claude completely non-functional
if the hook encounters any unexpected input format. Since the hook runs before every
single tool call, a false-positive block would be catastrophic for usability. The
realistic threat model is Claude accidentally referencing wrong paths, not Claude
sending malformed JSON to its own hook.

---

## Known CVEs and how this template addresses them

These are real vulnerabilities that have been found in Claude Code. The template
addresses each where possible, but some require keeping Claude Code updated.

### CVE-2026-25724 — Symlink deny bypass

**What**: Deny rules could be bypassed by creating a symbolic link inside the
project that points to a file outside the project. Claude Code's deny rule
matching checked the literal path, not the resolved symlink target.

**Fixed in**: Claude Code v2.1.7 (for built-in deny rules).

**Our mitigation**: The directory guard hook uses `os.path.realpath()` to resolve
symlinks to their canonical path before checking containment. Additionally,
`mklink` is in the deny list so Claude can't create symlinks.

Source: https://github.com/anthropics/claude-code/security/advisories/GHSA-4q92-rfm6-2cqx

### CVE-2025-59536 / CVE-2026-21852 — Malicious project config RCE

**What**: A malicious `.claude/settings.json` or hooks in a cloned repo could
execute arbitrary commands and exfiltrate API keys, including Anthropic API tokens,
before the user even saw a trust dialog.

**Fixed in**: Claude Code v2.0.65.

**Our mitigation**: We block network exfiltration tools (curl, wget, WebFetch,
WebSearch). We recommend `managed-settings.json` at `C:\ProgramData\ClaudeCode\`
for deny rules that can't be overridden by project settings. Keep Claude Code
updated — this class of attack depends on the version.

Source: https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/

### Multiline command deny bypass (no CVE, known issue)

**What**: Deny rules match against command strings, but multiline commands and
commands with embedded strings can contain paths that aren't caught by pattern
matching. For example: `python -c "import os; os.remove('C:\\secret\\file')"`.

**Status**: No fix. This is a fundamental limitation of command-string inspection.

**Our mitigation**: The hook catches the realistic cases (accidental wrong paths
in direct arguments). The CLAUDE.md soft instructions prevent intentional bypass.
For true protection against this, use Docker or Windows Sandbox.

Source: https://dev.to/boucle2026/how-to-fix-claude-codes-broken-permissions-with-hooks-23gl

---

## What this does NOT protect against

This template is honest about its limitations:

1. **Dynamic path construction**: A command like
   `python -c "import os; os.remove('C:\\important\\file')"` would pass the hook
   because the path is inside a Python string, not a bare command argument.

2. **Child process behavior**: When Claude runs `pip install malicious-package`, pip
   could execute arbitrary code. The hook only sees `pip install malicious-package`,
   not what pip does internally.

3. **PowerShell variable concatenation**: `$p = "C:" + "\Users"; cat $p\file` constructs
   a path at runtime that the hook can't catch from the command string.

4. **Package manager network access**: pip, npm, cargo still reach the internet.
   Blocking those would make development impossible.

5. **Claude intentionally trying to escape**: Claude won't do this in practice, but
   the template cannot prevent it the way a kernel sandbox can. The soft instructions
   in CLAUDE.md are what prevent intentional escape — Claude follows instructions
   reliably, but they're not walls.

6. **Outdated Claude Code versions**: Several of the CVEs above were fixed in specific
   versions. **Keep Claude Code updated.** Run `claude --version` to check.

For true OS-level isolation on Windows, consider:
- **Windows Sandbox** — lightweight disposable VM built into Windows 11 Pro
- **Docker Desktop** with `--network none` — container isolation with no network
- **WSL2 + bubblewrap/firejail** — Linux kernel namespaces via Windows Subsystem for Linux

This template can layer on top of any of those for defense in depth.

---

## managed-settings.json (optional, strongest Windows protection)

Claude Code on Windows supports a machine-level settings file at:

```
C:\ProgramData\ClaudeCode\managed-settings.json
```

Rules in this file **cannot be overridden** by user settings (`~/.claude/settings.json`)
or project settings (`.claude/settings.json`). This makes it the strongest enforcement
point available on Windows without a VM.

### Why it matters

A malicious `.claude/settings.json` in a cloned repo could:
- Add allow rules that override your system denies (in theory)
- Add hooks that exfiltrate data
- Remove deny rules you depend on

Managed settings prevent all of this. Rules set in managed-settings.json are
immutable from Claude's perspective.

### Setup

```powershell
# Run as Administrator
New-Item -Path "C:\ProgramData\ClaudeCode" -ItemType Directory -Force

# Create managed-settings.json with your most critical deny rules
@'
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(rm -r *)",
      "Bash(format *)",
      "Bash(diskpart*)",
      "Bash(bcdedit*)",
      "Bash(reg *)",
      "Bash(regedit*)",
      "Bash(shutdown *)",
      "Bash(taskkill *)",
      "Bash(net user*)",
      "Bash(netsh *)",
      "Bash(git push*)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Read(**/.env)",
      "Read(**/.env.*)",
      "Read(**/*.key)",
      "Read(**/*.pem)",
      "WebFetch",
      "WebSearch"
    ]
  }
}
'@ | Set-Content "C:\ProgramData\ClaudeCode\managed-settings.json"

# Protect with NTFS permissions (optional but recommended)
$acl = Get-Acl "C:\ProgramData\ClaudeCode\managed-settings.json"
$acl.SetAccessRuleProtection($true, $false)
$adminRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators","FullControl","Allow")
$acl.AddAccessRule($adminRule)
$userRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Users","ReadAndExecute","Allow")
$acl.AddAccessRule($userRule)
Set-Acl "C:\ProgramData\ClaudeCode\managed-settings.json" $acl
```

Source: https://managed-settings.com/

---

## Research sources

These are the sources that informed the design decisions in this template. They may
become outdated as Claude Code evolves — check the dates and verify against current docs.

| Source | Date | What it informed |
|---|---|---|
| [Claude Code Best Practices (official docs)](https://code.claude.com/docs/en/best-practices) | 2025-2026 | max-turns, CLAUDE.md structure |
| [Claude Code Permissions (official docs)](https://code.claude.com/docs/en/permissions) | 2025-2026 | Deny rule syntax, evaluation order, settings merge behavior |
| [Claude Code Hooks Reference (official docs)](https://code.claude.com/docs/en/hooks) | 2025-2026 | PreToolUse hook exit codes, matcher syntax, JSON input format |
| [Claude Code Settings (official docs)](https://code.claude.com/docs/en/settings) | 2025-2026 | Settings file locations, merge hierarchy, managed-settings.json |
| [108 Hours Unattended - Every Accident (yurukusa)](https://dev.to/yurukusa/i-let-claude-code-run-unattended-for-108-hours-heres-every-accident-that-happened-51cm) | 2025 | Retry loop problem, 3-attempt rule, checkpoint strategy |
| [4 Hooks for Autonomous Operation (yurukusa)](https://dev.to/yurukusa/4-hooks-that-let-claude-code-run-autonomously-with-zero-babysitting-1748) | 2025 | no-ask-human hook design, AskUserQuestion blocking |
| [10 Hooks from 108 Hours (yurukusa)](https://dev.to/yurukusa/10-claude-code-hooks-i-collected-from-108-hours-of-autonomous-operation-now-open-source-5633) | 2025 | Safety hook patterns, context monitoring |
| [Safe Usage Guide + Configs (ksred)](https://www.ksred.com/claude-code-dangerously-skip-permissions-when-to-use-it-and-when-you-absolutely-shouldnt/) | 2025 | Network isolation rationale, sandbox-first approach |
| [Security Best Practices (Backslash)](https://www.backslash.security/blog/claude-code-security-best-practices) | 2025 | Secrets file blocking, .env protection |
| [managed-settings.json Guide](https://managed-settings.com/) | 2025-2026 | Windows machine-level enforcement, NTFS permissions |
| [eesel AI settings.json Guide](https://www.eesel.ai/blog/settings-json-claude-code) | 2026 | Deny rule patterns, statistics on unintended modifications |
| [Claude Code Enterprise Windows Deployment (GitHub)](https://github.com/subhashdasyam/claude-code-enterprise-windows-deployment) | 2025 | Windows ProgramData path, enterprise deployment patterns |
| [CVE-2026-25724 — Symlink deny bypass](https://github.com/anthropics/claude-code/security/advisories/GHSA-4q92-rfm6-2cqx) | 2026 | Symlink resolution in directory guard, realpath() fix |
| [CVE-2025-59536 — Malicious project config RCE (Check Point)](https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/) | 2026 | Network exfiltration blocking, managed-settings.json |
| [Multiline deny bypass (DEV Community)](https://dev.to/boucle2026/how-to-fix-claude-codes-broken-permissions-with-hooks-23gl) | 2026 | Limitations of command-string inspection |
| [Claude Code Permissions Docs — deny evaluation order](https://code.claude.com/docs/en/permissions) | 2026 | Deny-always-wins confirmation, rule precedence |
| [PermissionRequest hook bug (GitHub issue)](https://github.com/anthropics/claude-code/issues/19298) | 2025 | Why we use PreToolUse instead of PermissionRequest |

---

## File inventory

| File | Purpose | Installed to project? |
|---|---|---|
| `SETUP.md` | Instructions Claude follows to self-install. | No (read by Claude, stays in folder) |
| `DangerousClaudeFlagReadme.md` | This file. Design rationales and research. | No (reference only) |
| `HOW_TO_USE_claudeMDsupplement.md` | Step-by-step human instructions. | No (reference only) |
| `claudeMDsupplement.md` | Sandbox rules prepended to CLAUDE.md. | Yes → project root as/into CLAUDE.md |
| `settings_to_merge.json` | Permission allow/deny rules + hook config. | Yes → .claude/settings.json (merged) |
| `hooks/directory_guard.py` | Blocks file access outside project. | Yes → .claude/hooks/directory_guard.py |
| `hooks/no_ask_human.py` | Blocks questions, logs them instead. | Yes → .claude/hooks/no_ask_human.py |
