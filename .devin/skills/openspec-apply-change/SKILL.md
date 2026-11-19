---
name: openspec-apply-change
description: Implement tasks from an OpenSpec change. Use when the user wants to start implementing, continue implementation, or work through tasks.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.3.1"
---

Implement tasks from an OpenSpec change one task at a time.

---

**Steps**

1. Select the active change (or ask the user if multiple changes exist via `openspec list --json`).
2. Read the change context files (`proposal.md`, `design.md`, `specs/`, `tasks.md`).
3. For each pending task (`- [ ]`) in `tasks.md`:
   - Announce which task is being worked on.
   - Write or update tests alongside the minimal code changes required.
   - Run the deterministic verification suite (`test`, `lint`, `openspec validate --changes`).
   - Mark the task complete (`- [x]`) in `tasks.md` immediately after verification passes.
4. Pause on blockers, ambiguous requirements, or design issues and report status to the user.
