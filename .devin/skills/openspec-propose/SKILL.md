---
name: openspec-propose
description: Propose a new change with all artifacts generated in one step. Use when the user wants to quickly describe what they want to build and get a complete proposal with design, specs, and tasks ready for implementation.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.3.1"
---

Propose a new change - create the change and generate all artifacts in one step:
- `proposal.md` (what & why)
- `design.md` (how)
- `tasks.md` (implementation steps)
- `specs/` (delta capabilities)

When ready to implement, run `/opsx:apply`.

---

**Steps**

1. If no clear input is provided, ask the user what change they want to build and derive a kebab-case name.
2. Create or verify the change directory with `openspec new change "<name>"` (or inspect `openspec/changes/<name>/`).
3. Check artifact build order with `openspec status --change "<name>" --json`.
4. Generate `proposal.md`, `design.md`, `specs/<capability>/spec.md`, and `tasks.md` following the project context and schema rules.
5. Validate the resulting change with `openspec validate --changes`.
6. Summarize the created artifacts and ask for user confirmation before moving to `/opsx:apply`.
