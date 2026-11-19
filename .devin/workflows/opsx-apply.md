---
name: "OPSX: Apply"
description: "Implement tasks from an OpenSpec change"
category: Workflow
tags: [workflow, artifacts, experimental]
---

Implement tasks from `openspec/changes/<name>/tasks.md` (Fases 4–6) one task at a time:

1. Read `proposal.md`, `design.md`, `specs/`, and `tasks.md`.
2. Implement one pending task at a time with its corresponding tests.
3. Run `test`, `lint`, and `openspec validate --changes`.
4. Mark the task `- [x]` in `tasks.md` only after verification succeeds.
