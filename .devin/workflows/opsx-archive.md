---
name: "OPSX: Archive"
description: "Archive a completed change in the experimental workflow"
category: Workflow
tags: [workflow, archive, experimental]
---

Finalize and archive a completed OpenSpec change (Fase 7):

1. Verify all tasks in `tasks.md` are marked `- [x]` and `openspec validate --changes` passes.
2. Sync delta specs into `openspec/specs/` when applicable.
3. Archive `openspec/changes/<name>` into `openspec/changes/archive/YYYY-MM-DD-<name>`.
4. Update `MEMORY.md` and run `graphify update .` when Graphify is enabled.
