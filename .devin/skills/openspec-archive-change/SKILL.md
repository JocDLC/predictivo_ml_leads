---
name: openspec-archive-change
description: Archive a completed change in the experimental workflow. Use when the user wants to finalize and archive a change after implementation is complete.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.3.1"
---

Archive a completed OpenSpec change after verifying all tasks and artifacts are done.

---

**Steps**

1. Verify artifact and task completion with `openspec status --change "<name>" --json` and check that all tasks in `tasks.md` are marked `- [x]`.
2. Assess delta specs under `openspec/changes/<name>/specs/` and sync them into `openspec/specs/` if confirmed by the user.
3. Move `openspec/changes/<name>` to `openspec/changes/archive/YYYY-MM-DD-<name>` (or run `openspec archive "<name>"`).
4. Refresh the knowledge graph (`graphify update .`) if the `graphify` profile is active.
5. Present the completion summary to the user.
