---
name: "OPSX: Propose"
description: "Propose a new change - create it and generate all artifacts in one step"
category: Workflow
tags: [workflow, artifacts, experimental]
---

Create or update an OpenSpec change proposal (Fase 2) with all required artifacts:

1. Create or select `openspec/changes/<name>/`.
2. Generate `proposal.md`, `design.md`, `specs/<capability>/spec.md`, and `tasks.md`.
3. Validate contracts with `openspec validate --changes`.
4. Resolve unapproved scope/design decisions before `/opsx:apply`; reuse valid approvals for the same scope under AGENTS.md. This workflow belongs to the SDD route.
