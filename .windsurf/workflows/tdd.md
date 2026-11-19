---
description: Vía TDD (rojo → verde → refactor) para cambios chicos
---

1. Clasificar con la checklist de la Fase 0.5 de `AGENTS.md` y confirmar la vía con el usuario.
2. RED: escribir el test mínimo en `tests/` y correr `python scripts/harness.py tdd-red <test_id>` hasta que falle por el motivo correcto.
3. GREEN: implementar lo mínimo y correr `python scripts/harness.py tdd-green <test_id>` (test + suite + lint en verde).
4. Refactor opcional re-corriendo `tdd-green`.
5. Cierre: entrada en `CHANGELOG.md` `[Sin publicar]`, `log`/`stop` en Gantt, `graphify update .`, y pedir confirmación para commitear.

Detalles y racionalizaciones en `.devin/skills/tdd/SKILL.md`.
