# AGENTS.md — predictivo_ml_leads

Política de desarrollo (harness adoptado de `dev-harness-template`) + contexto
del proyecto. El contexto de negocio completo está en `.windsurfrules`;
`.harness/context.json` declara rol `consumer` y perfiles activos
(`base, openspec, graphify, ml-timeseries`).

## Contexto y autoridad

Leer `.harness/context.json` antes de elegir herramientas. OpenSpec, Graphify y
ml-timeseries solo aplican con perfil activo y capacidad disponible. No inventar
slash commands ni instalar herramientas implícitamente. Una consulta o
investigación se mantiene en lectura salvo autorización de cambio.

## Fase 0.5 — Clasificación (TDD / ODD / SDD)

Todo cambio se clasifica antes de planificar. El agente propone la vía en una
frase (riesgo, incertidumbre, impacto) y el usuario la confirma.

Va por **TDD** solo si TODAS se cumplen:

1. Es un bugfix, un ajuste de regla o validación puntual, o un refactor con el
   comportamiento cubierto por tests.
2. Toca como máximo 3 archivos productivos, en un solo módulo o capa
   (`core/`, `components/`, `src/`).
3. No agrega vistas de UI, formatos de artefacto/BD ni contratos nuevos.
4. No toca el modelo (`models/*.joblib`), el target, el umbral, las features
   de training, ni datos reales o PII.
5. El comportamiento esperado se puede escribir en una sola frase sin
   ambigüedad.
6. Se puede verificar con un test determinístico sobre fixtures sintéticas
   (sin Excel real, sin Streamlit corriendo).

Si falla alguna:

- **ODD** si el resultado es claro, el alcance acotado, el cambio reversible,
  verificable y de bajo riesgo, sin nuevas decisiones sensibles de datos,
  permisos, modelos, contratos o dependencias. Guía: `docs/workflows/odd.md`.
- **SDD** para trabajo sustancial o sensible, incertidumbre importante o
  contratos nuevos. Guía: `docs/workflows/sdd.md`. Con OpenSpec activo usar
  `/opsx:explore`, `/opsx:propose`, `/opsx:apply`, `/opsx:archive`.
- Con `ml-timeseries` activo, consultar `docs/profiles/ml-timeseries.md`
  antes de trabajo con ejes temporales (split cronológico, madurez de
  etiquetas, fugas).

Si durante TDD/ODD aparece riesgo o alcance nuevo, escalar antes de
implementarlo. El flujo TDD es T1–T5 del skill `tdd`
(`.devin/skills/tdd/`): clasificar → `tdd-red` → `tdd-green` → refactor →
cierre.

## Controles comunes

- Código legible, cambios coherentes y verificables. No debilitar checks para
  obtener un resultado verde; corregir la causa de una regresión.
- Checks pertinentes: `python scripts/harness.py unit` (tests sintéticos),
  `python scripts/harness.py lint` (ruff), `./scripts/lint.ps1` (escáner de
  secretos), `./scripts/test.ps1` y `./scripts/build.ps1` (harness + OpenSpec),
  `python scripts/run_tests.py` (smoke del producto con datos reales, local).
- `VERSIONS.md` fija versiones del harness. El doctor es
  `scripts/verify-setup.ps1` / `verify-setup.sh`.
- No leer, persistir ni imprimir credenciales ni datos personales de leads.
  Tests y ejemplos con fixtures sintéticas.
- Al cerrar, informar cambio, evidencia y límites. No declarar completo un
  check pendiente.

## Contexto del proyecto

### Arranque
- No existe `.venv` en el repo. Corre con el Python del sistema (3.14) que
  tiene streamlit/sklearn/shap:
  `python -m streamlit run streamlit_app/app.py` → http://localhost:8501
- Smoke del producto: `python scripts/run_tests.py`
- Suite unitaria (sin datos reales): `python scripts/harness.py unit`

### Modelo desplegado (fuente de verdad: el artefacto, no la doc)
- `models/best_model.joblib` = **GradientBoostingClassifier**, 22 features.
- Umbral efectivo en la app: 0.325 (`inference.py:resolve_umbral`); el
  artefacto guarda 0.35.

### Documentos de análisis (gitignored, no van al repo)
- `docs/CASO_BD_MX_2026_LIMPIEZA.md` — hallazgos BD 2026, dudas Q1–Q12, plan de limpieza v3.
- `docs/ANALISIS_ESTRATEGICO_2026-10.md` — revisión, bugs B1–B9, arquitectura unificada.
- `docs/MAPA_ORIGEN_CAMPANAS_2026.md` — jerarquía Origen(UTM) → Origen de creación(canal) → Formulario.
- `docs/CONTRASTE_REUNION_MX_2026-11-05.md` — respuestas de MX a Q1–Q12, ventana R-Solve de 6 meses (impacta el split), 3 KPIs Sales on Leads.

### Workflow
- NUNCA `git commit`/`push` sin preguntar.
- No subir datos con PII (`data/`, exportes de Salesforce, outputs de
  notebooks con filas de leads) al repo.

## Tracker Gantt

Este proyecto está traqueado en el tablero Gantt del usuario
(`C:\Users\ax24611\Downloads\_dev\Grantt`). Al empezar/terminar trabajo
traqueable, reportar con:
`node C:\Users\ax24611\Downloads\_dev\Grantt\scripts\update-tracker.mjs <cmd>`
Comandos: `list` | `start <id>` | `stop <id> --progress N` | `status <id> <estado>` |
`blocker <id> "motivo"` | `unblock <id>` | `release mx-model "vX.Y — nombre"` | `log <id> "avance"`
- Este repo corresponde al proyecto `mx-model` del tracker.
- `release` ≠ `marker`: versiones van con `release`; `marker` solo para stoppers externos.
- `--push` solo si el usuario lo pide. Ver skill `gantt-tracker`.
