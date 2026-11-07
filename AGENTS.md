# AGENTS.md — predictivo_ml_leads

Las reglas completas del proyecto están en `.windsurfrules` (contexto, arquitectura, parámetros críticos, decisiones de negocio). Este archivo existe para agentes que no leen `.windsurfrules`.

## Arranque
- No existe `.venv` en el repo. Hoy corre con el Python del sistema (3.14) que tiene streamlit/sklearn/shap:
  `python -m streamlit run streamlit_app/app.py` → http://localhost:8501
- Tests de humo: `python scripts/run_tests.py`

## Modelo desplegado (fuente de verdad: el artefacto, no la doc)
- `models/best_model.joblib` = **GradientBoostingClassifier**, 22 features. (CONTEXTO.md dice Random Forest: está desactualizado.)
- Umbral efectivo en la app: 0.325 (`inference.py:resolve_umbral`); el artefacto guarda 0.35.

## Documentos de análisis
- `docs/CASO_BD_MX_2026_LIMPIEZA.md` — hallazgos BD 2026, dudas para MX, plan de limpieza.
- `docs/ANALISIS_ESTRATEGICO_2026-10.md` — revisión, bugs B1–B9, arquitectura unificada.

## Workflow
- NUNCA `git commit`/`push` sin preguntar.
- No subir datos con PII (`data/`, exportes de Salesforce) al repo.

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
