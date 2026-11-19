# Changelog

Todas las novedades notables del proyecto se registran acá, siguiendo
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Sin publicar]

### Agregado

- Harness de `dev-harness-template` (perfiles `base, openspec, graphify,
  ml-timeseries`): política ODD/SDD, reglas multi-IDE, scripts
  `test`/`lint`/`build`/`verify-setup`, escáner de secretos y CI del
  consumidor.
- Vía TDD local (`scripts/harness.py` con `unit`, `lint`, `tdd-red`,
  `tdd-green`), skill `tdd` para Devin/Windsurf y clasificación Fase 0.5
  en `AGENTS.md`.
- Suite `tests/` con unittest y fixtures sintéticas: tests del propio
  harness (`test_harness_tdd.py`) y primer test-regla de
  `core/inference.py` (`test_inference_units.py`).
- CI de GitHub Actions que corre tests, lint y build del harness en
  Windows y Ubuntu.

### Corregido

- La app ya no crashea al arrancar en despliegues sin `data/processed/`
  (p. ej. Streamlit Cloud): `try_load_report_artifacts()` devuelve
  `None` y la navegación oculta "Reporte del Modelo". Test-regla:
  `tests.test_model_report_units.TryLoadReportArtifactsTests`.

### Cambiado

- `AGENTS.md` reescrito: política ODD/SDD + Fase 0.5 TDD + contexto del
  proyecto (antes solo puntero a `.windsurfrules`).
- `.gitignore`: `docs/workflows/` y `docs/profiles/` ahora se versionan;
  `.agents/` deja de ignorarse (es contenido del harness); se ignoran
  `nul`, `streamlit*.log` y `notebooks/models/`.
