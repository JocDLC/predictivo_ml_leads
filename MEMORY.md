# Memoria del proyecto

## Non-goals

- No reemplazar el agente nativo de cada IDE (Cascade, Claude, Codex, Devin, etc.). El arnés es agnóstico.
- No crear un instalador `.exe`/`.pkg` propietario. Usamos compositor por perfiles en PowerShell y Bash.
- No incluir `ruflo` ni `omniroute` en este harness hasta haberlos probado y evaluado por separado.
- No copiar colecciones masivas de subagentes (`awesome-claude-code-subagents` se mantiene solo como referencia evaluada).
- No leer, persistir ni imprimir credenciales, tokens o sesiones OAuth de GitHub/Codex.

## Decisiones arquitectónicas

- **Política híbrida (2026-10-19, aprobada)**: AGENTS elige ODD o SDD por riesgo e
  incertidumbre, con guías consultadas según la tarea. ODD no exige artefactos ni
  confirmaciones por paso; SDD reutiliza contratos/aprobaciones y registra cambios
  sensibles antes de implementar. No se instala un orquestador adicional.

- **Composición modular por perfiles (`scripts/profiles-manifest.json`)**: `base` obligatorio + perfiles opcionales `codex`, `openspec`, `graphify`, `archify` y `ml-timeseries`.
- **Series temporales (2026-10-18)**: `ml-timeseries` aporta reglas y contratos para un caso futuro. Inactivo en el contexto actual; `all` lo incluye. Sin modelos ni librerías ML instaladas.
- **AGENTS.md y VERSIONS.md** como fuentes de verdad universales; configuraciones por IDE son derivadas.
- **OpenSpec** para Spec-Driven Development con workflows y skills nativos (`opsx-explore`, `opsx-propose`, `opsx-apply`, `opsx-archive`) en `.devin/` y `.windsurf/`.
- **Graphify** para context engineering del agente y reducción de tokens (`graphify-out/`).
- **Archify (`tt-a1i/archify` v3.1.0)** para documentación visual interactiva de arquitectura (`docs/architecture/`) con `ARCHIFY_UPDATE_CHECK_DISABLED=1`.
- **Guardrails de OneDrive y Disco `C:`**: verificación mecánica de espacio libre en `verify-setup` y exclusión por defecto de audio, modelos binarios y temporales de Office/OneDrive en `.gitignore` y `.graphifyignore`.
- **Escaneo determinístico de secretos**: integrado en `scripts/lint.ps1` y `scripts/lint.sh`.

## Failure patterns

- `C:` con poco espacio bloquea instalaciones de extensiones y cachés de `npm`/`pip`/`uv` → `verify-setup` avisa cuando hay menos de 5 GB libres.
- Repositorios en OneDrive sincronizando binarios pesados, grabaciones `.mp3`/`.wav` o `node_modules`/`.venv` → bloqueados en `.gitignore` y `.graphifyignore`.
- Instalar `@latest` sin fijar versiones rompe reproducibilidad → todas las versiones están fijadas en `VERSIONS.md`.
- Mantener Graphify AST al cambiar código y arquitectura cuando cambien estructura, flujos o contratos; omitir esas actualizaciones pertinentes deja contexto obsoleto.

## Estado actual

- Integración a `master` y preparación para adopción autorizadas por el usuario
  el 2026-10-19. README incluye composición conservadora, revisión del diff,
  combinación de reglas y checks del producto. El estado efectivo del merge y
  la CI se consulta en GitHub; los resultados siguientes conservan su SHA probado.
- CI de `66cc730`: push 39273517442 y PR 39273810318 aprobados en Windows y
  Ubuntu; 91 Pester, 34 Bats Linux y 33 Bats Git Bash/Windows + 1 omitida por
  enlaces reales, cubiertos en Linux. No demuestra comportamiento de los LLM.

- Política híbrida implementada en 4d952c3, CI 39270959007 aprobada en Ubuntu y
  Windows/Git Bash. Guías base, carriers y escenarios verificados. La clasificación
  sigue siendo una instrucción al agente, sin garantía automática de obediencia.
- Diagramas workflow conservan bytes en Git; validación con checkout Windows.
  El grafo AST no sustituye las reglas vigentes ni una actualización semántica
  de documentos. No se modificó predictivo_ml_leads en esta entrega.

- Commit del compositor `f31b966` publicado en `feat/reliable-profile-composer`
  con autorización del usuario. Acceso del conector habilitado 2026-10-19.
  La CI de `86f011d` pasó PowerShell y canarios Windows; falló en invocaciones
  directas sin permiso Linux y en preparación Git Bash/Windows. Corregido en
  d34cc4c: CI 39255792897 aprobada en Ubuntu y Windows, incluido Git Bash.
  Estos resultados corresponden a las revisiones anteriores a la integración.
- CI Ubuntu invoca scripts con Bash explícito: los archivos se guardan con modo
  Git 100644 y la invocación directa no es portable a Linux.

- Cambio `reliable-profile-composer`: requisitos aprobados 2026-10-17 y diseño /
  implementación-verificación aprobados 2026-10-18. Contexto por rol/perfiles,
  preflight completo, smoke y CI consumidor, sondas reales y estados explícitos.
  Verificación de plataformas completada: CI d34cc4c y 4d952c3 aprobada. Consultar
  `specs/reliable-profile-composer/harness.md` para evidencia y límites del entorno.
- Actualizar el grafo no prueba que cubra todo el repo ni cuantifica ahorro total
  de tokens. La metadata observada del launcher global Graphify es 0.4.23 y uv es
  0.11.25; difieren de los pines, sin actualización global automática en este cambio.

- Cambio `minimal-codex-github-harness` implementado con perfiles `base`, `codex`, `openspec`, `graphify` y `archify`.
- Workflows nativos de OpenSpec, integración de Archify, escáner de secretos en `lint` y guardrails de OneDrive/disco activos y verificados con tests automatizados.
