# SDD: trabajo sustancial o sensible

Aplicar después de elegir ruta en [AGENTS.md](../../AGENTS.md). El enfoque y los
criterios deben estar acordados antes de implementar; esa aprobación puede estar
en el pedido y decisiones vigentes. No repetir confirmaciones para el mismo alcance.
Las dudas de producto o cambios de alcance se consultan antes de ampliarlo.

## Fase 0: contexto pertinente

Revisar instrucciones, memoria y código afectados. Con Graphify activo, consultar
su reporte/grafo y actualizar AST si cambió código. Usar capacidades disponibles;
no imponer Graphify ni OpenSpec al perfil base.

## Fase 1: requisitos

Definir intención, criterios de aceptación, restricciones, no objetivos y preguntas
abiertas. Para cambios no triviales, mantener
`specs/<feature>/{requirements,design,tasks,harness}.md`. Reutilizar el SDD existente
si el cambio ya está cubierto. Acordar las decisiones pendientes con el usuario.

## Fase 2: diseño y tareas

Diseñar sobre requisitos aprobados. Describir arquitectura, contratos, flujo de
datos, errores, observabilidad, privacidad y rollout cuando sean relevantes.
Ordenar tareas pequeñas, asociadas a aceptación y comandos de verificación.
Con Archify activo, documentar visualmente cambios de estructura, flujos o
contratos usando la versión fijada y `ARCHIFY_UPDATE_CHECK_DISABLED=1`.

Con OpenSpec activo y disponible, usar sus workflows nativos
`/opsx:explore`, `/opsx:propose`, `/opsx:apply`, `/opsx:archive` si el cliente los
soporta, o la CLI real. Mantener propuesta, diseño, tareas y delta specs con una
fuente normativa clara; enlazar los requisitos en vez de duplicarlos. Validar
`openspec validate --changes`. Con base, los cuatro archivos specs y terminal
son suficientes; no inventar comandos de otro cliente.

## Fase 3: harness antes de features

Identificar dependencias reproducibles, smoke funcional, linter/formateador,
typecheck cuando aplique, tests, build y CI pertinentes al producto. El harness
de autoría y el smoke del consumidor cumplen propósitos distintos. Resolver lo
necesario para verificar antes de implementar. El doctor evalúa perfiles activos;
reportar las carencias y acciones manuales sin instalaciones implícitas.

## Fases 4 a 6: implementar, verificar y corregir

Trabajar una tarea a la vez. Para un contrato o regresión con prueba determinística
aplicable, observar el fallo antes de corregir y luego el resultado aprobado.
Ejecutar checks focalizados durante la iteración y suites completas pertinentes
al cerrar la funcionalidad. El entrenamiento costoso o uso de datos reales requiere
su propio alcance; no ejecutarlo por rutina para un cambio ajeno al modelo.

Comprobar escenarios de aceptación y registrar evidencia en harness/tareas.
Si falla, identificar causa raíz, corregir mínimamente y repetir la verificación
afectada. No debilitar tests ni afirmar que el análisis de texto prueba la
obediencia futura de un agente.

## Fase 7: cierre

Cerrar aceptación y tareas solo con evidencia, o registrar checks pendientes y
riesgo residual. Sincronizar/archivar únicamente cambios completados de OpenSpec
cuando corresponda; sin OpenSpec conservar los cuatro archivos y el cierre.
Actualizar grafo AST por código y diagramas cuando cambia el contrato/estructura.
Informar resultados y límites; respetar autorización vigente de commit/push y
separar decisiones de merge/publicación si no están concedidas.

Una nueva tarea pequeña fuera de este contrato puede usar la [ruta ODD](odd.md).
Dentro del SDD aprobado, ejecutar la tarea existente sin abrir otro ciclo.
