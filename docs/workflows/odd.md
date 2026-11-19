# ODD: cambios acotados de bajo riesgo

Esta es la ruta breve de [AGENTS.md](../../AGENTS.md). Se inspira en la
proporcionalidad de ODD; no instala Gentle AI, Engram ni otro orquestador.

## Objetivo y aceptación

Usar el pedido o conversación como contrato breve: qué cambia, qué debe pasar
y qué queda fuera. Una tarea ya autorizada no necesita nuevas aprobaciones por
cada paso. Preguntar solamente por decisiones que faltan. Las investigaciones
sin permiso de cambio se mantienen en lectura.

## Implementar y verificar

Revisar contexto relevante, confirmar que el cambio es local, reversible y
verificable e implementar una unidad coherente. No crear por rutina odd/tasks,
proposal, design ni cuatro documentos de SDD para una tarea pequeña.

Ejecutar los checks pertinentes. Una modificación de lógica necesita pruebas
de comportamiento y una regresión reproducible, su cobertura. Para estilo o
texto, puede corresponder una comprobación visual/interactiva y de accesibilidad,
además de los checks existentes aplicables. El criterio no es escribir un test
por línea ni reentrenar el modelo por cambiar un texto.

Durante la iteración usar checks focalizados; al entregar, ejecutar el alcance
de verificación que corresponda y conservar CI general al publicar. No omitir
una comprobación requerida por llamarlo ODD. Si un check no aplica o no puede
ejecutarse, registrar motivo y riesgo, con FAIL/PENDIENTE cuando corresponda.

## Cierre y cambio de ruta

Registrar cambio, evidencia y límites en conversación/commit/PR existente; no
crear un artefacto persistente adicional por obligación. La entrega respeta el
alcance y las autorizaciones del usuario.

Si aparece una dependencia sensible, un permiso, un nuevo contrato de datos o
modelo, o incertidumbre sustancial, detener la ampliación e iniciar la
[ruta SDD](sdd.md) para ese alcance antes de implementarlo. Conservar evidencia
del trabajo ya hecho y no presentar la ampliación como parte del permiso inicial.

## Ejemplos y fronteras

- Botón que abre un panel existente, CSS o texto: ODD si cumple los criterios.
- Botón que conecta una función aprobada y probada, sin cambiar permisos o
  contrato: puede ser ODD. Verificar el comportamiento afectado.
- Nueva operación de borrado o exportación con permisos por definir: SDD.
- Cambio del predictor cubierto por un protocolo aprobado: ejecutar bajo ese
  SDD, con su evidencia; no recrear toda la documentación.

Elegir por comportamiento/riesgo, no por el nombre del botón o cantidad de líneas.
