# Perfil opcional: series temporales

Estado: preparado para un caso futuro, inactivo en el contexto de autoría actual.
Incluye reglas y contratos; no incluye un motor de forecasting ni dependencias ML.
Solo se carga al seleccionar `ml-timeseries` o `all`. No modifica otros proyectos.

## Antes de elegir un modelo

Crear el SDD del caso en `specs/<feature>/`: requirements, design, tasks y harness.
Definir objetivo, identificador de serie/entidad, origen de predicción, horizonte,
frecuencia, zona horaria y cuándo estará disponible cada dato. Diferenciar fecha
del evento de fecha de disponibilidad. Acordar manejo de huecos, duplicados y
revisiones, métrica operativa y criterio de éxito. Elegir librerías y fijar sus
versiones recién con ese contexto.

## Contrato de evaluación

- Separar entrenamiento, validación y test en orden temporal; reservar un test
  final sin usarlo para seleccionar el modelo. No mezclar aleatoriamente pasado
  y futuro de una misma serie. Evaluar mediante backtesting por varios orígenes.
- Definir ventana expansiva o móvil según el caso. Si hace falta separación
  entre ventanas, calcularla según horizonte, disponibilidad de etiquetas y
  variables; no asumir un tamaño universal. Para datos irregulares, definir
  ventanas por fechas. `TimeSeriesSplit` requiere muestras igualmente espaciadas
  para comparar duraciones de test y expresa `gap` en cantidad de muestras.
- Ajustar imputación, escalado y selección de variables exclusivamente con el
  entrenamiento de cada fold. Calcular lags y agregados con información pasada
  disponible en cada origen. Una covariable futura es válida solo si realmente
  se conoce al predecir, por ejemplo el calendario.
- En múltiples series, elegir cortes temporales y separación de entidades según
  si se pronostican entidades conocidas o nuevas. Evitar cruces que filtren datos.
- Comparar contra un baseline ingenuo y, si hay estacionalidad justificada, uno
  estacional, con los mismos horizontes y orígenes. Reportar métricas por horizonte
  y serie y acordar la agregación. Evitar MAPE cuando ceros o valores cercanos a
  cero lo vuelvan inadecuado.

## Checks que debe agregar el futuro producto

Usar fixtures sintéticos para probar cortes cronológicos, disponibilidad de
variables, cálculo de lags y ajuste de transformaciones por fold. Verificar que
entrenamiento e inferencia producen las mismas features. Registrar configuración,
ventanas/cutoff, huella de los datos, versiones y semilla cuando corresponda.
Fijar condiciones de fallo para fugas y una regresión respecto del baseline.
Los datos de trabajo requieren la política corporativa aplicable antes de usar
servicios externos; este perfil no concede ese permiso.

Los smoke actuales solo prueban que el perfil se compone y conserva sus recursos.
No validan un modelo ni prueban que una predicción sea útil.

Referencia: [TimeSeriesSplit, documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
