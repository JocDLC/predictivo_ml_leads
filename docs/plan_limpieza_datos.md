# Plan de Limpieza de Datos — Lead Scoring

## Dataset Original
- **Archivo:** `data/raw/leads_raw.csv`
- **Registros:** 13,516 filas × 27 columnas (la columna email fue removida previamente por PII)
- **Encoding:** latin-1, separador: `;`

---

## Contexto de Negocio

Una agencia en México realiza campañas digitales para Renault. Las personas completan
formularios solicitando información sobre vehículos. Estos leads son contactados por el
equipo de Sales & Marketing, se los filtra con un checklist, y los calificados como
"Contacto interesado" se derivan al concesionario para cerrar la venta.

### Significado de las Cualificaciones:
- **Contacto interesado:** Lead verificado con checklist y enviado al concesionario.
  Estos son por los que cada país paga. **TARGET POSITIVO (1).**
- **Rechazo argumentado:** Se contactó al lead y este indicó que no estaba interesado.
  El motivo se detalla en "Sub-Cualificación".
- **Rechazo no argumentado:** No se pudo contactar al lead (ilocalizable, número errado,
  número falso, etc.). No hubo respuesta del lead, por eso "no argumentado".
- **Cualificación en blanco (NaN):** Leads procesados por un bot (chatbot/WhatsApp)
  que NO fueron tratados por el equipo humano. No son relevantes para el modelo.

---

## Pasos de Limpieza

### PASO 1 — Fix de encoding (latin-1 → UTF-8)
- **Qué:** Leer el CSV con `encoding='latin-1'` y `sep=';'`, luego re-exportar en UTF-8.
- **Por qué:** El archivo original fue exportado desde Excel/CRM con codificación
  latin-1 (ISO-8859-1). Esto corrompe caracteres especiales del español
  (ej: "CualificaciÇün" en vez de "Cualificación"). UTF-8 es el estándar
  universal y necesario para que pandas, scikit-learn y cualquier herramienta
  procesen los datos correctamente.
- **Acción:** Renombrar todas las columnas a nombres limpios en español (snake_case).

### PASO 2 — Eliminar leads del bot (Cualificación en blanco)
- **Qué:** Eliminar todas las filas donde `Cualificación` es NaN (4,506 filas).
- **Por qué:** Son leads manejados exclusivamente por el bot (plataforma
  `MX_LEAD_CHATBOT_QUALIF`) que nunca pasaron por el equipo humano de Sales &
  Marketing. La plataforma `MX_LEAD_QUALIF` corresponde a leads que completaron
  el formulario y sí fueron gestionados por el equipo humano.
  Los leads del bot no representan el proceso que queremos predecir.
- **Registros eliminados:** 4,506 filas.
- **Registros restantes:** 9,010 filas.

### PASO 3 — Crear variable target binaria
- **Qué:** Crear columna `target` → 1 si "Contacto interesado", 0 si cualquier "Rechazo".
- **Por qué:** Nuestro objetivo es predecir si un lead será derivado al concesionario.
  Es una clasificación binaria.
- **Distribución resultante (antes de eliminar duplicados):**
  - 1 (Contacto interesado): 6,058 → 67.2%
  - 0 (Rechazo argumentado + no argumentado): 2,952 → 32.8%

### PASO 4 — Eliminar columnas con data leakage
Columnas que se completan DESPUÉS de la cualificación y que no estarían disponibles
al momento de predecir un lead nuevo:

| Columna | Razón de eliminación |
|---|---|
| `Sub-Cualificación` | Se completa después de calificar. Es consecuencia del target. |
| `Otra información` | Resultado del proceso de cualificación. |
| `Comentario` | Notas de seguimiento post-contacto. |
| `Descripción` | Detalle del chatbot/proceso, posterior. |
| `Fecha reasignación del lead cualificado` | Timestamp posterior a la cualificación. |
| `Primera fecha de acción` | Timestamp de cuando se actuó sobre el lead. |
| `Qualified lead reassignment user` | Usuario que reasignó, dato posterior. |
| `Numero de matricula` | Solo se completa para leads HOT (derivados). Es data leakage directa: si tiene valor → siempre es target=1. |
| `Propietario del candidato` | Se asigna/modifica durante el proceso de gestión, no al ingreso. Podría correlacionar con el resultado. |
| `Último alias modificado` | Usuario interno que modificó el lead durante el proceso, no disponible al ingreso. |
| `Tiempo procesado hasta primer contacto` | Ocurre DESPUÉS del ingreso. Útil para análisis exploratorio y modelo secundario, pero no para predecir al ingreso. |

### PASO 5 — Eliminar columnas sin valor predictivo
| Columna | Razón de eliminación |
|---|---|
| `Lead ID` | Identificador único, cero información predictiva. |
| `Correo electrónico` | Dato personal (PII), no aporta al modelo. Además, casi todos son únicos (11,612 valores distintos). |
| `Tipo de interés` | Constante: todos los registros dicen "Vehículo Nuevo". No aporta información. |
| `Cualificación` | Se reemplaza por la columna `target` binaria. Ya no se necesita la original. |

### PASO 6 — Evaluar columna "Nombre corto de la Concesión"
- **Qué:** Mantener esta columna como feature.
- **Por qué:** Tiene 92 concesionarios distintos. Puede revelar patrones geográficos
  de conversión (ej: ciertos concesionarios convierten más que otros). Se le hará
  encoding en la fase de Feature Engineering.
- **Contexto de negocio:** Cuando la persona completa el formulario, el sistema le
  sugiere el concesionario más cercano. Sin embargo, la persona puede no seleccionar
  ninguno, por lo que el campo queda vacío.
- **Nulos:** Solo 28 (0.2%). Se imputan como "sin_concesion". Nota: un lead que no
  eligió concesionario podría estar menos comprometido, por lo que el nulo en sí
  mismo es una señal predictiva.

### PASO 7 — Evaluar columna "Campaña"
- **Qué:** Evaluar si mantener o simplificar.
- **Por qué:** Tiene 24 valores únicos pero 45% nulos (6,142 NaN). Los nulos
  corresponden mayormente a leads que entraron por canales orgánicos (no por campaña
  paga). Se puede crear una feature binaria `tiene_campana` (1/0) o imputar como
  "sin_campana".

### PASO 8 — Manejo de nulos en features restantes
Nulos medidos DESPUÉS de filtrar leads del bot (9,010 filas):

| Columna | Nulos | % | Estrategia |
|---|---|---|---|
| `campana` | 4,588 | 50.9% | Imputar como "sin_campana" (lead orgánico, sin campaña paga) |
| `origen` | 887 | 9.8% | Imputar como "desconocido" |
| `vehiculo_interes` | 20 | 0.2% | Imputación diferenciada: **Hot Leads** (1 caso) → moda, porque el analista debió completarlo (error humano). **Cold Leads** (19 casos) → "sin_vehiculo", persona indecisa o formulario no obligatorio. |
| `concesion` | 5 | 0.1% | Imputar como "sin_concesion" |

### PASO 9 — Eliminar filas duplicadas
- **Qué:** Identificar y eliminar filas exactamente iguales en todas las columnas.
- **Duplicados encontrados:** 588 filas.
- **Por qué:** Son registros repetidos que sesgarían al modelo (le daría más peso a esos
  leads). Pueden haberse generado por re-envíos del formulario o duplicaciones del CRM.
- **Registros restantes:** 8,422 filas.

### PASO 10 — Columnas finales para el modelo
Después de la limpieza, las features candidatas son:

**Features temporales (pre-procesadas antes del export del CRM):**
El dataset original en Salesforce contenía una columna `Fecha de creación` (datetime
con fecha y hora). Antes de exportar el CSV, esta columna fue descompuesta en 5
componentes individuales para facilitar el análisis de patrones temporales:
- **Hora** → ¿Los leads de madrugada convierten menos que los de horario laboral?
- **Día de semana** → ¿Los leads de fin de semana son más o menos serios?
- **Mes** → Estacionalidad (diciembre vs enero pueden tener comportamientos distintos)
- **Día del mes** → Patrón secundario (quincena, fin de mes, etc.)

| Columna | Tipo | Descripción |
|---|---|---|
| `anio_creacion` | int | Año en que se creó el lead |
| `mes_creacion` | int | Mes de creación |
| `dia_creacion` | int | Día del mes |
| `hora_creacion` | int | Hora del día (0-23) |
| `dia_semana_creacion` | cat | Día de la semana |
| `nombre_formulario` | cat | Tipo de formulario (17 valores) |
| `campana` | cat | Campaña de marketing (24 valores + "sin_campana") |
| `plataforma` | cat | Chatbot vs Manual (2 valores) |
| `origen_creacion` | cat | ONE/Facebook/WhatsApp/RRSS (4 valores) |
| `subtipo_interes` | cat | Tipo de solicitud (5 valores) |
| `vehiculo_interes` | cat | Modelo del auto (16 valores) |
| `concesion` | cat | Nombre del concesionario (92 valores) |
| `origen` | cat | Canal de marketing (15 valores) |
| **`target`** | **int** | **1=Hot Lead (Contacto interesado), 0=Rechazo** |

---

## Resumen de Impacto (Resultados Reales)

| Métrica | Raw | Limpio |
|---|---|---|
| Filas | 13,516 | **8,422** |
| Columnas | 27 | **14** (13 features + target) |
| Valores nulos totales | 83,107 | **0** |
| Filas eliminadas (bot) | — | 4,506 |
| Filas eliminadas (duplicados) | — | 588 |
| Columnas eliminadas | — | 13 (leakage + constantes + IDs) |

| Target | Cantidad | % |
|---|---|---|
| 1 — Hot Lead (Contacto interesado) | 5,782 | 68.7% |
| 0 — Cold Lead (Rechazo arg. + no arg.) | 2,640 | 31.3% |

---

## Transformaciones de Feature Engineering (post-EDA)

> **Nota:** Los pasos 11 a 18 no son arbitrarios. Cada decisión se tomó a partir de los
> descubrimientos del Análisis Exploratorio de Datos (EDA) documentado en
> `01_exploratory_data_analysis(EDA).ipynb`. El EDA analizó distribuciones, tasas de
> conversión por categoría, correlaciones (Pearson y Cramér's V), heatmaps temporales
> y detección de categorías de baja frecuencia. Las justificaciones de cada paso
> referencian los hallazgos específicos que los motivaron.

Transformaciones aplicadas en `02_feature_engineering.ipynb`:

### PASO 11 — Eliminar `anio_creacion`
- **Qué:** Eliminar la columna `anio_creacion` del dataset.
- **Por qué:** El dataset solo contiene datos de **diciembre 2025** (6,358 leads) y
  **enero 2026** (2,064 leads). Con solo 2 valores posibles (2025 y 2026), la feature
  no aporta variabilidad suficiente para que el modelo aprenda patrones generalizables.
  Además, está altamente correlacionada con `mes_creacion` (r = -0.996), lo cual genera
  redundancia. Si en el futuro se agregan más datos con mayor rango temporal, se podría
  reconsiderar.

### PASO 12 — Eliminar `subtipo_interes`
- **Qué:** Eliminar la columna `subtipo_interes` del dataset.
- **Por qué:** El 96.5% de los registros tienen el valor "Solicitud de compra". Las otras
  categorías son: "Solicitud general" (208), "TestDrive Request" (76),
  "Solicitud de estimación" (6) y "Reveal" (3). Una feature donde un solo valor domina
  tan abrumadoramente no le da al modelo información útil para discriminar entre Hot y Cold.
  Su Cramér's V con el target es muy bajo, confirmando la falta de asociación estadística.

### PASO 13 — Crear feature `es_fin_de_semana`
- **Qué:** Nueva columna binaria: 1 si el lead se creó en sábado o domingo, 0 si fue día laboral.
- **Por qué:** El EDA mostró que los leads de fin de semana tienen mayor tasa de conversión
  (sábado 69.9%, domingo 75.1%) vs días laborales (promedio ~65%). La hipótesis es que las
  personas que buscan un auto en fin de semana tienen más tiempo libre y una intención de
  compra más seria. Esta feature simplifica la señal de `dia_semana_creacion` en una variable
  fácil de interpretar para el modelo.

### PASO 14 — Crear feature `franja_horaria`
- **Qué:** Nueva columna categórica que agrupa `hora_creacion` en 4 franjas:
  - `madrugada` (00:00 - 05:59)
  - `manana` (06:00 - 11:59)
  - `tarde` (12:00 - 17:59)
  - `noche` (18:00 - 23:59)
- **Por qué:** El EDA reveló que la madrugada tiene la mayor tasa de conversión (~73-77%),
  mientras que el horario laboral convierte menos (~63-65%). Quien llena un formulario de
  madrugada probablemente tiene un interés más genuino. Agrupar en franjas reduce la
  granularidad de 24 horas a 4 categorías, capturando el patrón sin agregar ruido.

### PASO 15 — Agrupar categorías de baja frecuencia en "otros"
- **Qué:** Para las features `nombre_formulario`, `vehiculo_interes`, `origen`, `campana`
  y `concesion`, se agrupan en "otros" todas las categorías con menos del 1% del total
  de registros (< 84 leads).
- **Por qué:** Las categorías con muy pocos registros generan problemas:
  1. El modelo no tiene suficientes ejemplos para aprender patrones confiables.
  2. Con one-hot encoding, crean columnas casi vacías que consumen dimensionalidad sin aportar.
  3. Pueden causar sobreajuste: el modelo memoriza esos pocos casos en vez de generalizar.
  Agruparlas en "otros" reduce el ruido y la dimensionalidad.

### PASO 16 — Target encoding para `concesion`
- **Qué:** Reemplazar cada concesionario por su **tasa de conversión promedio** calculada
  solo con datos de entrenamiento (para evitar data leakage).
- **Por qué:** Aún después de agrupar categorías de baja frecuencia, `concesion` tiene
  alta cardinalidad. Con one-hot encoding generaría demasiadas columnas sparse. El target
  encoding resume la información de cada concesionario en un solo número (su probabilidad
  histórica de producir un Hot Lead). Los concesionarios no vistos en entrenamiento reciben
  la media global como fallback.
- **Resultado:** `concesion` → `concesion_target_enc` (float entre 0 y 1).

### PASO 17 — One-hot encoding para el resto de categóricas
- **Qué:** Aplicar one-hot encoding (con `drop_first=True`) a:
  `dia_semana_creacion`, `nombre_formulario`, `campana`, `plataforma`,
  `origen_creacion`, `vehiculo_interes`, `origen`, `franja_horaria`.
- **Por qué:** Son features categóricas nominales (sin orden natural). One-hot encoding
  es la forma estándar de representarlas numéricamente para modelos de machine learning.
  Se usa `drop_first=True` para evitar multicolinealidad perfecta (la categoría eliminada
  queda implícita cuando todas las demás son 0).
- **Resultado:** 8 columnas categóricas → 44 columnas binarias (0/1).

### PASO 18 — Split train/test estratificado
- **Qué:** Dividir el dataset en 80% entrenamiento y 20% test, estratificando por `target`.
- **Por qué:** La estratificación garantiza que ambos conjuntos mantengan la misma proporción
  de Hot/Cold (~68.7%/31.3%). Sin esto, podríamos tener por azar un test set con proporción
  diferente, lo que sesgaría la evaluación del modelo.
- **Resultado:**
  - Train: 6,737 filas (68.7% Hot)
  - Test: 1,685 filas (68.7% Hot)
- **Random state:** 42 (fijo para reproducibilidad).

---

## Dataset Final para Modelado

| Métrica | Valor |
|---|---|
| Features totales | **49** (5 numéricas + 44 dummies) |
| Filas train | 6,737 |
| Filas test | 1,685 |
| Nulos | 0 |

### Features numéricas (5):
| Feature | Origen | Descripción |
|---|---|---|
| `mes_creacion` | Original | Mes de creación (1, 12) |
| `dia_creacion` | Original | Día del mes (1-31) |
| `hora_creacion` | Original | Hora del día (0-23) |
| `es_fin_de_semana` | **Nueva** | 1=sábado/domingo, 0=laboral |
| `concesion_target_enc` | **Transformada** | Tasa de conversión del concesionario |

### Features dummy (44):
Generadas por one-hot encoding de: `dia_semana_creacion` (6), `nombre_formulario` (8),
`campana` (8), `plataforma` (1), `origen_creacion` (2), `vehiculo_interes` (8),
`origen` (7), `franja_horaria` (3).

---

## Archivos Generados

| Archivo | Descripción |
|---|---|
| `data/raw/leads_raw.csv` | Dataset original (sin email), encoding latin-1 |
| `data/processed/leads_cleaned.csv` | Dataset limpio, encoding UTF-8, 0 nulos |
| `data/processed/X_train.csv` | Features de entrenamiento (6,737 × 49) |
| `data/processed/X_test.csv` | Features de test (1,685 × 49) |
| `data/processed/y_train.csv` | Target de entrenamiento |
| `data/processed/y_test.csv` | Target de test |
| `notebooks/00_data_engineering.ipynb` | Notebook de limpieza de datos |
| `notebooks/01_exploratory_data_analysis(EDA).ipynb` | Notebook de análisis exploratorio |
| `notebooks/02_feature_engineering.ipynb` | Notebook de feature engineering |
