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
  "MX_LEAD_CHATBOT_QUALIF" o "MX_LEAD_QUALIF") que nunca pasaron por el equipo
  humano de Sales & Marketing. No representan el proceso que queremos predecir.
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
| `vehiculo_interes` | 20 | 0.2% | Imputar con la moda ("KWID") |
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
| 0 — No Hot (Rechazo arg. + no arg.) | 2,640 | 31.3% |

---

## Archivos Generados

| Archivo | Descripción |
|---|---|
| `data/raw/leads_raw.csv` | Dataset original (sin email), encoding latin-1 |
| `data/processed/leads_cleaned.csv` | Dataset limpio, encoding UTF-8, 0 nulos |
| `notebooks/00_data_engineering.ipynb` | Notebook con todo el proceso documentado y ejecutado |
