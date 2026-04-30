# 🧠 CONTEXTO DEL PROYECTO — predictivo_ml_leads
<!-- 
  INSTRUCCIÓN PARA ANTIGRAVITY:
  Lee este archivo PRIMERO al inicio de cada sesión de trabajo en este proyecto.
  Evita explorar la estructura de directorios si ya está documentada aquí.
  Solo vuelve a leer archivos individuales cuando debas modificarlos.
-->

## Resumen ejecutivo
App Streamlit de clasificación de leads (Hot 🔥 / Cold ❄️) para Renault México.  
Modelo: **Random Forest** local (`models/best_model.joblib`).  
Sin LLM externo. Sin API de pago. Todo corre offline.

---

## Stack técnico
| Componente | Tecnología |
|-----------|------------|
| UI | Streamlit |
| Modelo | Random Forest (scikit-learn) |
| Artefactos | joblib (`.joblib`) |
| Encoding | category_encoders 2.9.0 |
| Explicabilidad | SHAP |
| Memoria | SQLite (`data/memory/history.db`) |
| Entorno | `.venv` (Python 3.14) |

---

## Estructura de archivos clave
```
predictivo_ml_leads/
├── streamlit_app/
│   ├── app.py                     ← Punto de entrada Streamlit
│   ├── core/
│   │   ├── inference.py           ← Pipeline completo: Excel → predicciones
│   │   ├── memory.py              ← Sistema SQLite: save_session, get_sessions
│   │   └── shap_explainer.py      ← Explicaciones SHAP por lead
│   └── components/
│       ├── upload.py              ← Widget de carga de archivo
│       ├── results_grid.py        ← Tabla filtrable de resultados
│       ├── lead_detail.py         ← Detalle individual + gráfico SHAP
│       ├── model_report.py        ← Métricas y reporte del modelo
│       └── history.py             ← Tab historial (sesiones SQLite)
├── src/
│   ├── train.py                   ← Entrenamiento del modelo
│   ├── predict.py                 ← Inferencia batch (CLI)
│   └── save_artifacts.py          ← Genera preprocessing_config.joblib
├── models/
│   ├── best_model.joblib          ← Modelo entrenado
│   └── preprocessing_config.joblib ← Encoders, umbrales, columnas
├── data/memory/history.db         ← Base de datos SQLite de historial
├── configs/                       ← Configuraciones YAML
├── requirements.txt
└── CONTEXTO.md                    ← Este archivo ✅
```

---

## Pipeline de inferencia (`inference.py`)
1. `fix_encoding()` — corrige nombres de columnas con encoding roto
2. `extract_date_features()` — extrae mes, día, hora, día_semana desde "Fecha de creación"
3. `select_columns()` — mapea nombres de columnas del Excel a nombres internos
4. `clean_nulls()` — imputa nulos por tipo de columna
5. `apply_feature_engineering()` — target encoding (concesionario), one-hot, alinea columnas
6. `model.predict_proba()` — umbral 0.35 por defecto
7. `save_session()` — guarda en SQLite

---

## Sistema de memoria SQLite
- **Tablas:** `sessions` (metadatos) + `predictions` (resultados individuales)
- **DB path:** `data/memory/history.db`
- **Init:** automático en arranque de la app (`init_db()`)
- **Funciones:** `save_session`, `get_sessions`, `get_session_predictions`, `get_trend_data`, `delete_session`

---

## Cómo arrancar el servidor
```powershell
# Desde la raíz del proyecto
.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py
# URL: http://localhost:8501
```

---

## Variables / parámetros críticos
| Variable | Valor | Dónde se usa |
|---------|-------|-------------|
| `umbral` | 0.35 | `artifacts["umbral"]` en `inference.py` |
| `MODEL_PATH` | `models/best_model.joblib` | `inference.py` |
| `ARTIFACTS_PATH` | `models/preprocessing_config.joblib` | `inference.py` |
| `DB_PATH` | `data/memory/history.db` | `memory.py` |

---

## Columnas esperadas en el Excel de entrada
| Columna original (Excel) | Nombre interno |
|------------------------|---------------|
| Fecha de creación | → mes/dia/hora/dia_semana_creacion |
| Nombre del formulario | nombre_formulario |
| Campaña | campana |
| Origen de creación | origen_creacion |
| Vehículo de interés | vehiculo_interes |
| Nombre corto concesión | concesionario |
| Cualificación | cualificacion (opcional, para accuracy) |
| Lead ID | lead_id (opcional) |

---

## Decisiones de data leakage documentadas

| Fecha | Variable/Fuente | Acción | Motivo |
|-------|----------------|--------|--------|
| V1 | `plataforma_MX_LEAD_QUALIF` | Eliminada | Cola CRM = respuesta ya asignada (41.2% importancia) |
| Abr 2026 | `MX_Renault_0125_Arkana_Lead` | Leads eliminados | Enviados como Hot sin cualificación humana (+36.6pp vs ONE-PR) |
| Abr 2026 | `MX_Renault_0125_Oroch_Lead` | Leads eliminados | Enviados como Hot sin cualificación humana (+35.2pp vs ONE-PR) |

> Validado con equipo comercial Renault México. Ver `PENDIENTES_VALIDACION.md` para detalle completo.

---

## Estado actual del proyecto
- ✅ App Streamlit funcional con SQLite
- ✅ Tab de Historial con tendencias y exportación CSV
- ✅ Explicabilidad SHAP por lead individual
- ✅ Data leakage de Arkana/Oroch identificado y corregido
- ⚠️ Warnings de Streamlit: reemplazar `use_container_width` → `width` (deprecado en dic 2025)
- ⚠️ `nombre_formulario` y `campana` son redundantes para formularios MX_Renault (relación 1:1)
- 📋 Ver `PENDIENTES_VALIDACION.md` para tareas pendientes
