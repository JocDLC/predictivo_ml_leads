# Predictivo ML Leads — Lead Scoring para Renault México

Sistema predictivo de clasificación binaria (Hot / Cold) para priorizar leads del sector automotriz. Proyecto final de Machine Learning e IA.

## Contexto del Negocio

Renault México recibe leads desde formularios web, plataformas de marketing y otras fuentes digitales. Sin un mecanismo automático de priorización, el equipo comercial dedica tiempo equivalente a oportunidades con alta intención de compra y a contactos fríos o de baja calidad.

**Objetivo:** Clasificar automáticamente cada lead como **Hot** (alta probabilidad de conversión) o **Cold**, y explicar la predicción con SHAP para que el equipo comercial confíe en el modelo.

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Backend / lógica | Python 3.13, scikit-learn |
| Frontend / UI | Streamlit + Plotly |
| Modelo | GradientBoostingClassifier (.joblib) |
| Explicabilidad | SHAP (TreeExplainer) |
| Base de datos | SQLite (historial local) |
| Encoding | category_encoders (Bayesian Target Encoding) |

---

## Estructura del Proyecto

```
predictivo_ml_leads/
├── data/
│   ├── raw/                           → Insumos crudos y snapshots locales
│   ├── processed/                     → leads_cleaned.csv, X_train_v2.csv, X_test_v2.csv, y_train_v2.csv, y_test_v2.csv
│   ├── memory/history.db              → Historial SQLite de la app
│   └── test/                          → Archivos de prueba para inferencia
├── models/
│   ├── best_model.joblib              → GradientBoostingClassifier desplegado
│   └── preprocessing_config.joblib    → Artefactos de preprocesamiento (umbral legacy 0.35)
├── notebooks/
│   ├── 00_data_engineering_v2_2025.ipynb
│   ├── 01_exploratory_data_analysis_v2.ipynb
│   ├── 02_feature_engineering_v2.ipynb
│   ├── 03_modeling_v2.ipynb
│   ├── 04_evaluation_v2.ipynb
│   └── v1/                            → Notebooks legacy de referencia histórica
├── streamlit_app/
│   ├── app.py                         → Entrada principal de la app
│   ├── core/
│   │   ├── inference.py               → Pipeline de inferencia vigente
│   │   ├── memory.py                  → Persistencia SQLite
│   │   ├── shap_explainer.py          → Explicación SHAP por lead
│   │   └── theme.py                   → CSS custom para temas claro/oscuro
│   └── components/
│       ├── upload.py                  → Widget de carga de archivos
│       ├── results_grid.py            → Grilla de resultados con filtros
│       ├── lead_detail.py             → Detalle individual con SHAP
│       ├── model_report.py            → Reporte completo del modelo (5 tabs)
│       └── history.py                 → Historial de sesiones
├── src/
│   ├── predict.py                     → Flujo CLI / soporte programático
│   ├── save_artifacts.py              → Generación de preprocessing_config.joblib
│   └── train.py                       → Script de entrenamiento (legacy RF)
├── graphify-out/                      → Análisis de grafos del código
├── PENDIENTES_VALIDACION.md           → Estado de validaciones de leakage
├── CONTEXTO.md                        → Contexto general del proyecto
├── requirements.txt                   → Dependencias Python
└── README.md
```

---

## Métricas del Modelo (Test — Umbral 0.325)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.7828 |
| Accuracy | 0.6947 |
| Precision (Hot) | 0.5604 |
| Recall (Hot) | 0.6992 |
| F1 (Hot) | 0.6221 |

---

## Dataset V2

- **Fuente:** Exportación Salesforce CRM — Leads México (2025)
- **Registros limpios:** 58,546 leads (tras excluir chatbot, meses anómalos, leakage Arkana/Oroch y duplicados)
- **Target:** `target` → 1 = Hot Lead (Contacto interesado), 0 = Cold Lead
- **Features finales:** 22 (temporales, vehículo, origen, concesionario, formulario, campaña — con Bayesian Target Encoding y One-Hot)
- **Split:** 80/20 estratificado (train: 46,836 — test: 11,710)

---

## ⚡ Instalación rápida

### 1. Clonar el repositorio
```powershell
git clone https://github.com/JocDLC/predictivo_ml_leads.git
cd predictivo_ml_leads
```

### 2. Crear entorno virtual e instalar dependencias
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Levantar la app Streamlit
```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py
```
Acceder en: **http://localhost:8501**

### Requisitos del sistema
- Python 3.11+ (probado con 3.13)
- Git
- ~500 MB de espacio (dependencias)

---

## Flujo de Trabajo (Notebooks V2)

1. **00 — Data Engineering** → Limpieza, fix encoding, exclusión de leakage, filtro 2025
2. **01 — EDA** → Análisis exploratorio, distribuciones, correlaciones, patrones temporales
3. **02 — Feature Engineering** → Bayesian Target Encoding, One-Hot, split estratificado
4. **03 — Modeling** → Comparación de 4 modelos (LogReg, RF, GB, LightGBM) → Ganador: GradientBoosting
5. **04 — Evaluation** → Métricas, matriz de confusión, comparación de umbrales, SHAP

---

## Modelos Evaluados

| Modelo | ROC-AUC | F1 | Resultado |
|---|---|---|---|
| Logistic Regression | 0.742 | 0.513 | Baseline |
| Random Forest | 0.770 | 0.573 | — |
| **Gradient Boosting** | **0.779** | **0.586** | **★ Desplegado** |
| LightGBM | 0.777 | 0.581 | — |

---

**Proyecto Final — Curso de Machine Learning e IA — v2.2 (Mayo 2026)**
