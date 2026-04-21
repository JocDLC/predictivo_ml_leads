# Predictivo ML Leads — Lead Scoring para Sales & Marketing Automotriz

Proyecto final de Machine Learning e IA. Sistema predictivo de clasificación binaria para determinar la probabilidad de conversión de leads del sector automotriz (Renault México).

## Contexto del Negocio

Una agencia en México realiza campañas de marketing digital para que personas interesadas en vehículos completen un formulario. Estos leads son contactados por un equipo de Sales & Marketing que los filtra. Los leads calificados como "interesados" se derivan al concesionarioario para cerrar la venta.

**Objetivo:** Predecir si un lead será calificado como "Contacto interesado" (derivado al concesionarioario) o será rechazado, optimizando la priorización del equipo de ventas.

---

## Estructura del Proyecto

```bash
predictivo_ml_leads/
├── configs/                # Configuración YAML del mejor modelo
├── data/
│   ├── raw/                # Datos originales sin procesar
│   └── processed/          # Datos limpios y con features
├── deployment/
│   └── mlflow/             # Docker Compose para MLflow
├── models/
│   └── trained/            # Modelo y preprocessor exportados (.pkl)
├── notebooks/
│   ├── 00_data_engineering.ipynb
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
├── src/
│   ├── data/               # Script de limpieza/preprocesamiento
│   ├── features/           # Script de ingeniería de features
│   ├── models/             # Script de entrenamiento
│   └── api/                # FastAPI para servir predicciones
├── streamlit_app/          # UI interactiva
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.11+
- Git
- Docker Desktop (para MLflow)

## Configuración del Entorno

```bash
git clone <repo-url>
cd predictivo_ml_leads
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Flujo de Trabajo

1. **Data Engineering** — Limpieza, fix encoding, manejo de nulos
2. **EDA** — Análisis exploratorio, distribuciones, correlaciones
3. **Feature Engineering** — Creación de variables, encoding, scaling
4. **Experimentación** — Competencia de modelos (LogReg, RF, GB, XGB, SVM)
5. **Despliegue** — API FastAPI + UI Streamlit

## Modelos Candidatos

| Modelo | Tipo |
|---|---|
| LogisticRegression | Baseline lineal |
| RandomForestClassifier | Ensemble bagging |
| GradientBoostingClassifier | Ensemble boosting |
| XGBClassifier | Gradient boosting optimizado |
| SVC | Support Vector Machine |

## Métricas de Evaluación

- Accuracy, Precision, Recall, F1-Score, AUC-ROC

---

## Dataset

- **Fuente:** CRM de leads automotriz (Renault México)
- **Registros:** 13,516 leads
- **Target:** `Cualificación` → Binario (Contacto interesado = 1, Rechazo = 0)
- **Features:** 28 columnas originales (temporales, campaña, vehículo, origen, concesionario, etc.)

---

**Proyecto Final — Curso de Machine Learning e IA**
