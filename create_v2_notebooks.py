import nbformat as nbf
import os

def create_notebook(path, cells):
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for cell_type, content in cells:
        if cell_type == 'markdown':
            nb_cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            nb_cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = nb_cells
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generado con éxito: {path}")

# ==============================================================================
# 01 — EXPLORATORY DATA ANALYSIS V2
# ==============================================================================
eda_cells = [
    ('markdown', """# 01 — Exploratory Data Analysis V2 (2025)

**Objetivo:** Analizar el comportamiento de los leads reales del año 2025, eliminando el ruido estadístico de los bots y meses anómalos identificados en la V1.

**Comparativa V1 vs V2:**
- **V1:** Datos mezclados 2022-2024. Sesgo por chatbot (68% conversión).
- **V2:** Solo datos 2025. Leads 100% humanos. Conversión real cercana al 37%."""),
    ('code', """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/leads_cleaned.csv")
print(f"Dataset V2 cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""),
    ('markdown', """## 1. Distribución del Target (Real vs Sesgada)"""),
    ('code', """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df["target"].value_counts()
counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black", ax=axes[0])
axes[0].set_title("Conteo de Hot vs Cold Leads (V2)")
df["target"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"], ax=axes[1])
plt.show()"""),
    ('markdown', """## 2. Análisis Temporal y Conversión"""),
    ('code', """fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# Volumen por hora
sns.countplot(data=df, x='hora_creacion', color='steelblue', ax=axes[0,0])
axes[0,0].set_title("Volumen por Hora")
# Conversión por hora
conv_hora = df.groupby('hora_creacion')['target'].mean()
conv_hora.plot(kind='line', marker='o', color='seagreen', ax=axes[0,1])
axes[0,1].set_title("Tasa de Conversión por Hora")
# Heatmap Día/Hora
pivot_vol = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='count')
sns.heatmap(pivot_vol, cmap="YlGnBu", ax=axes[1,0])
axes[1,0].set_title("Heatmap Volumen Día/Hora")
plt.tight_layout()
plt.show()"""),
    ('markdown', """## 3. Correlación de Variables Numéricas"""),
    ('code', """plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación V2")
plt.show()""")
]

# ==============================================================================
# 02 — FEATURE ENGINEERING V2
# ==============================================================================
fe_cells = [
    ('markdown', """# 02 — Feature Engineering V2

**Objetivo:** Transformar el dataset limpio en un dataset listo para modelado utilizando técnicas modernas de encoding.

## 1. Eliminar features sin valor predictivo (Data Leakage)
Igual que en V1, descartamos `anio_creacion` (varianza cero en 2025) y `subtipo_interes` (información post-cualificación). 

## 2. Crear nuevas features derivadas
Creamos `es_fin_de_semana` y `franja_horaria` basándonos en los hallazgos del EDA sobre leads nocturnos y de fin de semana.

## 3. Agrupar categorías de baja frecuencia
Para evitar el sobreajuste, agrupamos categorías con < 1% de presencia en la etiqueta "otros".

## 4. Split Train/Test Estratificado
Dividimos el dataset (80/20) antes de cualquier encoding para evitar que el modelo "vea" el futuro."""),
    ('code', """import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/leads_cleaned.csv")
df = df.drop(columns=["subtipo_interes"])

# Features derivadas
df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(["sábado", "domingo"]).astype(int)
def clasificar_franja(hora):
    if 0 <= hora < 6: return "madrugada"
    elif 6 <= hora < 12: return "manana"
    elif 12 <= hora < 18: return "tarde"
    return "noche"
df["franja_horaria"] = df["hora_creacion"].apply(clasificar_franja)

# Split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""),
    ('markdown', """## 5. Bayesian Target Encoding (Reemplaza al método manual de V1)

### 5.1 ¿Por qué ya no usamos el "Encoding de contribución" manual?
En la **V1**, creábamos manualmente dos columnas (`_contribucion` y `_tasa_suavizada`) aplicando una fórmula de suavizado para evitar que categorías raras con 100% de conversión engañaran al modelo.

En la **V2**, utilizamos la librería oficial `TargetEncoder(smoothing=100)`. Esta aplica el **Suavizado Bayesiano** de forma automática y matemáticamente robusta:
1.  **Reduce la dimensionalidad:** Pasamos de 48 columnas (One-Hot) a solo 11 features densas.
2.  **Evita el Overfitting:** Penaliza automáticamente a los concesionarios o campañas con pocos leads.
3.  **Mayor Poder Predictivo:** Captura la probabilidad real de conversión en una sola variable numérica."""),
    ('code', """# Aplicando el encoding moderno
cat_cols = ["vehiculo_interes", "origen", "nombre_formulario", "campana", "concesionario"]
encoder = TargetEncoder(cols=cat_cols, smoothing=100)

X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)

# One-Hot para el resto de baja cardinalidad
X_train_final = pd.get_dummies(X_train_enc, drop_first=True, dtype=int)
X_test_final = pd.get_dummies(X_test_enc, drop_first=True, dtype=int).reindex(columns=X_train_final.columns, fill_value=0)

print(f"Features finales: {X_train_final.shape[1]} (V1 tenía 48)")"""),
    ('code', """# Exportar
X_train_final.to_csv("../data/processed/X_train_v2.csv", index=False)
X_test_final.to_csv("../data/processed/X_test_v2.csv", index=False)
y_train.to_csv("../data/processed/y_train_v2.csv", index=False)
y_test.to_csv("../data/processed/y_test_v2.csv", index=False)""")
]

# ==============================================================================
# 03 — MODELING V2
# ==============================================================================
mod_cells = [
    ('markdown', """# 03 — Modelado Predictivo V2

**Objetivo:** Entrenar el modelo ganador utilizando el pipeline simplificado de 11 features.

**Diferencias V1 vs V2:**
- **V1:** Random Forest con 48 features (One-Hot). Propenso a ruido en categorías pequeñas.
- **V2:** Modelos basados en árboles sobre features Target-Encoded. Mayor estabilidad y velocidad."""),
    ('code', """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

X_train = pd.read_csv("../data/processed/X_train_v2.csv")
X_test = pd.read_csv("../data/processed/X_test_v2.csv")
y_train = pd.read_csv("../data/processed/y_train_v2.csv")["target"]
y_test = pd.read_csv("../data/processed/y_test_v2.csv")["target"]"""),
    ('markdown', """## 1. Entrenamiento y Comparativa de Modelos"""),
    ('code', """models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    results.append({"Modelo": name, "ROC-AUC": auc})

df_res = pd.DataFrame(results)
print(df_res.sort_values(by="ROC-AUC", ascending=False))"""),
    ('markdown', """## 2. Curvas ROC Comparativas"""),
    ('code', """fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
plt.title("Comparativa de Curvas ROC (V2)")
plt.show()"""),
    ('markdown', """## 3. Matrices de Confusión Individuales"""),
    ('code', """fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"CM: {name}")
plt.show()"""),
    ('code', """# Guardar el mejor modelo
import joblib
joblib.dump(models["Random Forest"], "../models/best_model.joblib")""")
]

# ==============================================================================
# 04 — EVALUATION V2
# ==============================================================================
eval_cells = [
    ('markdown', """# 04 — Evaluación Detallada V2

**Objetivo:** Analizar el rendimiento del modelo final bajo métricas de negocio y explicar sus decisiones con SHAP.

**Contenido:**
1. Análisis de Umbral de Decisión.
2. Curva Precision-Recall.
3. Análisis de Errores (Falsos Positivos/Negativos).
4. Interpretabilidad Global y Local con SHAP."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

model = joblib.load("../models/best_model.joblib")
X_test = pd.read_csv("../data/processed/X_test_v2.csv")
y_test = pd.read_csv("../data/processed/y_test_v2.csv")["target"]
y_probs = model.predict_proba(X_test)[:, 1]"""),
    ('markdown', """## 1. Análisis de Umbral de Negocio (0.35)"""),
    ('code', """thresholds = [0.3, 0.35, 0.4, 0.5, 0.6]
print(f"{'Umbral':<10} | {'Precision':<10} | {'Recall':<10}")
for t in thresholds:
    preds = (y_probs >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    print(f"{t:<10.2f} | {precision_score(y_test, preds):<10.4f} | {recall_score(y_test, preds):<10.4f}")

# Seleccionamos 0.35 para priorizar captura de leads
y_pred_final = (y_probs >= 0.35).astype(int)"""),
    ('markdown', """## 2. Curva Precision-Recall"""),
    ('code', """precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall V2')
plt.show()"""),
    ('markdown', """## 3. Interpretabilidad con SHAP (Global)"""),
    ('code', """explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot (Beeswarm)
plt.title("SHAP Global: Impacto de Features en la Conversión")
shap.summary_plot(shap_values[1], X_test)"""),
    ('markdown', """## 4. Análisis de Lead Individual (Waterfall)"""),
    ('code', """# Seleccionamos un lead aleatorio del test set
idx = 10
print(f"Probabilidad del modelo: {y_probs[idx]:.2%}")
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][idx], feature_names=X_test.columns)""")
]

# ==============================================================================
# EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    create_notebook("notebooks/01_exploratory_data_analysis_v2.ipynb", eda_cells)
    create_notebook("notebooks/02_feature_engineering_v2.ipynb", fe_cells)
    create_notebook("notebooks/03_modeling_v2.ipynb", mod_cells)
    create_notebook("notebooks/04_evaluation_v2.ipynb", eval_cells)
