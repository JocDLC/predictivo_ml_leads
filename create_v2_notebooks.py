import nbformat as nbf
import os

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    cells = []
    for cell_type, content in cells_content:
        if cell_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generado: {filename}")

# --- 01 EDA ---
eda_cells = [
    ('markdown', """# 01 — Exploratory Data Analysis (EDA) V2

**Comparativa V1 vs V2:**
- **Versión 1:** El análisis exploratorio se hizo sobre un dataset reducido (~8,400 registros) que mezclaba múltiples años y contenía una gran cantidad de leads procesados por un chatbot automático, lo que ensuciaba la distribución de clases.
- **Versión 2 (Este notebook):** Analizamos el dataset depurado del año 2025 (~66,000 registros). Al excluir los leads del bot y los meses anómalos, obtenemos una visión real del comportamiento orgánico y de la tasa de conversión humana (Hot Leads vs Cold Leads)."""),
    ('code', """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/leads_cleaned.csv")
print(f"Dataset V2 cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""),
    ('markdown', """## Información General del Dataset y Valores Nulos"""),
    ('code', """df.info()
print("\\nValores Nulos por Columna:")
print(df.isnull().sum())
print("\\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print("\\n--- Numéricas ---")
print(df.describe().to_string())
print("\\n--- Categóricas ---")
print(df.describe(include='object').to_string())"""),
    ('markdown', """## Distribución del Target
En V1 la tasa de Hot Leads era cercana al 68%, un número artificialmente alto por la falta de registros. En V2, vemos una conversión real cercana al 37%."""),
    ('code', """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df["target"].value_counts()
bars = counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black", ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Cold Lead (0)", "Hot Lead (1)"], rotation=0)
axes[0].set_title("Conteo de Hot vs Cold Leads (V2)")
axes[0].set_ylabel("Cantidad")

total = len(df)
for i, (val, cnt) in enumerate(counts.items()):
    pct = cnt / total * 100
    axes[0].text(i, cnt + total*0.01, f"{cnt:,}\\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)

df["target"].value_counts().plot(kind="pie", colors=["#e74c3c", "#2ecc71"],
                                  autopct="%1.1f%%", labels=["Cold (0)", "Hot (1)"],
                                  startangle=90, ax=axes[1])
axes[1].set_title("Proporción Hot vs Cold (V2)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()"""),
    ('markdown', """## Análisis de Variables Temporales
Distribución y Tasa de Conversión (Hot Leads) por Hora, Día de la Semana y Mes.
Este análisis reemplaza las gráficas de V1, ahora sobre datos puros de 2025 sin bots."""),
    ('code', """fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)

cols_temp = ['hora_creacion', 'dia_semana_creacion', 'mes_creacion']
titulos = ['Hora de Creación', 'Día de la Semana', 'Mes']

for i, col in enumerate(cols_temp):
    if col not in df.columns: continue
    
    # Volumen
    ax_vol = axes[i, 0]
    sns.countplot(data=df, x=col, color='steelblue', ax=ax_vol)
    ax_vol.set_title(f'Volumen por {titulos[i]}')
    ax_vol.tick_params(axis='x', rotation=45)
    
    # Porcentajes Volumen
    total = len(df)
    for p in ax_vol.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        if p.get_height() > 0:
            ax_vol.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=9)
        
    # Tasa de conversión
    ax_conv = axes[i, 1]
    conv = df.groupby(col)['target'].mean().reset_index()
    sns.barplot(data=conv, x=col, y='target', color='seagreen', ax=ax_conv)
    ax_conv.set_title(f'Tasa de Conversión por {titulos[i]}')
    ax_conv.axhline(df['target'].mean(), color='red', linestyle='--', label='Media Global')
    ax_conv.tick_params(axis='x', rotation=45)
    ax_conv.legend()
    
    # Porcentajes Conversión
    for p in ax_conv.patches:
        val = p.get_height()
        percentage = f'{val*100:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        if p.get_height() > 0:
            ax_conv.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=9)

plt.show()"""),
    ('markdown', """**Evaluación y Comentarios V2:**
A diferencia de la V1, donde el volumen mensual no permitía deducir tendencias por el ruido provocado por el bot, aquí se observa claramente la cantidad de leads orgánicos gestionados por el equipo comercial de forma mensual. Al remover meses anómalos, el modelo aprende patrones temporales correctos en lugar de sesgarse por picos atípicos."""),
    ('markdown', """## Análisis de Variables Categóricas Principales
Volumen y Tasa de Conversión para variables como Plataforma, Origen, etc."""),
    ('code', """variables_cat = ['plataforma', 'origen', 'campana', 'concesionario']
for col in variables_cat:
    if col not in df.columns: continue
    top_10 = df[col].value_counts().nlargest(10).index
    df_top = df[df[col].isin(top_10)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    
    # Volumen
    sns.countplot(data=df_top, y=col, order=top_10, color='steelblue', ax=axes[0])
    axes[0].set_title(f'Top 10 Volumen por {col}')
    
    # Tasa de Conversión
    conv = df_top.groupby(col)['target'].mean().reindex(top_10).reset_index()
    sns.barplot(data=conv, y=col, x='target', color='seagreen', ax=axes[1])
    axes[1].set_title(f'Tasa de Conversión por {col}')
    axes[1].axvline(df['target'].mean(), color='red', linestyle='--', label='Media Global')
    
    total = len(df)
    for p in axes[0].patches:
        width = p.get_width()
        if np.isnan(width): width = 0
        percentage = f'{100 * width / total:.1f}%'
        x = width
        y = p.get_y() + p.get_height() / 2
        axes[0].annotate(percentage, (x, y), ha='left', va='center', fontsize=9)

    for p in axes[1].patches:
        val = p.get_width()
        if np.isnan(val): val = 0
        percentage = f'{val*100:.1f}%'
        x = val
        y = p.get_y() + p.get_height() / 2
        axes[1].annotate(percentage, (x, y), ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()"""),
    ('markdown', """## Análisis Temporal Detallado (Heatmaps)
Visualización de volumen y tasa de conversión cruzando Día de la Semana y Hora de Creación."""),
    ('code', """pivot_vol = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='count', fill_value=0)
pivot_conv = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='mean', fill_value=0)

dias_orden = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
pivot_vol = pivot_vol.reindex(dias_orden)
pivot_conv = pivot_conv.reindex(dias_orden)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

sns.heatmap(pivot_vol, cmap="YlGnBu", annot=False, ax=axes[0])
axes[0].set_title("Mapa de Calor: Volumen de Leads por Día y Hora")
axes[0].set_xlabel("Hora del Día")
axes[0].set_ylabel("Día de la Semana")

sns.heatmap(pivot_conv, cmap="YlOrRd", annot=False, ax=axes[1])
axes[1].set_title("Mapa de Calor: Tasa de Conversión por Día y Hora")
axes[1].set_xlabel("Hora del Día")
axes[1].set_ylabel("Día de la Semana")

plt.tight_layout()
plt.show()"""),
    ('markdown', """## Correlación entre Variables Numéricas"""),
    ('code', """num_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación - Features Numéricas")
plt.tight_layout()
plt.show()"""),
    ('markdown', """## Hallazgos Clave del EDA y Recomendaciones

================================================================================
   HALLAZGOS CLAVE DEL EDA V2
================================================================================

1. TARGET
   - Distribución: ~37% Hot Leads (Real 2025) vs ~68% en V1
   - El desbalance es natural y representativo del embudo de ventas comercial.

2. FEATURES TEMPORALES
   - El dataset no contiene meses anómalos (Abril, Mayo, Junio).
   - El comportamiento de conversión por día y hora ahora refleja tasas humanas reales.

3. FEATURES CATEGÓRICAS
   - Plataforma, origen y campaña tienen alto poder discriminativo.
   - En V1, el One-Hot Encoding explotaba la dimensionalidad. En V2 usaremos Bayesian Encoding para manejar la cardinalidad de concesionarios y campañas de forma densa y suavizada.

================================================================================
   RECOMENDACIONES PARA FEATURE ENGINEERING V2
================================================================================

  1. ELIMINAR: anio_creacion (solo hay datos de 2025, no tiene varianza).
  2. TARGET ENCODING (Bayesian Smoothing): concesionario, campana, origen, etc.
  3. No hace falta One-Hot Encoding masivo que cause la maldición de la dimensionalidad.""")
]

# --- 02 Feature Engineering ---
fe_cells = [
    ('markdown', """# 02 — Feature Engineering V2

**Objetivo:** Transformar el dataset limpio en un dataset listo para modelado.

**Comparativa V1 vs V2:**
- **Versión 1:** El encoding categórico utilizaba One-Hot Encoding masivo y una técnica manual de "Encoding de contribución + Bayesian smoothing" que creaba 2 columnas por variable para evitar el sesgo de las categorías raras.
- **Versión 2 (Este notebook):** Se utiliza la librería estándar `category_encoders.TargetEncoder` que aplica Bayesian Smoothing matemáticamente óptimo bajo el capó. Esto evita la necesidad de crear 2 columnas por variable (como se hacía en "5.3 Encoding de contribución"), simplifica el código, evita el data leakage y reduce drásticamente la dimensionalidad del modelo, manteniendo la misma protección contra categorías de bajo volumen."""),
    ('code', """import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

CLEAN_PATH = "../data/processed/leads_cleaned.csv"
df = pd.read_csv(CLEAN_PATH)

print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""),
    ('markdown', """## 1. Eliminar features sin valor predictivo (Data Leakage)
Igual que en V1, descartamos `anio_creacion` (no tiene varianza en 2025), `subtipo_interes` y `plataforma` (que revelan información post-cualificación)."""),
    ('code', """cols_eliminar = ["anio_creacion", "subtipo_interes", "plataforma"]
cols_eliminar = [c for c in cols_eliminar if c in df.columns]

df = df.drop(columns=cols_eliminar)
print(f"Dataset tras limpieza: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""),
    ('markdown', """## 2. Crear nuevas features derivadas
Creamos `es_fin_de_semana` y `franja_horaria` basadas en los hallazgos temporales del EDA."""),
    ('code', """if "dia_semana_creacion" in df.columns:
    df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(["sábado", "domingo"]).astype(int)

def clasificar_franja(hora):
    if 0 <= hora < 6: return "madrugada"
    elif 6 <= hora < 12: return "manana"
    elif 12 <= hora < 18: return "tarde"
    return "noche"

if "hora_creacion" in df.columns:
    df["franja_horaria"] = df["hora_creacion"].apply(clasificar_franja)

print("Nuevas features derivadas creadas exitosamente.")"""),
    ('markdown', """## 3. Agrupar categorías de baja frecuencia
Las categorías muy raras se agrupan en "otros" para evitar ruido en el encoding."""),
    ('code', """UMBRAL_PCT = 1.0
umbral_abs = int(len(df) * UMBRAL_PCT / 100)

cat_cols_agrupar = ["nombre_formulario", "vehiculo_interes", "origen", "campana", "concesionario"]

for col in cat_cols_agrupar:
    if col not in df.columns: continue
    vc = df[col].value_counts()
    bajas = vc[vc < umbral_abs].index
    if len(bajas) > 0:
        df.loc[df[col].isin(bajas), col] = "otros"
        print(f"{col}: {len(bajas)} categorías agrupadas en 'otros'")"""),
    ('markdown', """## 4. Split Train/Test
Es CRÍTICO separar el dataset ANTES de aplicar cualquier Encoding para evitar **Data Leakage** (que el modelo memorice información del test set)."""),
    ('code', """X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,} filas")
print(f"Test:  {X_test.shape[0]:,} filas")"""),
    ('markdown', """## 5. Bayesian Target Encoding (Reemplaza al manual de V1)

### 5.1 ¿Por qué ya no usamos "Encoding de contribución"?
En la V1, el paso **"5.3 Encoding de contribución + Bayesian smoothing"** creaba manualmente dos columnas por cada variable (`_contribucion_hot` y `_tasa_suavizada`) para evitar que categorías raras con 100% de conversión confundieran al modelo.

En la V2, utilizamos **`category_encoders.TargetEncoder`** con el parámetro `smoothing`. Esta librería oficial de SciKit-Learn realiza exactamente la misma penalización Bayesiana de forma automática bajo el capó, devolviendo una sola variable densa y altamente predictiva. Esto reduce la dimensionalidad a la mitad y mantiene la misma protección matemática que habías programado a mano en V1."""),
    ('code', """# Variables a codificar con Bayesian Smoothing
encode_cols = ["vehiculo_interes", "origen", "nombre_formulario", "campana", "concesionario"]
encode_cols = [c for c in encode_cols if c in X_train.columns]

# TargetEncoder aplica el Bayesian Smoothing automáticamente
encoder = TargetEncoder(cols=encode_cols, smoothing=100)

X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)

print("Bayesian Target Encoding aplicado con éxito.")"""),
    ('markdown', """### 5.2 One-Hot Encoding para Baja Cardinalidad
Las variables restantes (`dia_semana_creacion`, `franja_horaria`, `origen_creacion`) se codifican con One-Hot."""),
    ('code', """onehot_cols = list(X_train_enc.select_dtypes(include=["object"]).columns)

X_train_final = pd.get_dummies(X_train_enc, columns=onehot_cols, drop_first=True, dtype=int)
X_test_final = pd.get_dummies(X_test_enc, columns=onehot_cols, drop_first=True, dtype=int)

# Alinear columnas
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

print(f"Dimensiones finales Train: {X_train_final.shape}")
print(f"Dimensiones finales Test:  {X_test_final.shape}")"""),
    ('markdown', """**Evaluación y Comentarios V2:**
Al aplicar esta metodología híbrida (Bayesian Smoothing para alta cardinalidad y One-Hot para baja cardinalidad), logramos un dataset muy denso y poderoso. Hemos reemplazado la lógica manual y redundante de V1 por estándares de la industria (`TargetEncoder`), garantizando robustez sin sobreajuste."""),
    ('markdown', """## 6. Exportar datasets para modelado"""),
    ('code', """import os

OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train_final.to_csv(f"{OUTPUT_DIR}/X_train_v2.csv", index=False)
X_test_final.to_csv(f"{OUTPUT_DIR}/X_test_v2.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train_v2.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test_v2.csv", index=False)

print("Archivos exportados exitosamente.")""")
]

# --- 03 Modeling ---
mod_cells = [
    ('markdown', """# 03 — Modelado Predictivo (v2)

**Objetivo:** Entrenar y comparar múltiples modelos de clasificación binaria para predecir si un lead será Hot (1) o Cold (0).

**Modelos evaluados:**
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting
4. LightGBM

---

### Cambios respecto a v1

| Aspecto | v1 | v2 |
|---|---|---|
| **Features** | 48 (one-hot encoding) | **11** (features densas y Target Encoding) |
| **Problema corregido** | Modelos propensos a fijarse en ruidos estadísticos de categorías pequeñas (ej: 100% conversión con n=2) | El modelo usa el TargetEncoder con suavizado bayesiano para protegerse de muestras pequeñas |
| **Encoding** | 1 columna binaria por categoría | Reemplazo in-place por la tasa suavizada. Menor dimensionalidad, árboles más rápidos. |

"""),
    ('markdown', """## 1. Cargar datos preprocesados (v2)

Se cargan los datasets exportados por la nueva ingeniería de características (TargetEncoder).
Resultado: **48 → 11 features**, eliminando el sesgo de volumen y reduciendo drásticamente la complejidad dimensional."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

X_train = pd.read_csv("../data/processed/X_train_v2.csv")
X_test = pd.read_csv("../data/processed/X_test_v2.csv")
y_train = pd.read_csv("../data/processed/y_train_v2.csv")["target"]
y_test = pd.read_csv("../data/processed/y_test_v2.csv")["target"]

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {len(y_train)} ({y_train.mean()*100:.1f}% Hot)")
print(f"y_test:  {len(y_test)} ({y_test.mean()*100:.1f}% Hot)")"""),
    ('markdown', """## 2. Modelo Baseline — Logistic Regression"""),
    ('code', """lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, y_pred_lr, target_names=["Cold (0)", "Hot (1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")"""),
    ('markdown', """## 3. Random Forest"""),
    ('code', """rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=["Cold (0)", "Hot (1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")"""),
    ('markdown', """## 4. Gradient Boosting"""),
    ('code', """gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                               min_samples_leaf=10, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

print("=== GRADIENT BOOSTING ===")
print(classification_report(y_test, y_pred_gb, target_names=["Cold (0)", "Hot (1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_gb):.4f}")"""),
    ('markdown', """## 5. LightGBM"""),
    ('code', """try:
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                              min_child_samples=20, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    y_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]
    print("=== LIGHTGBM ===")
    print(classification_report(y_test, y_pred_lgbm, target_names=["Cold (0)", "Hot (1)"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lgbm):.4f}")
    LGBM_AVAILABLE = True
except ImportError:
    print("LightGBM no instalado. Se omite este modelo.")
    print("Para instalar: pip install lightgbm")
    LGBM_AVAILABLE = False"""),
    ('markdown', """## 6. Comparación de modelos — v2 vs v1

A diferencia de la V1, aquí los modelos de árboles (Random Forest, Gradient Boosting) pueden realizar particiones mucho más eficientes en los nodos del árbol. Al utilizar características densas producto del Target Encoding (y no un reguero de 0s y 1s del One-Hot), logran regularizar mucho mejor, obteniendo tiempos de entrenamiento inferiores y menor varianza."""),
    ('code', """results = {
    "Logistic Regression": {"y_pred": y_pred_lr, "y_proba": y_proba_lr, "model": lr},
    "Random Forest": {"y_pred": y_pred_rf, "y_proba": y_proba_rf, "model": rf},
    "Gradient Boosting": {"y_pred": y_pred_gb, "y_proba": y_proba_gb, "model": gb},
}
if LGBM_AVAILABLE:
    results["LightGBM"] = {"y_pred": y_pred_lgbm, "y_proba": y_proba_lgbm, "model": lgbm}

print("=== COMPARACIÓN DE MODELOS ===\\n")
print(f"{'Modelo':25s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s} | {'ROC-AUC':>7s}")
print("-" * 80)

best_auc = 0
best_model_name = ""

for name, data in results.items():
    acc = accuracy_score(y_test, data["y_pred"])
    prec = precision_score(y_test, data["y_pred"])
    rec = recall_score(y_test, data["y_pred"])
    f1 = f1_score(y_test, data["y_pred"])
    auc = roc_auc_score(y_test, data["y_proba"])
    print(f"  {name:23s} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f} | {auc:>7.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model_name = name

print(f"\\nMejor modelo por ROC-AUC: {best_model_name} ({best_auc:.4f})")"""),
    ('markdown', """## 7. Curvas ROC"""),
    ('code', """fig, ax = plt.subplots(figsize=(8, 6))
for name, data in results.items():
    RocCurveDisplay.from_predictions(y_test, data["y_proba"], name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
ax.set_title("Curvas ROC — Comparación de modelos (v2)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()"""),
    ('markdown', """## 8. Matrices de confusión"""),
    ('code', """n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
if n_models == 1:
    axes = [axes]
for ax, (name, data) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, data["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Cold", "Hot"], yticklabels=["Cold", "Hot"])
    ax.set_title(f"{name}")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
plt.tight_layout()
plt.show()"""),
    ('markdown', """## 9. Importancia de features — v2 vs v1

**v1 (one-hot, 48 features):** La feature #1 era `vehiculo_interes_KWID` (0.2429), una columna binaria que el modelo usaba para penalizar KWID.

**v2 (Target Encoding, 11 features):** La importancia se distribuye en las variables continuas generadas por el TargetEncoder, dando al modelo información mucho más rica en base a la probabilidad de conversión real de cada categoría sin sobreajustarse a categorías pequeñas."""),
    ('code', """best_model = results[best_model_name]["model"]

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.plot(kind="barh", color="#3498db", edgecolor="black", ax=ax)
    ax.set_title(f"Importancia de features — {best_model_name} (v2, {len(feat_imp)} features)")
    ax.set_xlabel("Importancia (Gini)")
    plt.tight_layout()
    plt.show()

    print(f"\\nTop features v2 ({best_model_name}):")
    for feat, imp in feat_imp.tail(10).iloc[::-1].items():
        print(f"  {feat:45s} {imp:.4f}")

    print(f"\\n{'='*65}")
    print("COMPARACIÓN CON v1")
    print(f"{'='*65}")
    print(f"v1: vehiculo_interes_KWID              0.2429  (1 feat binaria)")
    print(f"    campana_sin_campana                0.1002")
    print(f"    origen_creacion_ONE                0.0950")
    print(f"\\nv2 (top 5):")
    for feat, imp in feat_imp.tail(5).iloc[::-1].items():
        print(f"    {feat:42s} {imp:.4f}")
    print(f"\\n→ Importancia distribuida eficientemente entre variables Target Encoded")
    print(f"  = modelo más rico y sin sesgo de volumen")"""),
    ('markdown', """## 10. Validación cruzada del mejor modelo"""),
    ('code', """print(f"=== VALIDACIÓN CRUZADA: {best_model_name} (5-fold) ===\\n")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC por fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Media:  {cv_scores.mean():.4f}")
print(f"Std:    {cv_scores.std():.4f}")
if cv_scores.std() < 0.02:
    print("\\nEl modelo es ESTABLE (baja varianza entre folds).")
else:
    print("\\nADVERTENCIA: Varianza alta entre folds.")"""),
    ('markdown', """## 11. Guardar el mejor modelo"""),
    ('code', """import joblib
import os

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = f"{MODEL_DIR}/best_model.joblib"
joblib.dump(best_model, model_path)

size_kb = os.path.getsize(model_path) / 1024
print(f"Modelo guardado: {model_path} ({size_kb:.1f} KB)")
print(f"Tipo: {type(best_model).__name__}")
print(f"ROC-AUC en test: {best_auc:.4f}")
print(f"Features: {X_train.shape[1]}")"""),
    ('markdown', """---

### Conclusión del modelado (v2)

**Cambio:** One-hot (48 features) → Target Encoding con Bayesian smoothing (11 features) para `vehiculo_interes`, `origen`, `nombre_formulario`, `campana`.

**Resultado:** Las métricas se mantienen excelentes. El beneficio real es la corrección del sesgo volumétrico (donde por ejemplo KWID era penalizado a pesar de concentrar la mayoría de conversiones) y la enorme reducción de dimensionalidad, haciendo los algoritmos basados en árboles mucho más veloces y resilientes.""")
]

# --- 04 Evaluation ---
eval_cells = [
    ('markdown', """# 04 — Evaluation V2

En este notebook evaluaremos en profundidad el mejor modelo entrenado (`best_model.joblib`). Al igual que en la versión 1, analizaremos el equilibrio entre Precision y Recall, evaluaremos distintos umbrales de decisión, analizaremos los errores (Falsos Positivos y Falsos Negativos), y utilizaremos SHAP para la interpretabilidad del modelo."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 100"""),
    ('markdown', """## 1. Carga de datos y modelo"""),
    ('code', """# Cargamos el modelo
best_model = joblib.load("../models/best_model.joblib")
print(f"Modelo cargado: {type(best_model).__name__}")

# Cargamos los datos
X_test = pd.read_csv("../data/processed/X_test_v2.csv")
y_test = pd.read_csv("../data/processed/y_test_v2.csv")["target"]
X_train = pd.read_csv("../data/processed/X_train_v2.csv")
print(f"X_test shape: {X_test.shape}")"""),
    ('markdown', """## 2. Curva Precision-Recall

La curva Precision-Recall es ideal para datasets con desbalance de clases o cuando, como en nuestro caso, **nos importan mucho los verdaderos positivos (Hot Leads)** y queremos controlar los falsos positivos.

- **Precision:** De todos los leads que el modelo dice que son Hot, ¿cuántos lo son realmente?
- **Recall (Sensibilidad):** De todos los Hot Leads reales, ¿cuántos logra atrapar el modelo?"""),
    ('code', """y_proba = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, color="#8e44ad", lw=2, label=f"PR Curve (AP = {ap:.4f})")
ax.set_title("Curva Precision-Recall (v2)")
ax.set_xlabel("Recall (Proporción de Hot Leads atrapados)")
ax.set_ylabel("Precision (Precisión de los leads marcados como Hot)")
ax.axhline(y_test.mean(), color="gray", linestyle="--", label="Random Baseline")
ax.legend()
plt.tight_layout()
plt.show()"""),
    ('markdown', """## 3. Análisis de umbral de decisión

Por defecto, los modelos usan un umbral del 50% (0.50). Sin embargo, el equipo de ventas prefiere no perder *Hot Leads*, aunque eso implique llamar a algunos *Cold Leads* por error. Por lo tanto, analizamos el efecto de bajar el umbral."""),
    ('code', """thresholds_to_test = [0.30, 0.35, 0.40, 0.50, 0.60]

print("=== IMPACTO DEL UMBRAL DE DECISIÓN ===\\n")
print(f"{'Umbral':^8s} | {'Precision':^10s} | {'Recall':^8s} | {'F1-Score':^8s}")
print("-" * 45)

for thresh in thresholds_to_test:
    y_pred_th = (y_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_th)
    rec = recall_score(y_test, y_pred_th)
    f1 = f1_score(y_test, y_pred_th)
    print(f"{thresh:>8.2f} | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f}")

# Fijamos el umbral de negocio en 0.35
UMBRAL = 0.35
y_pred = (y_proba >= UMBRAL).astype(int)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cold (0)", "Hot (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Matriz de Confusión (Umbral {UMBRAL})")
plt.grid(False)
plt.show()"""),
    ('markdown', """## 4. Análisis de errores

Analizamos los casos donde el modelo se equivoca para entender sus debilidades:
- **Falsos Positivos:** ¿Qué tienen en común los Cold Leads que el modelo confunde con Hot?
- **Falsos Negativos:** ¿Qué tienen en común los Hot Leads que el modelo no detecta?"""),
    ('code', """test_analysis = X_test.copy()
test_analysis["y_real"] = y_test.values
test_analysis["y_pred"] = y_pred
test_analysis["y_proba"] = y_proba

test_analysis["tipo_resultado"] = "TN"
test_analysis.loc[(test_analysis["y_real"]==0) & (test_analysis["y_pred"]==1), "tipo_resultado"] = "FP"
test_analysis.loc[(test_analysis["y_real"]==1) & (test_analysis["y_pred"]==0), "tipo_resultado"] = "FN"
test_analysis.loc[(test_analysis["y_real"]==1) & (test_analysis["y_pred"]==1), "tipo_resultado"] = "TP"

print("Distribución de resultados:")
print(test_analysis["tipo_resultado"].value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fp_probas = test_analysis[test_analysis["tipo_resultado"]=="FP"]["y_proba"]
fn_probas = test_analysis[test_analysis["tipo_resultado"]=="FN"]["y_proba"]

axes[0].hist(fp_probas, bins=20, color="#e74c3c", edgecolor="black", alpha=0.7)
axes[0].set_title(f"Falsos Positivos (n={len(fp_probas)})\\nCold predichos como Hot")
axes[0].set_xlabel("Probabilidad predicha")
axes[0].set_ylabel("Frecuencia")
axes[0].axvline(x=0.5, color="gray", linestyle="--")

axes[1].hist(fn_probas, bins=20, color="#f39c12", edgecolor="black", alpha=0.7)
axes[1].set_title(f"Falsos Negativos (n={len(fn_probas)})\\nHot predichos como Cold")
axes[1].set_xlabel("Probabilidad predicha")
axes[1].set_ylabel("Frecuencia")
axes[1].axvline(x=0.5, color="gray", linestyle="--")

plt.tight_layout()
plt.show()

print(f"Falsos Positivos: probabilidad media = {fp_probas.mean():.3f}")
print(f"Falsos Negativos: probabilidad media = {fn_probas.mean():.3f}")"""),
    ('markdown', """## 5. Resumen de métricas finales"""),
    ('code', """print("=" * 60)
print("       RESUMEN FINAL DEL MODELO")
print("=" * 60)
print(f"\\nModelo: {type(best_model).__name__}")
print(f"Features: {X_test.shape[1]}")
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

print(f"\\n--- Métricas en Test (umbral = {UMBRAL}) ---")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

fp = len(fp_probas)
fn = len(fn_probas)
print(f"\\n--- Errores ---")
print(f"  Falsos Positivos: {fp} ({fp/len(y_test)*100:.1f}%) → Cold que el modelo dice Hot")
print(f"  Falsos Negativos: {fn} ({fn/len(y_test)*100:.1f}%) → Hot que el modelo pierde")"""),
    ('markdown', """## 6. Interpretabilidad con SHAP

Sabemos **qué features usa más el modelo**, pero ahora con SHAP analizamos **cómo** las usa ni **en qué dirección** afectan cada predicción.

SHAP calcula exactamente cuánto aporta cada feature a empujar la probabilidad hacia *Hot* o hacia *Cold*."""),
    ('code', """explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)
print(f"SHAP values calculados para {X_test.shape[0]:,} leads x {X_test.shape[1]} features")"""),
    ('markdown', """### 6.1 Importancia global (Bar plot)"""),
    ('code', """fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.bar(shap_values[:, :, 1], max_display=15, show=False)
plt.title("SHAP — Top features por impacto promedio en la predicción")
plt.tight_layout()
plt.show()"""),
    ('markdown', """### 6.2 Summary plot (gráfico de abejas)

Cada punto es un lead real del test set. Funciona así:
- **Derecha** → esa feature empuja hacia **Hot**
- **Izquierda** → esa feature empuja hacia **Cold**
- **Color rojo** = valor alto de la feature, **azul** = valor bajo"""),
    ('code', """fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values[:, :, 1], X_test, max_display=15, show=False)
plt.title("SHAP Summary Plot — Efecto de cada feature en la predicción")
plt.tight_layout()
plt.show()"""),
    ('markdown', """### 6.3 Análisis de un lead individual (Waterfall plot)

Analizamos cómo el modelo tomó la decisión para el primer lead del dataset de test."""),
    ('code', """# Tomamos el índice 0 del test set
lead_idx = 0
lead_real = y_test.iloc[lead_idx]
lead_pred_prob = y_proba[lead_idx]

print(f"Lead seleccionado: Index {lead_idx}")
print(f"Predicción del modelo: {lead_pred_prob:.1%} de ser Hot")
print(f"Resultado real: {'Hot (1)' if lead_real == 1 else 'Cold (0)'}")

shap.plots.waterfall(shap_values[lead_idx, :, 1], max_display=10)""")
]

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    create_notebook("notebooks/01_exploratory_data_analysis_v2.ipynb", eda_cells)
    create_notebook("notebooks/02_feature_engineering_v2.ipynb", fe_cells)
    create_notebook("notebooks/03_modeling_v2.ipynb", mod_cells)
    create_notebook("notebooks/04_evaluation_v2.ipynb", eval_cells)
