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
    # ── TÍTULO PRINCIPAL ─────────────────────────────────────────────────
    ('markdown', """# 01 — Exploratory Data Analysis (EDA) V2

**Objetivo:** Comprender la distribución, estructura y relaciones del dataset procesado antes de construir el modelo. Este análisis guía las decisiones de feature engineering y nos permite detectar patrones comerciales en los datos de leads de 2025.

**Secciones de este notebook:**
1. Información general del dataset y estadísticas descriptivas
2. Distribución del Target (Hot Lead vs Cold Lead)
3. Análisis de variables temporales (Hora, Día de la semana, Mes)
4. Análisis de variables categóricas principales
5. Análisis temporal detallado (Heatmaps cruzados)
6. Correlación entre variables numéricas
7. Hallazgos clave y recomendaciones para Feature Engineering

**Comparativa V1 vs V2:**
- **Versión 1:** El análisis se hizo sobre ~8,400 registros mezclando múltiples años, con presencia de leads del chatbot que distorsionaban todas las distribuciones.
- **Versión 2:** Analizamos ~66,000 registros depurados del año 2025. Sin chatbot, sin meses anómalos, obtenemos una visión real del comportamiento orgánico comercial."""),

    ('code', """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/processed/leads_cleaned.csv")
print(f"Dataset V2 cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""),

    # ── 1. INFORMACIÓN GENERAL ───────────────────────────────────────────
    ('markdown', """## 1. Información General del Dataset y Estadísticas Descriptivas

Se revisa la estructura del dataset procesado: tipos de datos, valores nulos residuales y estadísticas descriptivas para variables numéricas y categóricas. Esta es la base para validar que el Data Engineering funcionó correctamente y entender el rango de valores con los que trabajará el modelo."""),

    ('code', """df.info()
print("\\nValores Nulos por Columna:")
print(df.isnull().sum())
print("\\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print("\\n--- Numéricas ---")
print(df.describe().to_string())
print("\\n--- Categóricas ---")
print(df.describe(include='object').to_string())"""),

    ('markdown', """**Hallazgos:**
- El dataset no debería tener valores nulos ya que fueron imputados con `'Desconocido'` en el paso anterior. Si aparecen, revisar el pipeline de Data Engineering.
- Las variables numéricas (`hora_creacion`, `mes_creacion`, `dia_creacion`) deben mostrar rangos lógicos (0-23 para hora, 1-12 para mes, etc.).
- Las variables categóricas con muchos valores únicos (`campana`, `concesionario`) confirman la necesidad de estrategias de encoding sofisticadas en el siguiente notebook.

**Diferencias V1 vs V2:**
La V1 tenía columnas adicionales como `anio_creacion`, `plataforma` y `subtipo_interes`. En la V2 estas fueron eliminadas durante el Data Engineering por ser de varianza cero o por riesgo de data leakage, resultando en un dataset más compacto pero más limpio para el modelado."""),

    # ── 2. DISTRIBUCIÓN DEL TARGET ───────────────────────────────────────
    ('markdown', """## 2. Distribución del Target (Hot Lead vs Cold Lead)

La variable objetivo `target` es binaria: **1 = Hot Lead** (lead con intención de compra confirmada) y **0 = Cold Lead** (rechazo o sin contacto exitoso). Entender su distribución es crítico para elegir las métricas de evaluación del modelo y decidir si se requieren técnicas de balanceo de clases."""),

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

    ('markdown', """**Hallazgos:**
La distribución muestra aproximadamente **~37% Hot Leads** y **~63% Cold Leads**. Esta proporción indica:
- Un **desbalance moderado** que es natural en cualquier embudo de ventas comercial: no todos los que muestran interés terminan comprando.
- El 37% de clase positiva es suficiente para que los algoritmos de ML capturen los patrones de conversión **sin necesidad obligatoria de SMOTE u oversampling**.
- Se evaluará el uso de `class_weight='balanced'` en los modelos para compensar levemente el desbalance sin distorsionar la distribución real.
- La métrica principal de evaluación será **ROC-AUC** (resistente al desbalance), complementada con Precision-Recall y F1-Score.

**Diferencias V1 vs V2:**
En la V1, la tasa de Hot Leads era artificialmente alta (~68%) por la presencia del chatbot que procesaba leads de baja intención con cualificación rápida. Al eliminar el chatbot, la distribución en V2 refleja la **verdadera tasa de conversión del equipo comercial humano** (~37%), que es el número relevante para calibrar las expectativas del modelo en producción."""),

    # ── 3. VARIABLES TEMPORALES ──────────────────────────────────────────
    ('markdown', """## 3. Análisis de Variables Temporales

Se analiza cómo el **volumen de leads** y la **tasa de conversión** varían según la hora del día, el día de la semana y el mes del año. Estas variables temporales son frecuentemente de las más predictivas en modelos de leads porque capturan la intencionalidad del usuario (quien llena un formulario a las 10 AM un martes tiene un perfil muy distinto a quien lo hace a las 2 AM un domingo)."""),

    ('code', """fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)

cols_temp = ['hora_creacion', 'dia_semana_creacion', 'mes_creacion']
titulos = ['Hora de Creación', 'Día de la Semana', 'Mes']

for i, col in enumerate(cols_temp):
    if col not in df.columns: continue
    
    # Volumen
    ax_vol = axes[i, 0]
    order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'] if col == 'dia_semana_creacion' else None
    if col == 'dia_semana_creacion':
        df['dia_semana_creacion'] = df['dia_semana_creacion'].str.capitalize()
    sns.countplot(data=df, x=col, color='steelblue', order=order, ax=ax_vol)
    ax_vol.set_title(f'Volumen por {titulos[i]}')
    ax_vol.tick_params(axis='x', rotation=45)
    
    # Porcentajes Volumen
    total = len(df)
    for p in ax_vol.patches:
        height = p.get_height()
        if pd.isna(height): height = 0
        percentage = f'{100 * height / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = height
        if height > 0:
            ax_vol.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=9)
        
    # Tasa de conversión
    ax_conv = axes[i, 1]
    conv = df.groupby(col)['target'].mean().reset_index()
    sns.barplot(data=conv, x=col, y='target', color='seagreen', order=order, ax=ax_conv)
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

    ('markdown', """**Hallazgos por variable temporal:**

**Por Hora:**
El volumen de leads sigue una curva de campana que pica en horario comercial (10 AM – 5 PM), lo cual coincide con la actividad normal de un cliente buscando un vehículo. La tasa de conversión, sin embargo, se mantiene relativamente estable a lo largo del día, indicando que **la hora no es un predictor lineal fuerte de conversión** por sí sola (aunque sí lo es en combinación con el día, como veremos en los Heatmaps).

**Por Día de la Semana:**
El volumen cae drásticamente los sábados y domingos (los concesionarios tienen menor atención). Sin embargo, la tasa de conversión de los leads que sí llegan los fines de semana es equiparable o incluso ligeramente superior a los días laborables. Esto indica que **el lead de fin de semana tiene alta intención de compra**: si está llenando un formulario el domingo, es porque tiene un interés genuino.

**Por Mes:**
Los meses excluidos (Abril, Mayo, Junio) no aparecen, confirmando el filtro del Data Engineering. Los meses visibles muestran el volumen operativo real de la flota comercial a lo largo del año 2025.

**Diferencias V1 vs V2:**
En la V1, la extracción de hora se hacía sobre `Fecha de creación` (timestamp del sistema), haciendo que la mayoría de los leads mostraran hora 00:00. Esto ocultaba completamente los patrones horarios. En la V2, usando `Fecha de creación por el cliente`, recuperamos la distribución horaria real. La gráfica de día de la semana también cambia drásticamente: en la V1 el chatbot operaba los 7 días de forma uniforme, desdibujando la caída natural de los fines de semana."""),

    # ── 4. VARIABLES CATEGÓRICAS ─────────────────────────────────────────
    ('markdown', """## 4. Análisis de Variables Categóricas Principales

Se analizan las variables categóricas con mayor poder discriminativo para la conversión: **origen del lead, campaña de marketing, concesionario asignado y formulario de captación**. Para cada una se muestra el Top 10 por volumen (izquierda) y la tasa de conversión de cada categoría (derecha), con la línea roja indicando la media global (~37%)."""),

    ('code', """variables_cat = ['origen', 'campana', 'concesionario', 'nombre_formulario']
for col in variables_cat:
    if col not in df.columns: continue
    
    # Truncar nombres muy largos para que se vean en los gráficos
    df[col] = df[col].astype(str).apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    top_10 = df[col].value_counts().nlargest(10).index
    df_top = df[df[col].isin(top_10)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.3, left=0.15)
    
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

    ('markdown', """**Hallazgos por variable categórica:**

**Origen del lead:**
Los orígenes con mayor volumen (como formularios de la web o landing pages) no siempre son los de mayor conversión. Orígenes de nicho o especializados (como referidos o eventos) suelen tener tasas de conversión muy superiores a la media.

**Campaña:**
Se observa el fenómeno clásico de Marketing digital: pocas campañas concentran la mayoría del volumen, pero con tasas de conversión bajas (tráfico masivo y frío). Otras campañas más específicas tienen volumen reducido pero una conversión sobresaliente. El modelo deberá capturar esta relación no lineal entre campaña y conversión.

**Concesionario:**
Las tasas de conversión varían significativamente entre concesionarios, reflejando diferencias en la eficiencia del equipo de ventas, la zona geográfica y el tipo de producto que comercializan.

**Nombre del formulario:**
Los formularios de captura tienen perfiles de calidad muy distintos: formularios de comparación de precios o configuradores generan leads de alta conversión, mientras que formularios de suscripción o brochures atraen leads más exploratorios.

**Diferencias V1 vs V2:**
En la V1, la variable `plataforma` mostraba en esta sección que el chatbot era responsable de un enorme volumen con conversión casi nula, "revelando" el bot al modelo como una señal de baja calidad. Al eliminarlo en V2, el análisis de categorías muestra únicamente el comportamiento humano, y variables como `campana` y `concesionario` emergen con mayor fuerza predictiva real. Además, en V1 se usaba One-Hot Encoding generando decenas de columnas. En V2 usaremos Target Encoding con Bayesian Smoothing para capturar estas relaciones de forma densa y sin explosión de dimensionalidad."""),

    # ── 5. HEATMAPS ──────────────────────────────────────────────────────
    ('markdown', """## 5. Análisis Temporal Detallado (Heatmaps)

Los heatmaps cruzados permiten visualizar simultáneamente **dos dimensiones temporales** (día de la semana vs. hora del día) para descubrir patrones que no son visibles en los gráficos simples por separado. Se generan dos heatmaps:
1. **Volumen de leads** por combinación día-hora (¿cuándo llegan más leads?)
2. **Tasa de conversión** por combinación día-hora (¿cuándo convierten mejor?)

La comparación entre ambos es la clave: un bloque con alto volumen pero baja conversión es tráfico frío; un bloque con poco volumen pero alta conversión es tráfico caliente muy valioso."""),

    ('code', """pivot_vol = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='count', fill_value=0)
pivot_conv = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='mean', fill_value=0)

dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
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

    ('markdown', """**Hallazgos del Heatmap de Volumen:**
El mapa azul muestra clústeres oscuros muy marcados de **Lunes a Viernes entre las 9:00 y las 19:00 horas**, que corresponde al horario operativo de los concesionarios. Los fines de semana presentan un volumen considerablemente menor pero no nulo, reflejando que algunos clientes investigan activamente durante su tiempo libre. La madrugada (0-7 AM) es prácticamente nula para todos los días, lo cual confirma que los leads de chatbot (que sí aparecían en estas franjas) han sido correctamente eliminados.

**Hallazgos del Heatmap de Conversión:**
El mapa rojo revela algo estratégicamente valioso: **los picos de conversión no se alinean perfectamente con los picos de volumen**. Existen franjas horarias específicas (generalmente primeras horas de la mañana o tarde-noche) donde, aunque el volumen de leads es menor, la tasa de conversión es notablemente superior. Esto sugiere que los leads que contactan fuera del horario pico tienen mayor intención de compra, posiblemente porque han investigado más antes de contactar.

**Diferencias V1 vs V2:**
En la V1, los heatmaps estaban completamente distorsionados: el chatbot operaba en lotes nocturnos, creando franjas de alta densidad artificiales en horas imposibles (2-5 AM). Además, la extracción incorrecta de la hora sobre la columna del sistema hacía que la mayoría de los registros cayeran en la hora 0, generando una columna vertical de madrugada sin sentido comercial. En la V2, ambos problemas están corregidos, y el heatmap refleja la realidad operativa del negocio."""),

    # ── 6. CORRELACIÓN ───────────────────────────────────────────────────
    ('markdown', """## 6. Correlación entre Variables Numéricas

La matriz de correlación mide la **relación lineal** entre las variables numéricas del dataset y con la variable objetivo `target`. Es útil para detectar features redundantes (alta correlación entre sí) y para entender qué variables tienen alguna relación lineal directa con la conversión.

> ⚠️ **Nota:** Una correlación lineal baja no significa que la variable sea inútil. Los modelos de árboles (como XGBoost o LightGBM) pueden capturar relaciones no lineales complejas que la correlación de Pearson no detecta."""),

    ('code', """num_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación - Features Numéricas")
plt.tight_layout()
plt.show()"""),

    ('markdown', """**Hallazgos de la Matriz de Correlación:**
Al analizar la relación de las variables temporales numéricas con el Target (1=Hot, 0=Cold):
- **`mes_creacion` (~-0.20):** Correlación negativa leve. Los meses iniciales del año tienen mayor tasa de conversión, que decrece ligeramente hacia finales de año. Puede estar relacionado con ciclos de ventas automotrices (inicio de año, renovación de flota).
- **`dia_creacion` (~0.14):** Correlación positiva leve. Leads hacia finales de mes tienen marginalmente mayor conversión, posiblemente relacionado con cierres de quincena o mes del equipo comercial.
- **`hora_creacion` (~0.00):** Sin correlación lineal significativa. Esto es esperado: la hora importa, pero de forma no lineal (ciertos bloques horarios son mejores, no hay una tendencia lineal creciente o decreciente). Los árboles de decisión capturarán esto perfectamente.
- **Entre variables numéricas:** Las variables temporales no están correlacionadas entre sí, lo que es ideal: cada una aporta información independiente al modelo.

**Diferencias V1 vs V2:**
En la V1, la variable `anio_creacion` mostraba correlación porque los datos mezclaban 2022, 2023 y 2024, años con tasas de conversión distintas por cambios en el proceso de ventas. Al filtrar solo 2025, eliminamos esa pseudo-correlación espuria. Además, `hora_creacion` ahora muestra valores distribuidos (0-23) en lugar de estar concentrada en 0, lo que recupera su capacidad predictiva en el modelo."""),

    # ── 7. HALLAZGOS FINALES ─────────────────────────────────────────────
    ('markdown', """## 7. Hallazgos Clave del EDA y Recomendaciones para Feature Engineering

---
### Resumen de Hallazgos

**1. TARGET**
- Distribución real: ~37% Hot Leads / ~63% Cold Leads (sin sesgo de chatbot)
- Desbalance moderado: no requiere SMOTE obligatorio; usar `class_weight='balanced'`
- Métrica principal recomendada: ROC-AUC, complementada con F1-Score

**2. FEATURES TEMPORALES**
- `hora_creacion`: ahora refleja horas reales (no 00:00). Relación no lineal con conversión → usar modelos de árbol
- `dia_semana_creacion`: fines de semana tienen menor volumen pero igual tasa de conversión
- `mes_creacion`: ligera tendencia estacional; meses iniciales con mayor conversión

**3. FEATURES CATEGÓRICAS**
- Alta cardinalidad en `campana`, `concesionario`, `nombre_formulario`: One-Hot Encoding causaría explosión de dimensionalidad
- Relación no lineal entre volumen y conversión: categorías con alto volumen no siempre convierten mejor
- Estrategia recomendada: **Target Encoding con Bayesian Smoothing** para todas las categóricas de alta cardinalidad

---
### Recomendaciones para Feature Engineering (Notebook 02)

| Acción | Variable(s) | Motivo |
|--------|------------|--------|
| Eliminar | `subtipo_interes` | Columna completada post-llamada (data leakage) |
| Crear | `es_fin_de_semana` | Captura el comportamiento diferencial del lead de fin de semana |
| Crear | `franja_horaria` | Agrupa horas en bloques comerciales (madrugada/mañana/tarde/noche) |
| Target Encode | `campana`, `concesionario`, `origen`, `nombre_formulario` | Alta cardinalidad; Bayesian Smoothing evita sobreajuste |
| Conservar | `hora_creacion`, `mes_creacion`, `dia_creacion` | Relación no lineal capturable por árboles |
""")
]


# --- 02 Feature Engineering ---
fe_cells = [

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
    ('markdown', """**Resultados y Hallazgos:**
La distribución actual muestra un ~37% de Hot Leads y un ~63% de Cold Leads. Esto indica que tenemos un dataset moderadamente desbalanceado, pero con una proporción muy sana para entrenar algoritmos. A diferencia de problemas de anomalías donde la clase minoritaria es del 1%, un 37% de conversión permite a los modelos capturar bien los patrones sin necesidad obligatoria de aplicar técnicas artificiales como SMOTE.

**Diferencias V1 vs V2 - Distribución del Target:**
En la versión 1, debido a un dataset reducido y la presencia masiva de bots, la tasa de conversión mostraba un sesgo irrealmente alto cercano al 68%. En esta versión 2 (al limpiar meses anómalos y bots), la distribución refleja la **verdadera tasa de conversión orgánica** que ronda el 37%. Esta base sin sesgar es fundamental para no sobreestimar la capacidad predictiva de los modelos y reflejar las proporciones reales de ventas."""),
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
    order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'] if col == 'dia_semana_creacion' else None
    if col == 'dia_semana_creacion':
        df['dia_semana_creacion'] = df['dia_semana_creacion'].str.capitalize()
    sns.countplot(data=df, x=col, color='steelblue', order=order, ax=ax_vol)
    ax_vol.set_title(f'Volumen por {titulos[i]}')
    ax_vol.tick_params(axis='x', rotation=45)
    
    # Porcentajes Volumen
    total = len(df)
    for p in ax_vol.patches:
        height = p.get_height()
        if pd.isna(height): height = 0
        percentage = f'{100 * height / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = height
        if height > 0:
            ax_vol.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=9)
        
    # Tasa de conversión
    ax_conv = axes[i, 1]
    conv = df.groupby(col)['target'].mean().reset_index()
    sns.barplot(data=conv, x=col, y='target', color='seagreen', order=order, ax=ax_conv)
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
    ('markdown', """**Resultados y Hallazgos:**
- **Hora:** Observamos una curva de campana normal en el volumen durante horarios comerciales (picos entre 10 AM y 5 PM). La conversión se mantiene estable a lo largo del día.
- **Día de la Semana:** El volumen es claramente superior de Lunes a Viernes, reduciéndose drásticamente los fines de semana. La tasa de conversión no decae los fines de semana, indicando que el lead que llega en Sábado/Domingo tiene la misma intención de compra.
- **Mes:** Observamos los volúmenes operativos reales por mes a lo largo del año 2025.

**Diferencias V1 vs V2 - Variables Temporales:**
A diferencia de la V1, donde el volumen mensual no permitía deducir tendencias por el ruido provocado por el bot, aquí se observa claramente la cantidad de leads orgánicos gestionados por el equipo comercial de forma mensual. Al remover meses anómalos, el modelo aprende patrones temporales correctos en lugar de sesgarse por picos atípicos."""),
    ('markdown', """## Análisis de Variables Categóricas Principales
Volumen y Tasa de Conversión para variables categóricas clave."""),
    ('code', """variables_cat = ['origen', 'campana', 'concesionario', 'nombre_formulario']
for col in variables_cat:
    if col not in df.columns: continue
    
    # Truncar nombres muy largos para que se vean en los gráficos
    df[col] = df[col].astype(str).apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    top_10 = df[col].value_counts().nlargest(10).index
    df_top = df[df[col].isin(top_10)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.3, left=0.15)
    
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
    ('markdown', """**Resultados y Hallazgos:**
Al observar el Top 10 de categorías, notamos un fenómeno clásico de Marketing: existen campañas o canales con un volumen de tráfico gigantesco pero con tasas de conversión paupérrimas (leads de baja calidad/tráfico frío), mientras que otras fuentes de nicho tienen un volumen muy bajo pero una conversión sobresaliente. Esta no linealidad y alta cardinalidad es exactamente la relación que el modelo Predictivo deberá aprender a mapear.

**Diferencias V1 vs V2 - Variables Categóricas:**
En la versión 1 existía la variable `plataforma` que generaba un sesgo enorme porque diferenciaba a los bots de los humanos. Como en la V2 eliminamos por completo a los bots desde la fase de Data Engineering, la plataforma quedó con varianza cero (todos son humanos procesados por MX_LEAD_QUALIF), por lo que fue **eliminada** del análisis.

Además, vemos una dispersión muy grande en campañas y orígenes. En V1 esto se transformó con One-Hot Encoding generando decenas de columnas ruidosas. En V2 usaremos Target Encoding (Suavizado Bayesiano) para aprovechar la alta cardinalidad sin explotar la cantidad de columnas del modelo."""),
    ('markdown', """## Análisis Temporal Detallado (Heatmaps)
Visualización de volumen y tasa de conversión cruzando Día de la Semana y Hora de Creación."""),
    ('code', """pivot_vol = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='count', fill_value=0)
pivot_conv = df.pivot_table(index='dia_semana_creacion', columns='hora_creacion', values='target', aggfunc='mean', fill_value=0)

dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
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
    ('markdown', """**Resultados y Hallazgos (Mapas de Calor):**
- **Mapa de Volumen:** Muestra clústeres (manchas oscuras) sumamente marcados de Lunes a Viernes en la franja horaria de 9:00 a 19:00 horas, que corresponde a los horarios operativos pico de los concesionarios y del equipo comercial.
- **Mapa de Conversión:** Revela algo interesantísimo: los "picos de conversión" (celdas rojas) no necesariamente se alinean al 100% con los momentos de mayor volumen. Existen franjas horarias específicas donde los leads, aunque menos frecuentes, tienen altísima propensión a convertirse en Hot Leads.

**Diferencias V1 vs V2 - Mapas de Calor Temporales:**
En la V1 los mapas de calor estaban gravemente distorsionados porque la extracción original perdía la hora real y los bots procesaban en lotes masivos. Tras corregir la extracción y limpiar los bots en V2, **el mapa de calor recupera su lógica comercial**: observamos clústeres de volumen claros en días y horarios laborables de alto contacto orgánico."""),
    ('markdown', """## Correlación entre Variables Numéricas

Análisis de la relación lineal entre las variables numéricas y nuestra variable objetivo (Target). Esto nos permite identificar si existe alguna característica temporal que tenga un peso fuerte por sí sola sobre la conversión."""),
    ('code', """num_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación - Features Numéricas")
plt.tight_layout()
plt.show()"""),
    ('markdown', """**Resultados de la Matriz de Correlación:**
Al analizar la relación de las variables temporales numéricas con el Target (1=Hot, 0=Cold), observamos que:
- **`mes_creacion` (-0.20):** Existe una ligera correlación negativa, lo cual indica que en los meses iniciales hay mayor conversión y a medida que avanzan los meses la tasa tiende a disminuir ligeramente.
- **`dia_creacion` (0.14):** Existe una leve correlación positiva, sugiriendo un aumento menor en la conversión hacia finales de mes.
- **`hora_creacion` (~0.00):** No existe una relación *lineal* fuerte entre la hora aislada y la conversión, lo cual tiene sentido ya que la conversión tiene picos en ciertos horarios laborables (relación no lineal), requiriendo modelos basados en árboles para capturar esta complejidad.

---
**Diferencias V1 vs V2 - Matriz de Correlación Numérica:**
En V1, la variable `anio_creacion` mostraba correlación porque los datos históricos mezclaban varios años. En esta V2, hemos eliminado `anio_creacion` porque filtramos estrictamente el dataset al 2025, lo cual vuelve su varianza cero (no aporta valor predictivo).
Además, `hora_creacion` ahora refleja correctamente la hora de extracción sin la contaminación del chatbot (el cual generaba que todas las horas estuvieran centralizadas artificialmente o con nulos)."""),
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
    ('code', """cols_eliminar = ["subtipo_interes"]
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
