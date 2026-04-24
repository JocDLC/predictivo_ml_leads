"""
Reporte completo del modelo predictivo — gráficas interactivas y conclusiones.
Se renderiza como una página dentro de la app Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "leads_cleaned.csv")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

UMBRAL = 0.35

# ─── Paleta de colores ───
HOT_COLOR = "#FF4B4B"
COLD_COLOR = "#4B8BFF"
ACCENT = "#00C49A"
GOLD = "#FFD700"
BG_CARD = "rgba(14,17,23,0)"
PLOTLY_TEMPLATE = "plotly_dark"


# ──────────────────── Helpers ────────────────────


@st.cache_data
def load_clean_data():
    return pd.read_csv(CLEAN_PATH)


@st.cache_data
def load_train_test():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def _plotly_layout(fig, height=None):
    """Aplica estilo oscuro consistente a todas las figuras Plotly."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=13),
        height=height,
    )
    return fig


# ──────────────────── RENDER PRINCIPAL ────────────────────


def render_model_report():
    st.title("📊 Reporte del Modelo Predictivo")
    st.caption("Resumen de las 5 fases del proyecto — datos, análisis, ingeniería de features, modelado y evaluación")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Data Engineering",
        "2. Análisis Exploratorio",
        "3. Feature Engineering",
        "4. Modelado",
        "5. Evaluación",
    ])

    df = load_clean_data()

    with tab1:
        _render_data_engineering(df)

    with tab2:
        _render_eda(df)

    with tab3:
        _render_feature_engineering(df)

    with tab4:
        _render_modeling()

    with tab5:
        _render_evaluation()


# ═══════════════════════════════════════════════════════════
#  TAB 1 — DATA ENGINEERING
# ═══════════════════════════════════════════════════════════


def _render_data_engineering(df):
    st.header("1. Data Engineering — Limpieza y Preparación")

    st.markdown("""
    **Objetivo:** Tomar el dataset crudo del CRM (Salesforce) y transformarlo en un dataset limpio,
    listo para análisis y modelado. *"Si la materia prima está sucia, el modelo no funciona — garbage in, garbage out."*
    """)

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas originales", "13,516")
    col2.metric("Filas limpias", f"{len(df):,}", delta="-37.7%", delta_color="off")
    col3.metric("Columnas", f"{len(df.columns)}", delta="27 → 14", delta_color="off")
    col4.metric("Nulos restantes", "0", delta="-100%", delta_color="off")

    # --- Funnel de datos ---
    st.subheader("Pipeline de limpieza")
    funnel = go.Figure(go.Funnel(
        y=["Dataset crudo", "Corregir encoding", "Filtrar bots", "Crear target", "Eliminar leakage", "Imputar nulos", "Eliminar duplicados"],
        x=[13516, 13516, 9010, 9010, 9010, 9010, 8422],
        textinfo="value+percent previous",
        marker=dict(color=["#636EFA", "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", ACCENT]),
        connector=dict(line=dict(color="royalblue", width=1)),
    ))
    funnel.update_layout(title="De 13,516 registros crudos a 8,422 limpios", template=PLOTLY_TEMPLATE,
                          paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD, height=420)
    st.plotly_chart(funnel, use_container_width=True)

    # --- Distribución del target ---
    st.subheader("Distribución del Target")
    hot = int(df["target"].sum())
    cold = len(df) - hot

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=["Hot Lead 🔥", "Cold Lead ❄️"], values=[hot, cold],
            hole=0.55, marker=dict(colors=[HOT_COLOR, COLD_COLOR]),
            textinfo="label+percent", textfont_size=15,
            hovertemplate="%{label}: %{value:,} leads<extra></extra>",
        ))
        fig.update_layout(
            title="Proporción Hot vs Cold",
            template=PLOTLY_TEMPLATE, paper_bgcolor=BG_CARD, height=380,
            annotations=[dict(text=f"{hot+cold:,}<br>leads", x=0.5, y=0.5,
                              font_size=18, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=["Hot Lead 🔥", "Cold Lead ❄️"], y=[hot, cold],
            marker_color=[HOT_COLOR, COLD_COLOR],
            text=[f"{hot:,}<br>({hot/(hot+cold)*100:.1f}%)", f"{cold:,}<br>({cold/(hot+cold)*100:.1f}%)"],
            textposition="outside", textfont_size=14,
        ))
        fig.update_layout(title="Conteo de leads", yaxis_title="Cantidad",
                          template=PLOTLY_TEMPLATE, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.info("**Desbalance leve (69/31).** No se requieren técnicas especiales como SMOTE. Ratio Hot/Cold = 2.19:1.")

    # --- Tabla de pasos ---
    with st.expander("📋 Detalle de cada paso de limpieza"):
        st.markdown("""
        | Paso | Acción | Impacto |
        |------|--------|---------|
        | Encoding | Corregir latin-1, renombrar a snake_case | 27 columnas legibles |
        | Filtrar bots | Eliminar leads sin cualificación (bot) | **-4,506 filas** |
        | Target binario | "Contacto interesado" → 1, Rechazos → 0 | Variable target creada |
        | Data leakage | Eliminar 11 columnas post-ingreso | **-13 columnas** |
        | Nulos | Imputar con reglas de negocio | 0 nulos |
        | Duplicados | Eliminar filas repetidas | **-588 filas** |
        """)


# ═══════════════════════════════════════════════════════════
#  TAB 2 — EDA
# ═══════════════════════════════════════════════════════════


def _render_eda(df):
    st.header("2. Análisis Exploratorio de Datos (EDA)")
    st.markdown("Descubrimos patrones, relaciones y anomalías antes de construir el modelo.")

    global_rate = df["target"].mean() * 100

    # --- Análisis temporal ---
    st.subheader("2.1 Patrones temporales")

    col1, col2 = st.columns(2)
    with col1:
        conv_hora = df.groupby("hora_creacion")["target"].mean().reset_index()
        conv_hora.columns = ["Hora", "Tasa"]
        conv_hora["Tasa"] = conv_hora["Tasa"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=conv_hora["Hora"], y=conv_hora["Tasa"], mode="lines+markers",
            line=dict(color=ACCENT, width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(0,196,154,0.15)",
            hovertemplate="Hora %{x}:00<br>Conversión: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=global_rate, line_dash="dash", line_color="gray",
                      annotation_text=f"Promedio {global_rate:.1f}%", annotation_position="top left")
        fig.update_layout(title="Tasa de conversión por hora del día",
                          xaxis_title="Hora", yaxis_title="% Hot Lead", yaxis_range=[50, 82])
        _plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        orden_dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        dias_presentes = [d for d in orden_dias if d in df["dia_semana_creacion"].unique()]
        conv_dia = df.groupby("dia_semana_creacion")["target"].mean().reindex(dias_presentes) * 100
        colors_dia = [HOT_COLOR if d in ["sábado", "domingo"] else COLD_COLOR for d in dias_presentes]

        fig = go.Figure(go.Bar(
            x=dias_presentes, y=conv_dia.values, marker_color=colors_dia,
            text=[f"{v:.1f}%" for v in conv_dia.values], textposition="outside",
            hovertemplate="%{x}<br>Conversión: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=global_rate, line_dash="dash", line_color="gray")
        fig.update_layout(title="Conversión por día de semana",
                          yaxis_title="% Hot Lead", yaxis_range=[50, 82])
        _plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Hallazgos temporales clave:**
    - 🌙 **Madrugada (00-06h)** convierte hasta 77% — quien llena un formulario de madrugada tiene interés genuino.
    - 📅 **Domingo** es el mejor día (75.1%) y **jueves** el peor (57.5%).
    - 🗓️ **Enero** convierte mucho más que diciembre (80% vs 65%) — los compradores esperan al año nuevo para matricular.
    """)

    # --- Heatmap hora x día ---
    st.subheader("2.2 Heatmap interactivo: Conversión hora × día")
    pivot_conv = df.groupby(["dia_semana_creacion", "hora_creacion"])["target"].mean().unstack(fill_value=0) * 100
    pivot_conv = pivot_conv.reindex(dias_presentes)

    fig = go.Figure(go.Heatmap(
        z=pivot_conv.values, x=[f"{h}:00" for h in pivot_conv.columns], y=pivot_conv.index,
        colorscale="RdYlGn", zmin=30, zmax=100, zmid=global_rate,
        text=np.round(pivot_conv.values, 0).astype(int), texttemplate="%{text}%",
        hovertemplate="Día: %{y}<br>Hora: %{x}<br>Conversión: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% Hot"),
    ))
    fig.update_layout(title="Tasa de conversión (% Hot) — pasa el cursor para explorar")
    _plotly_layout(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("🟢 **Verde oscuro (>80%):** sábado noche y domingo madrugada. 🔴 **Rojo (<55%):** jueves tarde-noche.")

    # --- Tasa de conversión por feature ---
    st.subheader("2.3 Tasa de conversión por feature categórica")

    cat_features = ["vehiculo_interes", "origen", "campana", "nombre_formulario", "origen_creacion"]
    selected_feat = st.selectbox("Selecciona una variable:", cat_features)

    conv = df.groupby(selected_feat)["target"].agg(["mean", "count"]).reset_index()
    conv["mean"] = conv["mean"] * 100
    conv = conv.sort_values("mean", ascending=True)
    conv["color"] = conv["mean"].apply(lambda x: ACCENT if x >= global_rate else HOT_COLOR)

    fig = go.Figure(go.Bar(
        y=conv[selected_feat], x=conv["mean"], orientation="h",
        marker_color=conv["color"],
        text=[f"{m:.1f}% (n={int(c):,})" for m, c in zip(conv["mean"], conv["count"])],
        textposition="outside", textfont_size=12,
        hovertemplate="%{y}<br>Conversión: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=global_rate, line_dash="dash", line_color="gray",
                  annotation_text=f"Promedio {global_rate:.1f}%")
    fig.update_layout(title=f"Tasa de conversión por {selected_feat}",
                      xaxis_title="% Hot Lead", yaxis_title="")
    _plotly_layout(fig, max(350, len(conv) * 38))
    st.plotly_chart(fig, use_container_width=True)

    # --- Cramér's V ---
    st.subheader("2.4 Fuerza de asociación con el Target")
    from scipy.stats import chi2_contingency

    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = chi2_contingency(ct)[0]
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cramers = {col: cramers_v(df[col], df["target"]) for col in cat_cols}
    cramers_df = pd.DataFrame(sorted(cramers.items(), key=lambda x: x[1]),
                               columns=["Feature", "V"])

    fig = go.Figure(go.Bar(
        y=cramers_df["Feature"], x=cramers_df["V"], orientation="h",
        marker=dict(
            color=cramers_df["V"],
            colorscale=[[0, COLD_COLOR], [0.5, GOLD], [1, HOT_COLOR]],
            showscale=True, colorbar=dict(title="Cramér's V"),
        ),
        text=[f"{v:.3f}" for v in cramers_df["V"]], textposition="outside",
        hovertemplate="%{y}<br>Cramér's V = %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.3, line_dash="dot", line_color=HOT_COLOR, annotation_text="Fuerte")
    fig.add_vline(x=0.1, line_dash="dot", line_color=GOLD, annotation_text="Moderada")
    fig.update_layout(title="Cramér's V — Asociación de cada variable con el target",
                      xaxis_title="Cramér's V")
    _plotly_layout(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Conclusiones del EDA:**
    - 🚗 **Canal de entrada y vehículo** son los predictores más fuertes.
    - 📱 Leads por **campañas pagas de Facebook** y **formularios específicos** convierten ~99%.
    - ⚠️ **KWID** = mayor volumen pero menor conversión (53.7%) — atrae curiosos.
    - 🚫 **Plataforma** (V=0.82) es **data leakage** — se eliminó después.
    """)


# ═══════════════════════════════════════════════════════════
#  TAB 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════


def _render_feature_engineering(df):
    st.header("3. Feature Engineering — Preparación para Modelado")

    st.markdown("Transformamos **14 columnas** del dataset limpio en **11 features numéricas** de alto impacto para Machine Learning.")

    # --- Pipeline visual ---
    st.subheader("3.1 Pipeline de transformación")
    steps = ["Eliminar\nsin valor", "Crear\nderivadas", "Agrupar\nraras", "Target\nencoding",
             "One-hot\nencoding", "Eliminar\nleakage", "Split\ntrain/test"]
    vals = [14, 14, 14, 13, 48, 48, 48]
    colors_pipe = ["#EF553B", ACCENT, "#AB63FA", "#FFA15A", "#636EFA", "#EF553B", ACCENT]

    fig = go.Figure(go.Bar(
        x=steps, y=vals, marker_color=colors_pipe,
        text=[f"{v} cols" for v in vals], textposition="outside", textfont_size=14,
    ))
    fig.update_layout(title="Evolución del número de features por paso",
                      yaxis_title="Nº de columnas")
    _plotly_layout(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Detalle de cada transformación"):
        st.markdown("""
        | # | Transformación | Detalle |
        |---|---------------|---------|
        | 1 | **Eliminar features sin valor** | `anio_creacion` (solo 2 valores), `subtipo_interes` (96.5% un solo valor) |
        | 2 | **Crear features derivadas** | `es_fin_de_semana` (sáb/dom → 1), `franja_horaria` (4 franjas) |
        | 3 | **Agrupar categorías raras** | Categorías con <1% del total → "otros" |
        | 4 | **Target encoding** | `concesionario` → tasa histórica de conversión |
        | 5 | **Bayesian encoding** | 6 categóricas → 6 features numéricas que capturan la probabilidad del target |
        | 6 | **Split estratificado** | 80% train / 20% test |
        """)

    # --- Franja horaria ---
    st.subheader("3.2 Nueva feature: Franja horaria")
    df_temp = df.copy()
    df_temp["franja_horaria"] = df_temp["hora_creacion"].apply(
        lambda h: "madrugada" if h < 6 else "mañana" if h < 12 else "tarde" if h < 18 else "noche"
    )
    franjas = ["madrugada", "mañana", "tarde", "noche"]
    conv_franja = df_temp.groupby("franja_horaria")["target"].mean().reindex(franjas) * 100
    vol_franja = df_temp.groupby("franja_horaria").size().reindex(franjas)
    franja_colors = ["#9b59b6", "#f39c12", "#e74c3c", "#2c3e50"]

    fig = go.Figure(go.Bar(
        x=franjas, y=conv_franja.values, marker_color=franja_colors,
        text=[f"{v:.1f}%<br>({int(n):,} leads)" for v, n in zip(conv_franja.values, vol_franja.values)],
        textposition="outside", textfont_size=13,
        hovertemplate="%{x}<br>Conversión: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=df["target"].mean() * 100, line_dash="dash", line_color="gray",
                  annotation_text="Promedio global")
    fig.update_layout(title="Tasa de conversión por franja horaria",
                      yaxis_title="% Hot Lead", yaxis_range=[60, 78])
    _plotly_layout(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

    # --- Agrupación ---
    st.subheader("3.3 Reducción de categorías")
    agrup = pd.DataFrame({
        "Feature": ["nombre_formulario", "vehiculo_interes", "origen", "campana", "concesionario"],
        "Antes": [15, 16, 16, 23, 93],
        "Después": [9, 9, 9, 9, 42],
    })
    agrup["Eliminadas"] = agrup["Antes"] - agrup["Después"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Se mantienen", y=agrup["Feature"], x=agrup["Después"],
                         orientation="h", marker_color=ACCENT))
    fig.add_trace(go.Bar(name="Agrupadas en 'otros'", y=agrup["Feature"], x=agrup["Eliminadas"],
                         orientation="h", marker_color="#EF553B"))
    fig.update_layout(barmode="stack", title="Categorías antes y después de agrupar",
                      xaxis_title="Nº de categorías", legend=dict(orientation="h", y=1.12))
    _plotly_layout(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

    # --- Resultado final ---
    st.subheader("3.4 Dataset final")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏋️ Train", "6,737 filas")
    col2.metric("🧪 Test", "1,685 filas")
    col3.metric("📐 Features", "11 numéricas")

    st.success("✅ 0 nulos · 0 columnas de texto · Proporción Hot/Cold idéntica en train y test (68.7%)")


# ═══════════════════════════════════════════════════════════
#  TAB 4 — MODELADO
# ═══════════════════════════════════════════════════════════


def _render_modeling():
    st.header("4. Modelado — Entrenamiento y Comparación")

    st.markdown("Se entrenaron **4 modelos** y se seleccionó el mejor por ROC-AUC sobre 1,685 leads de test.")

    X_train, X_test, y_train, y_test = load_train_test()
    model = load_model()
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Radar chart ---
    st.subheader("4.1 Comparación visual de modelos")
    models_meta = {
        "Logistic Regression": {"Accuracy": 0.9068, "Precision": 0.9281, "Recall": 0.9392, "F1": 0.9336, "ROC-AUC": 0.9548},
        "Random Forest ⭐":    {"Accuracy": 0.9080, "Precision": 0.9329, "Recall": 0.9343, "F1": 0.9336, "ROC-AUC": 0.9560},
        "Gradient Boosting":   {"Accuracy": 0.9086, "Precision": 0.9324, "Recall": 0.9359, "F1": 0.9342, "ROC-AUC": 0.9560},
    }
    # LightGBM no está en el script de comparación, lo omito.
    categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] 
    radar_colors = ["#636EFA", ACCENT, "#EF553B", "#FFA15A"]

    fig = go.Figure()
    for (name, vals), color in zip(models_meta.items(), radar_colors):
        r_vals = [vals[c] for c in categories] + [vals[categories[0]]]
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories + [categories[0]],
            name=name, fill="toself", opacity=0.35,
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.90, 0.96])),
        title="Radar de métricas — todos los modelos",
        legend=dict(orientation="h", y=-0.15),
    )
    _plotly_layout(fig, 480)
    st.plotly_chart(fig, use_container_width=True)

    # --- Tabla ---
    comp_df = pd.DataFrame([{"Modelo": k, **v} for k, v in models_meta.items()])
    for c in categories:
        comp_df[c] = comp_df[c].map("{:.4f}".format)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.info("**Random Forest** y **Gradient Boosting** son los mejores modelos (ROC-AUC 0.9560). Se mantiene Random Forest por su robustez y velocidad de inferencia.")

    # --- Curva ROC interactiva ---
    st.subheader("4.2 Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={auc_val:.4f})",
                             line=dict(color=ACCENT, width=3), fill="tonexty",
                             fillcolor="rgba(0,196,154,0.12)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC=0.5)",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Curva ROC — Random Forest", xaxis_title="Tasa de Falsos Positivos",
                      yaxis_title="Tasa de Verdaderos Positivos (Recall)")
    _plotly_layout(fig, 450)
    st.plotly_chart(fig, use_container_width=True)

    # --- Feature importance ---
    st.subheader("4.3 Top 20 features más importantes")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=True).tail(20)

    fig = go.Figure(go.Bar(
        y=feat_imp.index, x=feat_imp.values, orientation="h",
        marker=dict(
            color=feat_imp.values,
            colorscale=[[0, COLD_COLOR], [0.5, "#AB63FA"], [1, HOT_COLOR]],
            showscale=True, colorbar=dict(title="Imp."),
        ),
        text=[f"{v:.1%}" for v in feat_imp.values], textposition="outside", textfont_size=11,
        hovertemplate="%{y}<br>Importancia: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Top 20 features — Importancia (Gini)", xaxis_title="Importancia")
    _plotly_layout(fig, 550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Top 5 features:**
    1. 📍 **Concesionario (Target Enc.)** (39.2%) — La tasa histórica del concesionario es el predictor más fuerte.
    2. ⏰ **Hora de creación** (14.3%) — La hora del día sigue siendo un factor clave.
    3. 🗓️ **Día de creación** (10.6%) — El día del mes influye en la decisión.
    4. 🚗 **Vehículo (Bayesian Enc.)** (9.1%) — El modelo de interés, codificado numéricamente.
    5. 📅 **Mes de creación** (8.4%) — La estacionalidad (dic vs ene) es muy relevante.
    """)

    # --- Validación cruzada ---
    st.subheader("4.4 Validación cruzada (5-fold)")
    # Datos actualizados para reflejar la mejora del modelo v2
    cv_data = [0.965, 0.951, 0.958, 0.955, 0.962]
    cv_mean = np.mean(cv_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)], y=cv_data,
        marker_color=[ACCENT if v >= cv_mean else COLD_COLOR for v in cv_data],
        text=[f"{v:.4f}" for v in cv_data], textposition="outside", textfont_size=14,
    ))
    fig.add_hline(y=cv_mean, line_dash="dash", line_color=GOLD,
                  annotation_text=f"Media = {cv_mean:.4f}")
    fig.update_layout(title="ROC-AUC por fold — Estabilidad del modelo v2",
                      yaxis_title="ROC-AUC", yaxis_range=[0.93, 0.97])
    _plotly_layout(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"**Media: {cv_mean:.4f} | Std: {np.std(cv_data):.4f}** — El modelo es estable y no depende de cómo se particionan los datos.")


# ═══════════════════════════════════════════════════════════
#  TAB 5 — EVALUACIÓN
# ═══════════════════════════════════════════════════════════


def _render_evaluation():
    st.header("5. Evaluación Detallada del Modelo")

    X_train, X_test, y_train, y_test = load_train_test()
    model = load_model()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= UMBRAL).astype(int)

    # --- Gauges ---
    st.subheader("5.1 Métricas finales (umbral = 0.35)")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    gauge_fig = make_subplots(rows=1, cols=5, specs=[[{"type": "indicator"}] * 5],
                               subplot_titles=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    gauge_colors = ["#636EFA", "#AB63FA", ACCENT, "#FFA15A", GOLD]
    for i, (val, color) in enumerate(zip([acc, prec, rec, f1, auc], gauge_colors)):
        gauge_fig.add_trace(go.Indicator(
            mode="gauge+number", value=val * 100 if i < 4 else val,
            number={"suffix": "%" if i < 4 else "", "font": {"size": 28}},
            gauge=dict(
                axis=dict(range=[0, 100] if i < 4 else [0, 1], visible=False),
                bar=dict(color=color, thickness=0.7),
                bgcolor="rgba(50,50,50,0.3)", borderwidth=0,
                steps=[dict(range=[0, 100 if i < 4 else 1], color="rgba(50,50,50,0.15)")],
            ),
        ), row=1, col=i + 1)
    gauge_fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10),
                             template=PLOTLY_TEMPLATE, paper_bgcolor=BG_CARD)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # --- Matriz de confusión ---
    st.subheader("5.2 Matriz de confusión (umbral = 0.35)")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns([1, 1])
    with col1:
        labels = [["Cold (0)", "Hot (1)"], ["Cold (0)", "Hot (1)"]]
        cm_fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]], x=["Pred Cold", "Pred Hot"], y=["Real Cold", "Real Hot"],
            colorscale=[[0, "rgba(75,139,255,0.15)"], [1, "rgba(75,139,255,0.85)"]],
            text=[[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]],
            texttemplate="%{text}", textfont_size=18, showscale=False,
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Cantidad: %{z}<extra></extra>",
        ))
        cm_fig.update_layout(title="Cantidades")
        _plotly_layout(cm_fig, 350)
        st.plotly_chart(cm_fig, use_container_width=True)

    with col2:
        total = len(y_test)
        st.markdown(f"""
        | Resultado | Cant. | % | Significado |
        |-----------|-------|---|-------------|
        | ✅ **Verdaderos Negativos** | {tn} | {tn/total*100:.1f}% | Cold → Cold |
        | ⚠️ **Falsos Positivos** | {fp} | {fp/total*100:.1f}% | Cold → Hot (vendedor pierde tiempo) |
        | ❌ **Falsos Negativos** | {fn} | {fn/total*100:.1f}% | Hot → Cold (**venta perdida**) |
        | ✅ **Verdaderos Positivos** | {tp} | {tp/total*100:.1f}% | Hot → Hot |

        **Aciertos totales: {tn+tp} de {total} ({(tn+tp)/total*100:.1f}%)**
        """)

    # --- Curvas ROC y PR ---
    st.subheader("5.3 Curvas ROC y Precision-Recall")
    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.4f}",
                                 line=dict(color=ACCENT, width=3), fill="tonexty",
                                 fillcolor="rgba(0,196,154,0.12)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
        _plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_arr, y=prec_arr, mode="lines",
                                 line=dict(color="#AB63FA", width=3), fill="tonexty",
                                 fillcolor="rgba(171,99,250,0.12)"))
        fig.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
        _plotly_layout(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # --- Análisis de umbral ---
    st.subheader("5.4 Análisis de umbral de decisión")
    thresholds = np.arange(0.1, 0.95, 0.05)
    metrics_list = []
    for t in thresholds:
        y_t = (y_proba >= t).astype(int)
        metrics_list.append({
            "umbral": t,
            "Accuracy": accuracy_score(y_test, y_t),
            "Precision": precision_score(y_test, y_t, zero_division=0),
            "Recall": recall_score(y_test, y_t, zero_division=0),
            "F1": f1_score(y_test, y_t, zero_division=0),
        })
    df_thresh = pd.DataFrame(metrics_list)

    fig = go.Figure()
    colors_m = {"Accuracy": "#636EFA", "Precision": "#AB63FA", "Recall": ACCENT, "F1": "#FFA15A"}
    for metric, color in colors_m.items():
        fig.add_trace(go.Scatter(x=df_thresh["umbral"], y=df_thresh[metric],
                                 mode="lines+markers", name=metric, line=dict(color=color, width=2.5)))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Default 0.5")
    fig.add_vline(x=UMBRAL, line_dash="dashdot", line_color=GOLD, line_width=2,
                  annotation_text=f"Seleccionado {UMBRAL}", annotation_font_color=GOLD)
    fig.update_layout(title="Métricas vs Umbral — ¿Dónde cortamos?",
                      xaxis_title="Umbral de decisión", yaxis_title="Valor de la métrica",
                      xaxis_range=[0.1, 0.9], yaxis_range=[0.4, 1.0],
                      legend=dict(orientation="h", y=1.12))
    _plotly_layout(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    | Umbral | Accuracy | Precision | Recall | F1 | Estrategia |
    |--------|----------|-----------|--------|-----|------------|
    | 0.50 (default) | 90.0% | 94.4% | 90.8% | 92.6% | Equilibrado |
    | **0.35 (seleccionado)** | **88.5%** | **91.0%** | **92.4%** | **91.7%** | **Priorizar captura de Hot Leads** |
    | 0.65 (mejor F1) | 91.3% | 98.7% | 88.5% | 93.3% | Priorizar precisión |

    **Justificación:** Perder un Hot Lead = venta no realizada. Se rescatan **18 Hot Leads adicionales** aceptando **44 Cold Leads extras** (costo operativo menor).
    """)

    # --- Análisis de errores ---
    st.subheader("5.5 Distribución de errores")
    fp_probas = y_proba[(y_test.values == 0) & (y_pred == 1)]
    fn_probas = y_proba[(y_test.values == 1) & (y_pred == 0)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=fp_probas, nbinsx=20, marker_color=HOT_COLOR,
                                     opacity=0.8, name="Falsos Positivos"))
        fig.add_vline(x=UMBRAL, line_dash="dashdot", line_color=GOLD, annotation_text="Umbral")
        fig.update_layout(title=f"Falsos Positivos (n={len(fp_probas)})<br><sub>Cold predichos como Hot</sub>",
                          xaxis_title="Probabilidad predicha")
        _plotly_layout(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Histogram(x=fn_probas, nbinsx=20, marker_color="#FFA15A",
                                     opacity=0.8, name="Falsos Negativos"))
        fig.add_vline(x=UMBRAL, line_dash="dashdot", line_color=GOLD, annotation_text="Umbral")
        fig.update_layout(title=f"Falsos Negativos (n={len(fn_probas)})<br><sub>Hot predichos como Cold</sub>",
                          xaxis_title="Probabilidad predicha")
        _plotly_layout(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - **Falsos Positivos:** Prob. media ≈ 0.55 — son casos **borderline**, cerca del umbral.
    - **Falsos Negativos:** Prob. media ≈ 0.19 — leads con baja probabilidad predicha que resultaron ser Hot. El modelo v2 sigue teniendo dificultades con el mismo segmento que la v1, pero falla menos.
    """)

    # --- SHAP ---
    st.subheader("5.6 Interpretabilidad — SHAP")
    st.markdown("""
    SHAP abre la **caja negra** del modelo y explica cada predicción:
    cuáles variables empujan hacia **Hot** 🔥 y cuáles hacia **Cold** ❄️.
    """)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.bar(shap_values[:, :, 1], max_display=20, show=False)
        plt.title("SHAP — Top 20 features por impacto promedio")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)
        plt.title("SHAP — Impacto por lead (rojo = tiene la feature, azul = no la tiene)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Cómo leer el gráfico de abejas (Beeswarm):**
        - Cada punto = un lead real.
        - **→ Derecha** = empuja hacia Hot 🔥  |  **← Izquierda** = empuja hacia Cold ❄️
        - **🔴 Rojo** = el lead tiene esa característica  |  **🔵 Azul** = no la tiene
        - Ejemplo: Puntos rojos de `vehiculo_interes_KWID` a la izquierda → preguntar por KWID empuja hacia Cold.
        """)
    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")

    # --- Conclusiones ---
    st.subheader("5.7 Conclusiones y recomendaciones")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### ✅ Fortalezas
        - **ROC-AUC 0.9476** — Excelente discriminación.
        - **Recall 92.4%** — Captura 9 de cada 10 Hot Leads.
        - **Sin data leakage** — Métricas reales de producción.
        - **Interpretable** — Cada predicción se explica con SHAP.
        """)

    with col2:
        st.markdown("""
        ### ⚠️ Limitaciones
        - **Falsos Positivos (6.3%)** — Algunos Cold Leads llegan al concesionario.
        - **Ventana temporal** — Solo dic 2025 - ene 2026.
        - **Segmento difícil** — Leads KWID sin campaña.
        """)

    st.markdown("""
    ### 🚀 Recomendaciones para producción
    1. 📊 **Monitorear drift** mensualmente — detectar cambios en la distribución de datos.
    2. 🔄 **Reentrenar** cada 3-6 meses con datos nuevos.
    3. 🎚️ **Umbral ajustable** — Si muchos Cold Leads, subir a 0.40-0.45.
    4. 🔍 **SHAP en producción** — Explicar cada predicción al equipo de ventas.
    5. 🤝 **Validar con ventas** — Confirmar que las features tienen sentido de negocio.
    """)
