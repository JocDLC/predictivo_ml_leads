"""
Predictivo ML Leads — Interfaz de predicción de Hot/Cold Leads.

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

from components.upload import render_upload
from components.results_grid import render_stats, render_grid
from components.lead_detail import render_lead_selector, render_lead_detail
from core.inference import run_inference
from core.shap_explainer import explain_lead

# ─────────────────────────── Config ───────────────────────────

st.set_page_config(
    page_title="Predictivo ML Leads",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": "Predictivo ML Leads v1.0 — Clasificación de Hot/Cold Leads para Renault México.",
    },
)

# ─────────────────────────── Sidebar ───────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Predictivo ML Leads")
    st.markdown("---")

    umbral = st.slider(
        "Umbral de decisión",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Si la probabilidad Hot supera este umbral, el lead se clasifica como Hot.",
    )

    with st.expander("ℹ️ ¿Qué es el umbral?"):
        st.markdown(
            "El **umbral** define el punto de corte para clasificar un lead como Hot o Cold.\n\n"
            "- **Bajar el umbral** (ej. 0.20): Más leads se clasifican como Hot. "
            "Se capturan más oportunidades reales pero también se envían más leads fríos "
            "a los concesionarios (mayor Recall, menor Precision).\n\n"
            "- **Subir el umbral** (ej. 0.60): Solo los leads con alta confianza se marcan como Hot. "
            "Se envían menos leads fríos pero se pierden más oportunidades reales "
            "(menor Recall, mayor Precision).\n\n"
            "- **Valor recomendado: 0.35** — Optimizado para no perder Hot Leads, "
            "ya que en el negocio es peor perder una venta que atender un lead frío."
        )

    st.markdown("---")
    st.markdown(
        "**Modelo:** Random Forest\n\n"
        "**Métrica clave:** Recall 92.4%\n\n"
        "**ROC-AUC:** 0.9476"
    )
    st.markdown("---")
    st.caption("v1.0 — Entrenado con datos dic 2025 - ene 2026")

# ─────────────────────────── Main ───────────────────────────

st.title("🎯 Predicción de Hot/Cold Leads")

uploaded_file = render_upload()

if uploaded_file is not None:
    # Procesar archivo (o re-procesar si cambió el umbral)
    file_changed = st.session_state.get("file_name") != uploaded_file.name
    umbral_changed = st.session_state.get("umbral_used") != umbral

    if "results" not in st.session_state or file_changed or umbral_changed:
        with st.spinner("Procesando leads... esto puede tardar unos segundos."):
            try:
                results, df_model, stats = run_inference(uploaded_file, umbral)
                st.session_state["results"] = results
                st.session_state["df_model"] = df_model
                st.session_state["stats"] = stats
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["umbral_used"] = umbral
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
                st.stop()

    results = st.session_state["results"]
    df_model = st.session_state["df_model"]
    stats = st.session_state["stats"]

    # Métricas resumen
    st.markdown("---")
    render_stats(stats)

    # Grilla de resultados
    st.markdown("---")
    df_filtered = render_grid(results)

    # Detalle individual
    st.markdown("---")
    if len(df_filtered) > 0:
        selected_idx = render_lead_selector(df_filtered)

        if selected_idx is not None:
            # El índice en df_filtered corresponde al índice original en results
            original_idx = df_filtered.index[selected_idx]
            model_row = df_model.iloc[original_idx]

            with st.spinner("Calculando explicación SHAP..."):
                explanation_df, base_value, predicted_proba = explain_lead(model_row)

            render_lead_detail(df_filtered, explanation_df, base_value, predicted_proba, selected_idx)
    else:
        st.warning("No hay leads que coincidan con los filtros seleccionados.")

else:
    # Estado vacío
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1️⃣ Sube el Excel")
        st.markdown("Exporta los leads desde Salesforce en formato `.xlsx`")
    with col2:
        st.markdown("#### 2️⃣ Predicción automática")
        st.markdown("El modelo clasifica cada lead como **Hot** 🔥 o **Cold** ❄️")
    with col3:
        st.markdown("#### 3️⃣ Explora resultados")
        st.markdown("Filtra, ordena y ve **por qué** cada lead fue clasificado así")
