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
)

# ─────────────────────────── Sidebar ───────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Renault_2021_Text.svg", width=150)
    st.markdown("# Predictivo ML Leads")
    st.markdown("---")
    st.markdown(
        "**Modelo:** Random Forest\n\n"
        "**Umbral:** 0.35\n\n"
        "**Métrica clave:** Recall 92.4%\n\n"
        "**ROC-AUC:** 0.9476"
    )
    st.markdown("---")
    st.caption("v1.0 — Entrenado con datos dic 2025 - ene 2026")

# ─────────────────────────── Main ───────────────────────────

st.title("🎯 Predicción de Hot/Cold Leads")

uploaded_file = render_upload()

if uploaded_file is not None:
    # Procesar archivo
    if "results" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Procesando leads... esto puede tardar unos segundos."):
            try:
                results, df_model, stats = run_inference(uploaded_file)
                st.session_state["results"] = results
                st.session_state["df_model"] = df_model
                st.session_state["stats"] = stats
                st.session_state["file_name"] = uploaded_file.name
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
