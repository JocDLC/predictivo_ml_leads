"""
Predictivo ML Leads — Interfaz de predicción de Hot/Cold Leads.

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

from components.upload import render_upload
from components.results_grid import render_stats, render_grid
from components.lead_detail import render_lead_selector, render_lead_detail
from components.model_report import render_model_report
from core.inference import run_inference
from core.shap_explainer import explain_lead
from core.inference import load_model_and_artifacts

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

# Cargar los artefactos dinámicamente para los metadatos
_, artifacts = load_model_and_artifacts()
umbral = artifacts.get("umbral", 0.35)
# Métricas del modelo v2 (Random Forest)
metrics = artifacts.get("metrics", {"Recall": "93.4%", "ROC-AUC": "0.9560"})

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin-bottom: 0;">🚗 Predictivo ML</h2>
            <span style="color: #6c757d; font-size: 14px;">Lead Scoring Automation</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    page = st.radio(
        "Navegación",
        ["🎯 Predicción de Leads", "📊 Reporte del Modelo"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    st.info(f"🤖 **Modelo:** Random Forest\n\n🎚️ **Umbral:** {umbral}")
    st.markdown("### 📈 Rendimiento (Test)")
    st.success(f"🎯 **Recall:** {metrics.get('Recall')}\n\n🏆 **ROC-AUC:** {metrics.get('ROC-AUC')}")
    st.markdown("---")
    st.caption("v2.0 — Entrenado con datos dic 2025 - ene 2026")

# ─────────────────────────── Main ───────────────────────────

if page == "📊 Reporte del Modelo":
    render_model_report()
else:
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
            display_df, selected_idx = render_lead_selector(results, df_filtered)

            if selected_idx is not None:
                original_idx = display_df.index[selected_idx]
                model_row = df_model.iloc[original_idx]

                with st.spinner("Calculando explicación SHAP..."):
                    explanation_df, base_value, predicted_proba = explain_lead(model_row)

                render_lead_detail(display_df, explanation_df, base_value, predicted_proba, selected_idx)
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
