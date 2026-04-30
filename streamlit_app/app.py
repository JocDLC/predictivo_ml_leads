"""
Predictivo ML Leads — Interfaz de predicción de Hot/Cold Leads.

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

# 1. Configuración de página (SIEMPRE PRIMERO)
st.set_page_config(
    page_title="Predictivo ML Leads",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Importaciones
from components.upload import render_upload
from components.results_grid import render_stats, render_grid
from components.lead_detail import render_lead_selector, render_lead_detail
from components.model_report import render_model_report
from components.history import render_history_page
from core.inference import run_inference, load_model_and_artifacts
from core.shap_explainer import explain_lead
from core.memory import init_db, save_session
from core.theme import apply_theme

# 3. Inicialización
init_db()
apply_theme()

# 4. Sidebar y Navegación
_, artifacts = load_model_and_artifacts()
umbral = artifacts.get("umbral", 0.35)
metrics = artifacts.get("metrics", {"Recall": "93.4%", "ROC-AUC": "0.9560"})

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin-bottom: 0;">🚗 Predictivo ML</h2>
            <span style="color: #6c757d; font-size: 14px;">Lead Scoring Automation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navegación",
        ["🎯 Predicción de Leads", "📊 Reporte del Modelo", "📜 Historial"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"""
<div class="custom-card" style="padding: 15px; margin-bottom: 20px;">
    <h4 style="margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid rgba(150,150,150,0.2); padding-bottom: 10px; margin-bottom: 15px;">
        ⚙️ Configuración
    </h4>
    <div style="font-size: 0.95rem; margin-bottom: 8px;">🤖 <b>Modelo:</b> Random Forest</div>
    <div style="font-size: 0.95rem; margin-bottom: 20px;">🎚️ <b>Umbral:</b> {umbral}</div>
    <h4 style="margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid rgba(150,150,150,0.2); padding-bottom: 10px; margin-bottom: 15px;">
        📈 Rendimiento (Test)
    </h4>
    <div style="font-size: 0.95rem; margin-bottom: 8px;">🎯 <b>Recall:</b> {metrics.get('Recall')}</div>
    <div style="font-size: 0.95rem;">🏆 <b>ROC-AUC:</b> {metrics.get('ROC-AUC')}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v2.0 — Entrenado con datos depurados del año 2025")

# 5. Funciones de Página
def render_prediction_page():
    st.title("🎯 Predicción de Hot/Cold Leads")
    uploaded_file = render_upload()

    if uploaded_file is not None:
        if "results" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Procesando leads..."):
                try:
                    results, df_model, stats = run_inference(uploaded_file)
                    st.session_state["results"] = results
                    st.session_state["df_model"] = df_model
                    st.session_state["stats"] = stats
                    st.session_state["file_name"] = uploaded_file.name
                    save_session(uploaded_file.name, results, stats)
                    st.toast("✅ Sesión procesada y guardada")
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

        results = st.session_state["results"]
        df_model = st.session_state["df_model"]
        stats = st.session_state["stats"]

        st.markdown("---")
        render_stats(stats)
        st.markdown("---")
        df_filtered = render_grid(results)
        st.markdown("---")

        if len(df_filtered) > 0:
            display_df, selected_idx = render_lead_selector(results, df_filtered)
            if selected_idx is not None:
                original_idx = display_df.index[selected_idx]
                model_row = df_model.iloc[original_idx]
                with st.spinner("Calculando SHAP..."):
                    explanation_df, base_value, predicted_proba = explain_lead(model_row)
                render_lead_detail(display_df, explanation_df, base_value, predicted_proba, selected_idx)
    else:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1️⃣ Sube el Excel\nExporta los leads desde Salesforce en formato `.xlsx`")
        with col2:
            st.markdown("#### 2️⃣ Predicción automática\nEl modelo clasifica cada lead como **Hot** 🔥 o **Cold** ❄️")
        with col3:
            st.markdown("#### 3️⃣ Explora resultados\nFiltra, ordena y ve **por qué** cada lead fue clasificado así")

# 6. Routing (Lógica de exclusión mutua)
main_container = st.container()

with main_container:
    if page == "🎯 Predicción de Leads":
        render_prediction_page()
    elif page == "📊 Reporte del Modelo":
        render_model_report()
    elif page == "📜 Historial":
        render_history_page()
