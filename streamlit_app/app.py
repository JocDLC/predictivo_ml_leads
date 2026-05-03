"""
Predictivo ML Leads — Interfaz de predicción de Hot/Cold Leads.

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import re

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
from components.model_report import load_report_artifacts, render_model_report
from components.history import render_history_page
from core.inference import run_inference, load_model_and_artifacts
from core.shap_explainer import explain_lead
from core.memory import init_db, save_session
from core.theme import apply_theme

# 3. Inicialización
init_db()
apply_theme()


def format_model_name(model):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", type(model).__name__).strip()

# 4. Sidebar y Navegación
model, artifacts = load_model_and_artifacts()
report_context = load_report_artifacts()
model_name = format_model_name(model)
umbral = report_context["umbral"]
metrics = report_context.get("metrics_display", {})

with st.sidebar:
    st.image(r"G:\@ DESCARGAS\@ DESARROLLO\predictivo_ml_leads\RENAULT images.png", use_container_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 class="sidebar-brand-title">Predictivo ML</h2>
            <span class="sidebar-brand-subtitle">Lead Scoring Automation</span>
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
    <h4 class="sidebar-card-title" style="text-align: center;">
        ⚙️ Configuración
    </h4>
    <div class="sidebar-card-row">
        <span class="sidebar-card-label">🤖 Modelo</span>
        <span class="sidebar-card-value">{model_name}</span>
    </div>
    <div class="sidebar-card-row" style="margin-bottom: 20px;">
        <span class="sidebar-card-label">🎚️ Umbral</span>
        <span class="sidebar-card-value">{umbral}</span>
    </div>
    <h4 class="sidebar-card-title">
        📈 Rendimiento (Test)
    </h4>
    <div class="sidebar-card-row">
        <span class="sidebar-card-label" title="De todos los leads Hot reales, ¿cuántos logró detectar el modelo?">🎯 Recall</span>
        <span class="sidebar-card-value">{metrics.get('Recall', 'N/D')}</span>
    </div>
    <div class="sidebar-card-row">
        <span class="sidebar-card-label" title="Capacidad general del modelo para separar leads Hot de Cold (1.0 es excelente, 0.5 es azar).">🏆 ROC-AUC</span>
        <span class="sidebar-card-value">{metrics.get('ROC-AUC', 'N/D')}</span>
    </div>
    <div class="sidebar-card-row" style="margin-bottom: 0;">
        <span class="sidebar-card-label" title="Equilibrio general entre no perder oportunidades (Recall) y no generar falsas alarmas (Precisión).">⚖️ F1-Score</span>
        <span class="sidebar-card-value">{metrics.get('F1', 'N/D')}</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v2.2 — Entrenado con datos depurados del año 2025")

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

        total_leads = len(results)
        if "Predicción" in results.columns:
            hot_leads = (results["Predicción"] == "Hot").sum() if results["Predicción"].dtype == object else results["Predicción"].sum()
        elif "Prediccion" in results.columns:
            hot_leads = (results["Prediccion"] == "Hot").sum() if results["Prediccion"].dtype == object else results["Prediccion"].sum()
        else:
            hot_leads = stats.get("Hot", stats.get("hot", 0)) if isinstance(stats, dict) else 0
            
        cold_leads = total_leads - hot_leads
        hot_pct = (hot_leads / total_leads) * 100 if total_leads > 0 else 0
        
        st.success(f"🎉 **{total_leads} leads procesados con éxito:** {int(hot_leads)} Hot ({hot_pct:.1f}%) | {int(cold_leads)} Cold")

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
