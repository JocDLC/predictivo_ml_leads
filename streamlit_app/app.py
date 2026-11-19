"""
Predictivo ML Leads — Interfaz de predicción de Hot/Cold Leads.

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import re
from pathlib import Path

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
from components.model_report import render_model_report, try_load_report_artifacts
from components.history import render_history_page
from core.inference import run_inference, load_model_and_artifacts, resolve_umbral
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
report_context = try_load_report_artifacts()
model_name = format_model_name(model)
umbral = report_context["umbral"] if report_context else resolve_umbral(artifacts)
metrics = report_context.get("metrics_display", {}) if report_context else {}

with st.sidebar:
    logo_path = Path(__file__).resolve().parents[1] / "RENAULT images.png"
    if logo_path.exists():
        st.image(str(logo_path), width="stretch")
    else:
        st.caption("Logo no disponible")
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 class="sidebar-brand-title">Predictivo ML</h2>
            <span class="sidebar-brand-subtitle">Lead Scoring Automation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_options = ["🎯 Predicción de Leads", "📜 Historial"]
    if report_context is not None:
        nav_options.insert(1, "📊 Reporte del Modelo")
    page = st.radio(
        "Navegación",
        nav_options,
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
    st.markdown(
        """
        <div class="page-hero">
            <h1>🎯 Predicción de Hot/Cold Leads</h1>
            <p>Sube el archivo exportado desde Salesforce para clasificar automáticamente cada lead.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
                    st.toast("✅ Predicciones generadas y sesión guardada")
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

        results = st.session_state["results"]
        df_model = st.session_state["df_model"]
        stats = st.session_state["stats"]

        total_leads = int(stats.get("total_evaluados", len(results)))
        hot_leads = int(stats.get("n_hot", 0))
        cold_leads = int(stats.get("n_cold", total_leads - hot_leads))
        hot_pct = (hot_leads / total_leads) * 100 if total_leads > 0 else 0
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.success(f"✅ **{total_leads:,}** leads evaluados")
        with c2:
            st.markdown(
                f'<div style="text-align:center;padding:8px 0">' +
                f'<span class="badge-hot">🔥 Hot: {int(hot_leads):,} ({hot_pct:.1f}%)</span></div>',
                unsafe_allow_html=True,
            )
        with c3:
            cold_pct = 100 - hot_pct
            st.markdown(
                f'<div style="text-align:center;padding:8px 0">' +
                f'<span class="badge-cold">❄️ Cold: {int(cold_leads):,} ({cold_pct:.1f}%)</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        render_stats(stats)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        df_filtered = render_grid(results)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if len(df_filtered) > 0:
            display_df, selected_idx = render_lead_selector(results, df_filtered)
            if selected_idx is not None:
                original_idx = display_df.index[selected_idx]
                model_row = df_model.iloc[original_idx]
                with st.spinner("Calculando SHAP..."):
                    explanation_df, base_value, predicted_proba = explain_lead(model_row)
                render_lead_detail(display_df, explanation_df, base_value, predicted_proba, selected_idx)
    else:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("##### ¿Cómo funciona?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="step-card">'
                '<span class="step-emoji">📁</span>'
                '<div class="step-number">1</div>'
                '<div class="step-title">Sube el archivo</div>'
                '<div class="step-desc">Exporta los leads desde Salesforce en formato <strong>.xlsx</strong>, <strong>.xls</strong> o <strong>.csv</strong>.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="step-card">'
                '<span class="step-emoji">🤖</span>'
                '<div class="step-number">2</div>'
                '<div class="step-title">Predicción automática</div>'
                '<div class="step-desc">El modelo clasifica cada lead como <strong>Hot 🔥</strong> o <strong>Cold ❄️</strong> con su probabilidad.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="step-card">'
                '<span class="step-emoji">🔍</span>'
                '<div class="step-number">3</div>'
                '<div class="step-title">Explora resultados</div>'
                '<div class="step-desc">Filtra, ordena y descubre <strong>por qué</strong> cada lead fue clasificado así con explicaciones SHAP.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# 6. Routing (Lógica de exclusión mutua)
main_container = st.container()

with main_container:
    if page == "🎯 Predicción de Leads":
        render_prediction_page()
    elif page == "📊 Reporte del Modelo":
        render_model_report()
    elif page == "📜 Historial":
        render_history_page()


