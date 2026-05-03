import streamlit as st


def apply_theme():
    """Inyecta CSS personalizado compatible con los temas nativo de Streamlit.

    No controla el modo claro/oscuro — eso lo maneja el menú de Streamlit.
    Solo añade estilos para componentes custom (cards, SHAP bars, fuente).
    """
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, label, li {
        font-family: 'Inter', sans-serif;
    }

    /* Padding contenedor principal */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Tarjetas custom — usa variables CSS de Streamlit para adaptarse al tema */
    .custom-card {
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.6));
        color: var(--st-text-color, inherit);
        border: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.2));
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    .custom-card * {
        color: inherit;
    }

    .sidebar-brand-title {
        margin-bottom: 0;
        color: var(--st-heading-color, var(--st-text-color, inherit));
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        display: inline-block;
        color: var(--st-text-color, inherit);
        opacity: 0.72;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }

    .sidebar-card-title {
        margin-top: 0;
        margin-bottom: 15px;
        padding-bottom: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--st-heading-color, var(--st-text-color, inherit));
        border-bottom: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.2));
    }

    .sidebar-card-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.7rem;
        font-size: 0.98rem;
        line-height: 1.45;
    }

    .sidebar-card-label,
    .sidebar-card-value {
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--st-text-color, inherit);
    }

    .sidebar-card-value {
        text-align: right;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.6));
        border: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.2));
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.35));
        color: var(--st-text-color, inherit);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.2));
    }

    [data-testid="stFileUploader"] * {
        color: var(--st-text-color, inherit);
    }

    /* Barras SHAP — estilos base (también se definen inline en render_shap_chart) */
    .shap-container { margin-top: 15px; }
    .shap-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 0.85rem;
        color: var(--st-text-color, inherit);
    }
    .shap-feature-name {
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 60%;
        color: var(--st-text-color, inherit);
    }
    .shap-feature-val {
        font-weight: 600;
        color: var(--st-text-color, inherit);
    }
    .shap-val-hot  { color: #ef4444; }
    .shap-val-cold { color: #3b82f6; }
    .shap-bar-bg {
        width: 100%;
        background-color: var(--st-border-color, rgba(150,150,150,0.2));
        height: 6px;
        border-radius: 4px;
        position: relative;
        margin-bottom: 16px;
    }
    .shap-bar-fill-hot {
        height: 100%;
        background-color: #ef4444;
        border-radius: 4px;
        position: absolute;
        right: 50%;
    }
    .shap-bar-fill-cold {
        height: 100%;
        background-color: #3b82f6;
        border-radius: 4px;
        position: absolute;
        left: 50%;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
