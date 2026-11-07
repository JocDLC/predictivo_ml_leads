import streamlit as st


def apply_theme():
    """Inyecta CSS personalizado compatible con los temas nativo de Streamlit."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, label, li, span, div {
        font-family: 'Inter', sans-serif;
    }

    /* ── Layout principal ─────────────────────────────────────── */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px;
    }

    /* ── Hero banner de página ────────────────────────────────── */
    .page-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: #fff;
        position: relative;
        overflow: hidden;
    }
    .page-hero::after {
        content: "";
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: rgba(255,255,255,0.04);
    }
    .page-hero h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #fff;
    }
    .page-hero p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.78;
        color: #fff;
    }

    /* ── Tarjetas custom ──────────────────────────────────────── */
    .custom-card {
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.6));
        color: var(--st-text-color, inherit);
        border: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.18));
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.07);
        margin-bottom: 18px;
        transition: box-shadow 0.2s ease;
    }
    .custom-card:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    .custom-card * { color: inherit; }

    /* ── Métricas ─────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.6));
        border: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.18));
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; font-weight: 600; opacity: 0.72; }
    [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800; }

    /* ── Sidebar brand ────────────────────────────────────────── */
    .sidebar-brand-title {
        margin-bottom: 2px;
        color: var(--st-heading-color, var(--st-text-color, inherit));
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .sidebar-brand-subtitle {
        display: inline-block;
        color: var(--st-text-color, inherit);
        opacity: 0.65;
        font-size: 0.88rem;
        margin-top: 0.25rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* ── Sidebar config card ──────────────────────────────────── */
    .sidebar-card-title {
        margin-top: 0;
        margin-bottom: 12px;
        padding-bottom: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--st-text-color, inherit);
        opacity: 0.55;
        border-bottom: 1px solid var(--st-border-color, rgba(150, 150, 150, 0.2));
    }
    .sidebar-card-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        font-size: 0.93rem;
        line-height: 1.4;
    }
    .sidebar-card-label {
        font-size: 0.91rem;
        font-weight: 500;
        color: var(--st-text-color, inherit);
        opacity: 0.8;
    }
    .sidebar-card-value {
        font-size: 0.93rem;
        font-weight: 700;
        color: var(--st-text-color, inherit);
        text-align: right;
        background: var(--st-border-color, rgba(150,150,150,0.12));
        border-radius: 6px;
        padding: 2px 8px;
    }

    /* ── Badges Hot/Cold ──────────────────────────────────────── */
    .badge-hot {
        display: inline-block;
        background: linear-gradient(135deg, #ff4b4b, #e03131);
        color: #fff !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 14px;
        border-radius: 999px;
        box-shadow: 0 2px 8px rgba(255,75,75,0.35);
    }
    .badge-cold {
        display: inline-block;
        background: linear-gradient(135deg, #4b8bff, #1971c2);
        color: #fff !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 14px;
        border-radius: 999px;
        box-shadow: 0 2px 8px rgba(75,139,255,0.35);
    }

    /* ── Lead detail card ─────────────────────────────────────── */
    .lead-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        border-left: 5px solid #ff4b4b;
        background-color: var(--st-secondary-background-color, rgba(240, 242, 246, 0.55));
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }
    .lead-card.cold { border-left-color: #4b8bff; }
    .lead-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 14px 0;
        color: var(--st-heading-color, inherit);
    }
    .lead-info-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 24px;
        font-size: 0.92rem;
    }
    .lead-info-item {
        display: flex;
        flex-direction: column;
        gap: 1px;
    }
    .lead-info-label {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.55;
        color: var(--st-text-color, inherit);
    }
    .lead-info-value {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--st-text-color, inherit);
    }

    /* ── Upload zone ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
        border: 2px dashed var(--st-border-color, rgba(150,150,150,0.35)) !important;
        padding: 10px 16px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
    }

    /* ── File upload instructions card ───────────────────────── */
    .upload-guide {
        border-radius: 12px;
        padding: 16px 20px;
        background: var(--st-secondary-background-color, rgba(240,242,246,0.5));
        border: 1px solid var(--st-border-color, rgba(150,150,150,0.18));
        font-size: 0.88rem;
        margin-top: 12px;
    }
    .upload-guide ul { margin: 8px 0 0 0; padding-left: 18px; }
    .upload-guide li { margin-bottom: 4px; }

    /* ── Section headers ──────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .section-divider {
        border: none;
        border-top: 1px solid var(--st-border-color, rgba(150,150,150,0.2));
        margin: 24px 0;
    }

    /* ── How it works cards ───────────────────────────────────── */
    .step-card {
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        background: var(--st-secondary-background-color, rgba(240,242,246,0.55));
        border: 1px solid var(--st-border-color, rgba(150,150,150,0.18));
        height: 100%;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .step-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px; height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0f3460, #4b8bff);
        color: #fff;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 12px;
    }
    .step-emoji { font-size: 2rem; margin-bottom: 8px; display: block; }
    .step-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 8px; }
    .step-desc { font-size: 0.85rem; opacity: 0.72; line-height: 1.5; }

    /* ── SHAP chart ───────────────────────────────────────────── */
    .shap-container { margin-top: 10px; }
    .shap-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 3px;
        font-size: 0.86rem;
        color: var(--st-text-color, inherit);
    }
    .shap-feature-name {
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 62%;
        color: var(--st-text-color, inherit);
    }
    .shap-feature-val { font-weight: 700; }
    .shap-val-hot  { color: #ef4444; }
    .shap-val-cold { color: #3b82f6; }
    .shap-bar-bg {
        width: 100%;
        background-color: var(--st-border-color, rgba(150,150,150,0.2));
        height: 7px;
        border-radius: 4px;
        position: relative;
        margin-bottom: 14px;
    }
    .shap-bar-fill-hot {
        height: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ef4444);
        border-radius: 4px;
        position: absolute;
        right: 50%;
    }
    .shap-bar-fill-cold {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #4b8bff);
        border-radius: 4px;
        position: absolute;
        left: 50%;
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 18px;
    }

    /* ── Botón de descarga ────────────────────────────────────── */
    [data-testid="stDownloadButton"] button {
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 20px;
    }

    /* ── Scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(150,150,150,0.35); border-radius: 3px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
