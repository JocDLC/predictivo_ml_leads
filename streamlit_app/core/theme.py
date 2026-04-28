import streamlit as st

def apply_theme(is_dark_mode=False):
    """Aplica estilos CSS globales a la aplicación Streamlit, con toggle personalizado."""
    
    if not is_dark_mode:
        # Modo Claro
        bg_main = "#f4f6f8"
        bg_sidebar = "#ffffff"
        text_main = "#1e293b"
        text_sidebar = "#334155"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        shadow = "0 4px 6px rgba(0, 0, 0, 0.05)"
    else:
        # Modo Oscuro
        bg_main = "#121212"
        bg_sidebar = "#000000"
        text_main = "#e2e8f0"
        text_sidebar = "#e2e8f0"
        card_bg = "#1e1e1e"
        card_border = "#333333"
        shadow = "0 4px 6px rgba(0, 0, 0, 0.3)"

    css = f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Aplicar fuente solo a elementos de texto para no romper iconos */
    html, body, p, h1, h2, h3, h4, h5, h6, label, li {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Fondos */
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"] {{
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        color: {text_sidebar} !important;
    }}
    
    /* Textos en Sidebar */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {text_sidebar} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="radio"] div {{
        color: {text_sidebar} !important;
    }}
    
    /* Arreglo File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: {card_bg} !important;
        border-radius: 12px;
        padding: 16px;
        box-shadow: {shadow};
        border: 1px solid {card_border};
    }}
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] div[data-testid="stWidgetLabel"] p {{
        color: {text_main} !important;
    }}
    [data-testid="stFileUploadDropzone"] {{
        background-color: transparent !important;
        border: 2px dashed {card_border} !important;
        opacity: 0.6;
    }}
    [data-testid="stFileUploadDropzone"] div, 
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small {{
        color: {text_main} !important;
    }}
    
    /* Estilo de Tarjetas para Métricas */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: {shadow};
    }}
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {{
        color: {text_main} !important;
        font-weight: 500 !important;
        opacity: 0.8;
    }}
    [data-testid="stMetricValue"] {{
        color: {text_main} !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }}
    
    /* Custom Card class */
    .custom-card {{
        background-color: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 20px;
        box-shadow: {shadow};
        margin-bottom: 20px;
    }}
    
    /* Ocultar padding agresivo superior */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }}
    
    header {{
        background-color: transparent !important;
    }}
    
    /* Barras SHAP */
    .shap-container {{
        margin-top: 15px;
    }}
    .shap-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 0.85rem;
        color: {text_main} !important;
    }}
    .shap-feature-name {{
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 60%;
    }}
    .shap-feature-val {{
        font-weight: 600;
    }}
    .shap-val-hot {{ color: #ef4444; }}
    .shap-val-cold {{ color: #3b82f6; }}
    
    .shap-bar-bg {{
        width: 100%;
        background-color: {card_border};
        opacity: 0.5;
        height: 6px;
        border-radius: 4px;
        position: relative;
        margin-bottom: 16px;
    }}
    .shap-bar-fill-hot {{
        height: 100%;
        background-color: #ef4444;
        border-radius: 4px;
        position: absolute;
        right: 50%;
    }}
    .shap-bar-fill-cold {{
        height: 100%;
        background-color: #3b82f6;
        border-radius: 4px;
        position: absolute;
        left: 50%;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
