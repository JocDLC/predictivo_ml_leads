"""Componente de carga de archivo (Excel / CSV)."""

import streamlit as st


def render_upload():
    """Renderiza el widget de carga y retorna el archivo o None."""
    st.markdown("### 📂 Cargar archivo de Salesforce")
    st.caption(
        "Sube el archivo exportado desde Salesforce (.xlsx, .xls o .csv) con las 28 columnas estándar. "
        "El sistema procesará cada lead y generará una predicción Hot/Cold."
    )

    uploaded = st.file_uploader(
        "Selecciona el archivo (.xlsx, .xls, .csv)",
        type=["xlsx", "xls", "csv"],
        help="Formato esperado: exportación directa de Salesforce con columnas como "
             "'Fecha de creación', 'Nombre del formulario lead', 'Campaña', etc.",
    )

    return uploaded

