"""Componente de carga de archivo Excel."""

import streamlit as st


def render_upload():
    """Renderiza el widget de carga y retorna el archivo o None."""
    st.markdown("### 📂 Cargar archivo de Salesforce")
    st.caption(
        "Sube el Excel exportado desde Salesforce con las 28 columnas estándar. "
        "El sistema procesará cada lead y generará una predicción Hot/Cold."
    )

    uploaded = st.file_uploader(
        "Selecciona el archivo Excel (.xlsx)",
        type=["xlsx"],
        help="Formato esperado: exportación directa de Salesforce con columnas como "
             "'Fecha de creación', 'Nombre del formulario lead', 'Campaña', etc.",
    )

    return uploaded
