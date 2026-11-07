"""Componente de carga de archivo (Excel / CSV)."""

import streamlit as st


def render_upload():
    """Renderiza el widget de carga y retorna el archivo o None."""
    col_up, col_guide = st.columns([3, 2], gap="large")

    with col_up:
        st.markdown(
            '<div class="section-header"><h3>📂 Cargar archivo de Salesforce</h3></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Arrastra el archivo aquí o haz clic para seleccionar",
            type=["xlsx", "xls", "csv"],
            help="Exportación directa de Salesforce. Columnas requeridas: "
                 "Fecha de creación, Nombre del formulario lead, Campaña, "
                 "Origen de creación, Vehículo de interés, Nombre corto de la Concesión.",
            label_visibility="visible",
        )

    with col_guide:
        st.markdown(
            """
            <div class="upload-guide">
                <strong>📋 Formatos soportados</strong>
                <ul>
                    <li><strong>.xlsx / .xls</strong> — Excel estándar</li>
                    <li><strong>.csv</strong> — Separado por <code>,</code> o <code>;</code></li>
                </ul>
                <strong>✅ Columnas clave requeridas</strong>
                <ul>
                    <li>Fecha de creación por el cliente</li>
                    <li>Nombre del formulario lead</li>
                    <li>Campaña &amp; Origen de creación</li>
                    <li>Vehículo de interés</li>
                    <li>Nombre corto de la Concesión</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return uploaded
