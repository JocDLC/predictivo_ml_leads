"""
Componente de historial de sesiones — Sistema de memoria SQLite.

Renderiza el tab 📜 Historial con tabla de sesiones pasadas,
gráfico de tendencias y detalle de predicciones por sesión.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.memory import get_sessions, get_session_predictions, get_trend_data, delete_session


def render_trend_chart(df_trend: pd.DataFrame):
    """Renderiza gráfico de % Hot leads a lo largo del tiempo."""
    st.markdown("### 📈 Tendencia — % Hot Leads por sesión")
    if df_trend.empty:
        st.info("Sin datos de tendencia aún.")
        return

    chart_data = df_trend[["created_at", "pct_hot"]].copy()
    chart_data["created_at"] = pd.to_datetime(chart_data["created_at"])

    ACCENT = "#00C49A"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["created_at"],
            y=chart_data["pct_hot"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=3),
            name="% Hot Leads",
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        yaxis_title="% Hot Leads",
        yaxis=dict(
            ticksuffix="%",
            tickformat=".1f",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )

    st.plotly_chart(fig, use_container_width=True, key="history_trend_chart")


def render_sessions_table(df_sessions: pd.DataFrame) -> int | None:
    """
    Renderiza la tabla de sesiones históricas.

    Returns:
        session_id seleccionada o None.
    """
    st.markdown("### 🗂️ Sesiones anteriores")

    if df_sessions.empty:
        st.info("No hay sesiones guardadas aún. Procesa un archivo para comenzar el historial.")
        return None

    # Formatear columnas para display
    display = df_sessions.copy()
    display = display.rename(columns={
        "id":           "ID",
        "created_at":   "Fecha",
        "file_name":    "Archivo",
        "total_leads":  "Leads",
        "n_hot":        "Hot 🔥",
        "n_cold":       "Cold ❄️",
        "pct_hot":      "% Hot",
        "umbral":       "Umbral",
        "accuracy":     "Accuracy",
    })

    display["% Hot"]     = display["% Hot"].apply(lambda x: f"{x:.1f}%")
    display["Umbral"]    = display["Umbral"].apply(lambda x: f"{x:.2f}")
    display["Accuracy"]  = display["Accuracy"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID":     st.column_config.NumberColumn(width="small"),
            "% Hot":  st.column_config.TextColumn(width="small"),
            "Hot 🔥": st.column_config.NumberColumn(width="small"),
            "Cold ❄️":st.column_config.NumberColumn(width="small"),
        },
    )

    return None


def render_session_detail(session_ids: list[int]):
    """Permite seleccionar una sesión y ver sus predicciones."""
    st.markdown("### 🔍 Ver predicciones de sesión pasada")

    if not session_ids:
        return

    selected_id = st.selectbox(
        "Selecciona una sesión (ID)",
        options=session_ids,
        format_func=lambda x: f"Sesión #{x}",
        key="history_session_select",
    )

    col_view, col_export, col_delete = st.columns([1, 1, 1])

    with col_view:
        if st.button("📥 Cargar predicciones", key="btn_load_history"):
            df = get_session_predictions(selected_id)
            if df.empty:
                st.warning("No se encontraron predicciones para esta sesión.")
            else:
                st.session_state["history_predictions"] = df
                st.session_state["history_session_id"]  = selected_id

    with col_export:
        if "history_predictions" in st.session_state:
            df_export = st.session_state["history_predictions"]
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Exportar CSV",
                data=csv,
                file_name=f"sesion_{st.session_state.get('history_session_id', 'x')}_predicciones.csv",
                mime="text/csv",
                key="btn_export_history",
            )

    with col_delete:
        if st.button("🗑️ Eliminar sesión", key="btn_delete_session", type="secondary"):
            delete_session(selected_id)
            # Limpiar cache si era la sesión visible
            if st.session_state.get("history_session_id") == selected_id:
                st.session_state.pop("history_predictions", None)
                st.session_state.pop("history_session_id",  None)
            st.success(f"Sesión #{selected_id} eliminada.")
            st.rerun()

    if "history_predictions" in st.session_state:
        df_pred = st.session_state["history_predictions"]
        st.markdown(f"**{len(df_pred)} leads** — Sesión #{st.session_state.get('history_session_id')}")

        df_display = df_pred.rename(columns={
            "lead_id":      "Lead ID",
            "prediccion":   "Predicción",
            "prob_hot":     "Prob. Hot %",
            "vehiculo":     "Vehículo",
            "campana":      "Campaña",
            "concesionario":"Concesionario",
            "origen":       "Origen",
        })

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prob. Hot %": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )


def render_history_page():
    """Página completa del historial de sesiones."""
    st.title("📜 Historial de Sesiones")
    st.caption("Registro persistente de todas las predicciones realizadas.")

    df_sessions = get_sessions(limit=50)
    df_trend    = get_trend_data()

    # KPIs rápidos
    if not df_sessions.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total sesiones",   len(df_sessions))
        col2.metric("Total leads eval.", int(df_sessions["total_leads"].sum()))
        col3.metric("Promedio % Hot",   f"{df_sessions['pct_hot'].mean():.1f}%")
        col4.metric("Última sesión",    df_sessions["created_at"].iloc[0][:10])

        st.markdown("---")

    # Gráfico de tendencia
    render_trend_chart(df_trend)
    st.markdown("---")

    # Tabla de sesiones
    render_sessions_table(df_sessions)
    st.markdown("---")

    # Detalle de sesión
    if not df_sessions.empty:
        session_ids = df_sessions["id"].tolist()
        render_session_detail(session_ids)
