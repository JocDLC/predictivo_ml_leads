"""Componente de grilla de resultados con filtros."""

import streamlit as st
import pandas as pd


def render_stats(stats):
    """Muestra métricas resumen en tarjetas."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total evaluados", f"{stats['total_evaluados']:,}")
    with col2:
        st.metric("Hot 🔥", f"{stats['n_hot']:,}",
                   delta=f"{stats['n_hot']/stats['total_evaluados']*100:.1f}%")
    with col3:
        st.metric("Cold ❄️", f"{stats['n_cold']:,}",
                   delta=f"{stats['n_cold']/stats['total_evaluados']*100:.1f}%",
                   delta_color="inverse")
    with col4:
        if stats["accuracy"] is not None:
            st.metric("Accuracy vs real", f"{stats['accuracy']:.1f}%")
        else:
            st.metric("Umbral", f"{stats['umbral']}")

    if stats["n_bot"] > 0:
        st.info(f"ℹ️ Se excluyeron **{stats['n_bot']:,}** leads del bot (sin cualificación).")


def render_grid(results):
    """Renderiza la grilla interactiva de resultados con filtros."""
    st.markdown("### 📊 Resultados de predicción")

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        filtro_pred = st.multiselect(
            "Filtrar por predicción",
            options=results["prediccion"].unique().tolist(),
            default=results["prediccion"].unique().tolist(),
        )

    with col_f2:
        if "vehiculo_interes" in results.columns:
            vehiculos = ["Todos"] + sorted(results["vehiculo_interes"].dropna().unique().tolist())
            filtro_vehiculo = st.selectbox("Vehículo", vehiculos)
        else:
            filtro_vehiculo = "Todos"

    with col_f3:
        rango_prob = st.slider(
            "Probabilidad Hot (%)",
            min_value=0, max_value=100,
            value=(0, 100),
        )

    # Aplicar filtros
    mask = (
        results["prediccion"].isin(filtro_pred)
        & results["probabilidad_hot"].between(rango_prob[0], rango_prob[1])
    )

    if filtro_vehiculo != "Todos" and "vehiculo_interes" in results.columns:
        mask = mask & (results["vehiculo_interes"] == filtro_vehiculo)

    df_filtered = results[mask].copy()

    st.caption(f"Mostrando {len(df_filtered):,} de {len(results):,} leads")

    # Columnas a mostrar
    display_cols = ["prediccion", "probabilidad_hot"]
    if "lead_id" in df_filtered.columns:
        display_cols = ["lead_id"] + display_cols
    for col in ["vehiculo_interes", "concesionario", "campana",
                 "nombre_formulario", "origen"]:
        if col in df_filtered.columns:
            display_cols.append(col)
    if "real" in df_filtered.columns:
        display_cols.append("real")

    # Ordenar por probabilidad descendente
    df_show = df_filtered[display_cols].sort_values(
        "probabilidad_hot", ascending=False
    ).reset_index(drop=True)

    # Mostrar tabla
    st.dataframe(
        df_show,
        use_container_width=True,
        height=400,
        column_config={
            "prediccion": st.column_config.TextColumn("Predicción", width="small"),
            "probabilidad_hot": st.column_config.ProgressColumn(
                "Prob. Hot %", min_value=0, max_value=100, format="%.1f%%",
            ),
            "lead_id": st.column_config.TextColumn("Lead ID", width="medium"),
            "vehiculo_interes": st.column_config.TextColumn("Vehículo"),
            "concesionario": st.column_config.TextColumn("Concesionario"),
            "campana": st.column_config.TextColumn("Campaña"),
            "nombre_formulario": st.column_config.TextColumn("Formulario"),
            "origen": st.column_config.TextColumn("Canal"),
            "real": st.column_config.TextColumn("Real", width="small"),
        },
    )

    # Botón de descarga
    csv = results.to_csv(index=False, encoding="utf-8")
    st.download_button(
        "⬇️ Descargar CSV completo",
        data=csv,
        file_name="predicciones_leads.csv",
        mime="text/csv",
    )

    return df_filtered
