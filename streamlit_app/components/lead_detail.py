"""Componente de detalle individual con explicación SHAP."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def render_lead_selector(all_results, filtered_results):
    """Permite seleccionar un lead por búsqueda de ID (todos) o dropdown (filtrados).

    Returns:
        tuple(DataFrame, int): (display_df, positional_index) — el DataFrame de origen
        y el índice posicional dentro de él.
    """
    st.markdown("### 🔍 Detalle de un lead")
    st.caption("Busca por ID (busca en **todos** los leads) o selecciona de la lista filtrada.")

    has_lead_id = "lead_id" in all_results.columns

    # --- Layout ---
    col_search, col_select = st.columns([1, 2])

    # --- Búsqueda por ID (sobre todos los resultados) ---
    search_result = None  # (display_df, positional_idx)
    with col_search:
        search_id = st.text_input(
            "🔎 Buscar por ID",
            placeholder="Pega o escribe el Lead ID...",
            key="lead_id_search",
        )
        if search_id and has_lead_id:
            matches = all_results[all_results["lead_id"].astype(str).str.contains(search_id, case=False, na=False)]
            if len(matches) == 0:
                st.warning(f"No se encontró ningún lead con '{search_id}'")
            elif len(matches) == 1:
                pos = all_results.index.get_loc(matches.index[0])
                pred = matches.iloc[0]["prediccion"]
                prob = matches.iloc[0]["probabilidad_hot"]
                st.success(f"✅ **{matches.iloc[0]['lead_id']}** — {pred} ({prob}%)")
                search_result = (all_results, pos)
            else:
                st.info(f"{len(matches)} coincidencias:")
                match_options = matches["lead_id"].tolist()
                chosen = st.selectbox(
                    "Resultados de búsqueda",
                    options=range(len(match_options)),
                    format_func=lambda i: f"{match_options[i]} — {matches.iloc[i]['prediccion']} ({matches.iloc[i]['probabilidad_hot']}%)",
                    key="search_match_select",
                )
                pos = all_results.index.get_loc(matches.index[chosen])
                search_result = (all_results, pos)
        elif search_id and not has_lead_id:
            st.warning("El archivo no contiene columna de Lead ID.")

    # --- Dropdown (sobre resultados filtrados) ---
    with col_select:
        if has_lead_id:
            options = filtered_results["lead_id"].tolist()
            label_col = "lead_id"
        else:
            options = filtered_results.index.tolist()
            label_col = "index"

        max_display = min(len(options), 1000)
        selected_idx = st.selectbox(
            f"O selecciona de la lista ({label_col})",
            options=range(max_display),
            format_func=lambda i: f"{options[i]} — {filtered_results.iloc[i]['prediccion']} ({filtered_results.iloc[i]['probabilidad_hot']}%)",
            key="lead_dropdown_select",
        )

    # Búsqueda tiene prioridad
    if search_result is not None:
        return search_result

    return (filtered_results, selected_idx)


def render_lead_detail(results, explanation_df, base_value, predicted_proba, row_idx):
    """Renderiza el detalle de un lead con explicación SHAP."""
    row = results.iloc[row_idx]

    # Info del lead
    col1, col2, col3 = st.columns(3)
    with col1:
        pred = row["prediccion"]
        st.markdown(f"**Predicción:** {pred}")
    with col2:
        st.markdown(f"**Probabilidad:** {row['probabilidad_hot']}%")
    with col3:
        if "real" in row.index and pd.notna(row.get("real")):
            st.markdown(f"**Real:** {row['real']}")

    # Contexto del lead
    context_cols = {
        "vehiculo_interes": "Vehículo",
        "concesionario": "Concesionario",
        "campana": "Campaña",
        "nombre_formulario": "Formulario",
        "origen": "Canal",
        "origen_creacion": "Origen creación",
    }
    context_data = {}
    for col, label in context_cols.items():
        if col in row.index and pd.notna(row.get(col)):
            context_data[label] = row[col]

    if context_data:
        st.markdown("**Datos del lead:**")
        cols = st.columns(min(len(context_data), 3))
        for i, (label, value) in enumerate(context_data.items()):
            with cols[i % len(cols)]:
                st.markdown(f"- **{label}:** {value}")

    st.divider()

    # Explicación SHAP
    st.markdown("#### ¿Por qué esta clasificación?")
    st.caption(
        "Cada barra muestra cuánto influyó una característica en la predicción. "
        "Las barras rojas empujan hacia Hot, las azules hacia Cold."
    )

    render_shap_chart(explanation_df)

    # Tabla de impactos
    with st.expander("📋 Ver tabla detallada de impactos"):
        st.dataframe(
            explanation_df[["feature", "valor", "impacto", "direccion"]].rename(
                columns={
                    "feature": "Característica",
                    "valor": "Valor",
                    "impacto": "Impacto SHAP",
                    "direccion": "Dirección",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_shap_chart(explanation_df):
    """Renderiza gráfico de barras horizontal con impactos SHAP."""
    df_plot = explanation_df.sort_values("impacto_abs", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.5)))

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df_plot["impacto"]]

    ax.barh(
        range(len(df_plot)),
        df_plot["impacto"].values,
        color=colors,
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"].values, fontsize=10)
    ax.set_xlabel("Impacto SHAP (→ Hot | ← Cold)", fontsize=10)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_title("Top factores de la predicción", fontsize=12, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
