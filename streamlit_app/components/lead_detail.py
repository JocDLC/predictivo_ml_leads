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
    """Renderiza gráfico de barras horizontal con impactos SHAP estilo UI moderna."""
    df_plot = explanation_df.sort_values("impacto_abs", ascending=True).tail(6)

    # CSS inline para garantizar renderizado correcto independientemente del tema
    html = """
<style>
.shap-container { margin-top: 15px; }
.shap-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    font-size: 0.85rem;
}
.shap-feature-name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 60%;
}
.shap-feature-val { font-weight: 600; }
.shap-val-hot  { color: #ef4444; }
.shap-val-cold { color: #3b82f6; }
.shap-bar-bg {
    width: 100%;
    background-color: rgba(150,150,150,0.25);
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
<div class="shap-container">
"""

    for _, row in df_plot.iterrows():
        feature  = row["feature"]
        impacto  = row["impacto"]

        is_hot    = impacto > 0
        val_class = "shap-val-hot" if is_hot else "shap-val-cold"
        sign      = "+" if is_hot else ""

        max_abs = df_plot["impacto_abs"].max() or 1
        width_pct = (abs(impacto) / max_abs) * 50  # 50% max por lado

        if is_hot:
            bar_style = f"width: {width_pct}%; right: 50%;"
            bar_class = "shap-bar-fill-hot"
        else:
            bar_style = f"width: {width_pct}%; left: 50%;"
            bar_class = "shap-bar-fill-cold"

        html += f"""
<div class="shap-row">
    <span class="shap-feature-name" title="{feature}">{feature}</span>
    <span class="shap-feature-val {val_class}">{sign}{impacto:.2f}</span>
</div>
<div class="shap-bar-bg">
    <div class="{bar_class}" style="{bar_style}"></div>
</div>
"""

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

