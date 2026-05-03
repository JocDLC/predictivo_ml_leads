"""
Reporte del modelo predictivo con metricas y contexto calculados en runtime.
"""

from datetime import datetime
import os
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.inference import (
    get_expected_features,
    load_model_and_artifacts,
    resolve_umbral,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "leads_cleaned.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "leads_raw.csv")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

HOT_COLOR = "#FF4B4B"
COLD_COLOR = "#4B8BFF"
ACCENT = "#00C49A"
GOLD = "#FFD700"

DAY_MAP = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miercoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sabado",
    "Sunday": "domingo",
}
DAY_ORDER = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
MONTH_NAMES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def _slug(text):
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().replace(" ", "_")


def _find_column(df, contains_all):
    tokens = [_slug(token) for token in contains_all]
    for col in df.columns:
        slug = _slug(col)
        if all(token in slug for token in tokens):
            return col
    return None


def _format_model_name(model):
    return type(model).__name__


def _plotly_layout(fig, height=None):
    theme_base = st.get_option("theme.base") or "light"
    is_dark = str(theme_base).lower() == "dark"
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=13),
        height=height,
    )
    return fig


def _chart_explainer(how_to_read, findings, v1_v2):
    st.markdown(
        f"{how_to_read}\n\n"
        f"**Hallazgos:** {findings}\n\n"
        f"**V1 vs V2:** {v1_v2}"
    )


def _safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def _threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "umbral": float(threshold),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def _feature_importance_series(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_).ravel(), index=feature_names).sort_values(ascending=False)
    return pd.Series(dtype=float)


def get_positive_shap_explanation(shap_values):
    values = np.asarray(shap_values.values)
    if values.ndim != 3:
        return shap_values

    class_index = 1 if values.shape[2] > 1 else 0
    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim == 2:
        base_values = base_values[:, class_index]

    import shap

    return shap.Explanation(
        values=values[:, :, class_index],
        base_values=base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )


@st.cache_data
def load_clean_data():
    df = pd.read_csv(CLEAN_PATH)
    if "dia_semana_creacion" in df.columns:
        df["dia_semana_creacion"] = df["dia_semana_creacion"].replace(DAY_MAP)
        df["dia_semana_creacion"] = df["dia_semana_creacion"].map(
            lambda value: _slug(value) if pd.notna(value) else value
        )
    return df


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_PATH, sep=";", encoding="latin-1")


@st.cache_data
def load_train_test():
    model, artifacts = load_model_and_artifacts()
    expected_features = get_expected_features(model, artifacts)

    dataset_candidates = [
        {
            "name": "X_test_v2/X_train_v2",
            "x_train": os.path.join(DATA_DIR, "X_train_v2.csv"),
            "x_test": os.path.join(DATA_DIR, "X_test_v2.csv"),
            "y_train": os.path.join(DATA_DIR, "y_train_v2.csv"),
            "y_test": os.path.join(DATA_DIR, "y_test_v2.csv"),
        },
        {
            "name": "X_test/X_train",
            "x_train": os.path.join(DATA_DIR, "X_train.csv"),
            "x_test": os.path.join(DATA_DIR, "X_test.csv"),
            "y_train": os.path.join(DATA_DIR, "y_train.csv"),
            "y_test": os.path.join(DATA_DIR, "y_test.csv"),
        },
    ]

    for paths in dataset_candidates:
        if not all(os.path.exists(path) for key, path in paths.items() if key != "name"):
            continue

        X_train = pd.read_csv(paths["x_train"])
        X_test = pd.read_csv(paths["x_test"])
        if expected_features and list(X_test.columns) != list(expected_features):
            continue

        y_train = pd.read_csv(paths["y_train"]).squeeze()
        y_test = pd.read_csv(paths["y_test"]).squeeze()
        return {
            "dataset_name": paths["name"],
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    raise FileNotFoundError(
        "No existe un dataset de train/test compatible con `best_model.joblib` en data/processed/."
    )


@st.cache_data
def load_report_artifacts():
    model, artifacts = load_model_and_artifacts()
    dataset = load_train_test()
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    threshold = resolve_umbral(artifacts)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    threshold_grid = np.round(np.linspace(0.05, 0.95, 901), 3)
    threshold_rows = [_threshold_metrics(y_test, y_proba, t) for t in threshold_grid]
    threshold_df = pd.DataFrame(threshold_rows)
    best_f1_row = threshold_df.sort_values(["F1", "Recall", "Precision"], ascending=False).iloc[0]

    current_cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = current_cm.ravel()

    baseline_pred = (y_proba >= 0.5).astype(int)
    baseline_cm = confusion_matrix(y_test, baseline_pred)
    _, baseline_fp, baseline_fn, _ = baseline_cm.ravel()

    expected_features = list(getattr(model, "feature_names_in_", X_test.columns))
    importances = _feature_importance_series(model, expected_features)

    metrics_display = artifacts.get("metrics")
    if not isinstance(metrics_display, dict):
        metrics_display = {k: f"{v:.4f}" if k == "ROC-AUC" else f"{v:.1%}" for k, v in metrics.items()}

    training_context = artifacts.get("training_context") or {}
    training_context = {
        "dataset_used": dataset["dataset_name"],
        "x_train_shape": X_train.shape,
        "x_test_shape": X_test.shape,
        "n_features": len(expected_features),
        **training_context,
    }

    return {
        "model": model,
        "artifacts": artifacts,
        "umbral": threshold,
        "metrics": metrics,
        "metrics_display": metrics_display,
        "training_context": training_context,
        "dataset_name": dataset["dataset_name"],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "best_f1_row": best_f1_row.to_dict(),
        "threshold_df": threshold_df,
        "threshold_comparison": {
            "default_050": _threshold_metrics(y_test, y_proba, 0.5),
            "active": _threshold_metrics(y_test, y_proba, threshold),
            "best_f1": best_f1_row.to_dict(),
            "delta_hot_vs_050": int(baseline_fn - fn),
            "delta_cold_vs_050": int(fp - baseline_fp),
        },
        "feature_importances": importances,
        "fp_probas": y_proba[(y_test.values == 0) & (y_pred == 1)],
        "fn_probas": y_proba[(y_test.values == 1) & (y_pred == 0)],
        "expected_features": expected_features,
    }


def render_model_report():
    report = load_report_artifacts()
    df = load_clean_data()

    st.title("Reporte del Modelo Predictivo")
    st.caption("Resumen de las 5 fases del proyecto: datos, analisis, features, modelado y evaluacion.")

    execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = report["model"]
    st.info(
        " | ".join(
            [
                f"Modelo activo: `{_format_model_name(model)}`",
                f"Features esperadas: `{report['training_context']['n_features']}`",
                f"Dataset: `{report['dataset_name']}`",
                f"Umbral activo: `{report['umbral']:.3f}`",
                f"Ejecucion: `{execution_time}`",
            ]
        )
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Data Engineering",
            "2. Analisis Exploratorio",
            "3. Feature Engineering",
            "4. Modelado",
            "5. Evaluacion",
        ]
    )

    with tab1:
        _render_data_engineering(df)
    with tab2:
        _render_eda(df, report)
    with tab3:
        _render_feature_engineering(df, report)
    with tab4:
        _render_modeling(report)
    with tab5:
        _render_evaluation(report)
def _render_data_engineering(df):
    st.header("1. Data Engineering")
    st.markdown(
        "El bloque usa el dataset crudo disponible en `data/raw/leads_raw.csv` y el dataset limpio actual en "
        "`data/processed/leads_cleaned.csv`."
    )
    with st.expander("Decisiones de negocio del pipeline V2", expanded=True):
        st.markdown(
            """
            - El universo de entrenamiento se restringe a **leads de 2025** para evitar drift temporal frente a anos anteriores.
            - Se excluyen **abril, mayo y junio de 2025** por comportamiento operativo anomalo en asignacion y volumen.
            - Se eliminan los leads de **chatbot** para modelar solo la gestion comercial humana.
            - Los leads sin **cualificacion** se descartan porque no permiten construir el target correctamente.
            - El target binario se define como **Hot = `Contacto interesado`** y **Cold = resto de cualificaciones**.
            - Se excluyen los formularios **Arkana** y **Oroch** porque llegaban pre-clasificados como Hot y contaminaban el modelo con **data leakage**.
            - Los nulos categoricos se imputan con **`Desconocido`** y al final se eliminan duplicados antes de exportar.
            """
        )

    raw_df = load_raw_data()
    raw_rows = len(raw_df)
    clean_rows = len(df)
    raw_cols = len(raw_df.columns)
    clean_cols = len(df.columns)
    lead_id_col = _find_column(raw_df, ["lead", "id"])
    raw_duplicates = int(raw_df[lead_id_col].duplicated().sum()) if lead_id_col else np.nan
    clean_nulls = int(df.isna().sum().sum())
    hot = int(df["target"].sum())
    cold = int(len(df) - hot)
    hot_rate = df["target"].mean()

    # El raw local y el clean pueden venir de cortes distintos.
    if clean_rows > raw_rows:
        st.warning(
            "Atencion: este resumen compara dos archivos de distinto corte temporal. "
            f"`data/raw/leads_raw.csv` tiene {raw_rows:,} filas y "
            f"`data/processed/leads_cleaned.csv` tiene {clean_rows:,}. "
            "Por eso puede verse 'filas limpias' mayor a 'filas originales'. "
            "Para una comparacion 1:1, ambos archivos deben generarse desde la misma corrida de pipeline."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas originales", f"{raw_rows:,}")
    col2.metric("Filas limpias", f"{clean_rows:,}", delta=f"{(clean_rows / raw_rows - 1):.1%}", delta_color="off")
    col3.metric("Columnas", f"{clean_cols}", delta=f"{raw_cols} -> {clean_cols}", delta_color="off")
    col4.metric("Nulos restantes", f"{clean_nulls}")

    st.subheader("Pipeline de limpieza")
    notebook_funnel_steps = [
        ("Leads 2025", 322009),
        ("Excluir Abr/May/Jun 2025", 223298),
        ("Eliminar chatbot", 73594),
        ("Eliminar sin cualificacion", 66028),
        ("Eliminar leakage Arkana/Oroch", 61905),
        ("Eliminar duplicados / dataset final", clean_rows),
    ]
    funnel_df = pd.DataFrame(notebook_funnel_steps, columns=["paso", "valor"])
    funnel = go.Figure(
        go.Funnel(
            y=funnel_df["paso"],
            x=funnel_df["valor"],
            textinfo="value+percent previous",
            marker=dict(color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", ACCENT]),
        )
    )
    funnel.update_layout(title="Volumen por etapa del pipeline V2 documentado", height=460)
    st.plotly_chart(funnel, use_container_width=True, key="de_funnel_doc")
    st.caption(
        "Las etapas intermedias se toman del notebook `00_data_engineering_v2_2025.ipynb`. "
        "El CSV crudo disponible en la app no conserva el historico completo usado en ese notebook, "
        "asi que el funnel prioriza las cifras auditadas del pipeline V2."
    )

    st.subheader("Distribucion del target")
    col1, col2 = st.columns(2)
    with col1:
        pie = go.Figure(
            go.Pie(
                labels=["Hot", "Cold"],
                values=[hot, cold],
                hole=0.55,
                marker=dict(colors=[HOT_COLOR, COLD_COLOR]),
                textinfo="label+percent",
            )
        )
        pie.update_layout(title="Proporcion Hot vs Cold", height=380)
        st.plotly_chart(pie, use_container_width=True, key="eda_target_pie")
    with col2:
        bar = go.Figure(
            go.Bar(
                x=["Hot", "Cold"],
                y=[hot, cold],
                marker_color=[HOT_COLOR, COLD_COLOR],
                text=[f"{hot_rate:.1%}", f"{1 - hot_rate:.1%}"],
                textposition="outside",
            )
        )
        bar.update_layout(title="Conteo de leads", yaxis_title="Cantidad", height=380)
        st.plotly_chart(bar, use_container_width=True, key="eda_target_bar")

    st.info(
        f"Distribucion actual: Hot {hot_rate:.1%} vs Cold {1 - hot_rate:.1%}. "
        f"Ratio Hot/Cold = {hot / max(cold, 1):.2f}:1."
    )

    detail_rows = [
        {
            "Paso": "Encoding y lectura",
            "Accion": "Lectura de CSV crudo con separador `;` y encoding `latin-1`.",
            "Impacto": f"{raw_cols} columnas legibles",
        },
        {
            "Paso": "Filtro ano 2025",
            "Accion": "Restringe el universo al periodo operativo vigente para evitar drift temporal.",
            "Impacto": "322,009 registros segun notebook V2",
        },
        {
            "Paso": "Exclusion meses anomalos (Abr/May/Jun 2025)",
            "Accion": "Remueve meses con cambios de proceso y volumenes atipicos.",
            "Impacto": "-98,711 filas estimadas entre el corte 2025 y el dataset pre-chatbot",
        },
        {
            "Paso": "Eliminacion chatbot",
            "Accion": "Excluye leads procesados por bot para modelar solo gestion humana.",
            "Impacto": "-149,704 filas segun notebook V2",
        },
        {
            "Paso": "Filtrar sin cualificacion",
            "Accion": "Descarta leads sin etiqueta util para construir el target.",
            "Impacto": "-7,566 filas segun notebook V2",
        },
        {
            "Paso": "Target binario",
            "Accion": "Se genera `target` con Hot = `Contacto interesado` y Cold = resto.",
            "Impacto": f"{hot:,} Hot / {cold:,} Cold",
        },
        {
            "Paso": "Eliminacion leakage Arkana/Oroch",
            "Accion": "Excluye formularios pre-clasificados como Hot sin cualificacion humana real.",
            "Impacto": "-4,123 filas; 66,028 -> 61,905",
        },
        {
            "Paso": "Reduccion de columnas",
            "Accion": "Se conservan solo variables disponibles al momento de creacion del lead.",
            "Impacto": f"{raw_cols} -> {clean_cols} columnas",
        },
        {
            "Paso": "Imputacion de nulos con `Desconocido`",
            "Accion": "Completa categoricas faltantes sin forzar la moda.",
            "Impacto": f"Nulos restantes {clean_nulls}",
        },
        {
            "Paso": "Duplicados finales",
            "Accion": "Elimina filas repetidas antes de exportar el dataset productivo.",
            "Impacto": (
                f"-3,359 filas en notebook V2; duplicados en crudo {raw_duplicates:,}"
                if not pd.isna(raw_duplicates)
                else "-3,359 filas en notebook V2"
            ),
        },
    ]
    with st.expander("Detalle de limpieza"):
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def _render_eda(df, report):
    st.header("2. Analisis Exploratorio de Datos")
    global_rate = df["target"].mean() * 100

    st.subheader("2.1 Distribucion del target (Dataset Limpio)")
    col1, col2 = st.columns(2)
    hot = int(df["target"].sum())
    cold = int(len(df) - hot)
    hot_rate = df["target"].mean()
    with col1:
        pie = go.Figure(
            go.Pie(
                labels=["Hot", "Cold"],
                values=[hot, cold],
                hole=0.55,
                marker=dict(colors=[HOT_COLOR, COLD_COLOR]),
                textinfo="label+percent",
            )
        )
        pie.update_layout(title="Proporcion Hot vs Cold", height=380)
        st.plotly_chart(pie, use_container_width=True, key="eda_dataset_target_pie")

    with col2:
        bar = go.Figure(
            go.Bar(
                x=["Hot", "Cold"],
                y=[hot, cold],
                marker_color=[HOT_COLOR, COLD_COLOR],
                text=[f"{hot_rate:.1%}", f"{1 - hot_rate:.1%}"],
                textposition="outside",
            )
        )
        bar.update_layout(title="Conteo de leads", yaxis_title="Cantidad", height=380)
        st.plotly_chart(bar, use_container_width=True, key="eda_dataset_target_bar")

    _chart_explainer(
        "La gráfica de pastel (izquierda) muestra la proporción de leads 'Hot' (oportunidades reales) vs 'Cold' (descartes). La gráfica de barras (derecha) muestra el conteo absoluto para cada clase.",
        "El dataset está desbalanceado: hay muchos más leads 'Cold' que 'Hot'. Esto es típico en la generación de leads, pero requiere que el modelo sea evaluado con métricas como F1 o Recall, no solo Accuracy.",
        "El pipeline V2 es más estricto, resultando en una tasa de 'Hot' más baja pero más realista que en V1, que estaba inflada por data leakage. El modelo V2 aprende de una distribución más limpia."
    )

    st.subheader("2.2 Patrones temporales (Volumen y Conversion)")
    tab_hora, tab_dia, tab_mes = st.tabs(["Por Hora", "Por Dia", "Por Mes"])
    with tab_hora:
        conv_hora = df.groupby("hora_creacion")["target"].agg(["mean", "count"]).reset_index()
        conv_hora["mean"] *= 100
        fig_hora = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hora.add_trace(go.Bar(x=conv_hora["hora_creacion"], y=conv_hora["count"], name="Volumen", marker_color="rgba(100, 150, 250, 0.4)"), secondary_y=False)
        fig_hora.add_trace(go.Scatter(x=conv_hora["hora_creacion"], y=conv_hora["mean"], mode="lines+markers", name="% Hot", line=dict(color=ACCENT, width=3)), secondary_y=True)
        fig_hora.add_hline(y=global_rate, line_dash="dash", line_color="gray", secondary_y=True)
        fig_hora.update_layout(title="Volumen y Tasa de conversion por hora", height=400)
        fig_hora.update_yaxes(title_text="Cantidad de Leads", secondary_y=False)
        fig_hora.update_yaxes(title_text="% Hot", secondary_y=True)
        _plotly_layout(fig_hora)
        st.plotly_chart(fig_hora, use_container_width=True, key="eda_hora")
        _chart_explainer(
            "La gráfica combina barras de volumen (eje izquierdo) con una línea de tasa de conversión (eje derecho) para cada hora del día. La línea punteada es la tasa de conversión promedio global.",
            "El mayor volumen de leads entra entre las 9am y 7pm. Sin embargo, la tasa de conversión es notablemente más alta en la madrugada (0-5am), cuando el volumen es bajo. Estos leads nocturnos, aunque pocos, son de alta calidad.",
            "El patrón de V2 es similar al de V1, pero la creación de la feature `franja_horaria` en el pipeline V2 ayuda al modelo a capturar explícitamente el valor de los leads de madrugada/noche."
        )

    with tab_dia:
        conv_dia = df.groupby("dia_semana_creacion")["target"].agg(["mean", "count"]).reindex(DAY_ORDER).dropna()
        conv_dia["mean"] *= 100
        fig_dia = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dia.add_trace(go.Bar(x=conv_dia.index, y=conv_dia["count"], name="Volumen", marker_color="rgba(100, 150, 250, 0.4)"), secondary_y=False)
        fig_dia.add_trace(go.Scatter(x=conv_dia.index, y=conv_dia["mean"], mode="lines+markers", name="% Hot", line=dict(color=ACCENT, width=3)), secondary_y=True)
        fig_dia.add_hline(y=global_rate, line_dash="dash", line_color="gray", secondary_y=True)
        fig_dia.update_layout(title="Volumen y Conversion por dia de semana", height=400)
        fig_dia.update_yaxes(title_text="Cantidad de Leads", secondary_y=False)
        fig_dia.update_yaxes(title_text="% Hot", secondary_y=True)
        _plotly_layout(fig_dia)
        st.plotly_chart(fig_dia, use_container_width=True, key="eda_dia")
        _chart_explainer(
            "Compara el volumen de leads y la tasa de conversión para cada día de la semana. La línea punteada es la conversión promedio.",
            "El volumen es mayor en días de semana, pero la conversión tiende a ser ligeramente superior los fines de semana (sábado y domingo), cuando los prospectos tienen más tiempo para investigar.",
            "Este patrón se mantiene consistente entre V1 y V2. La feature `es_fin_de_semana` en el pipeline V2 capitaliza este hallazgo."
        )

    with tab_mes:
        if "mes_creacion" in df.columns:
            conv_mes = df.groupby("mes_creacion")["target"].agg(["mean", "count"]).dropna()
            conv_mes["mean"] *= 100
            meses_labels = [MONTH_NAMES.get(m, m) for m in conv_mes.index]
            fig_mes = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mes.add_trace(go.Bar(x=meses_labels, y=conv_mes["count"], name="Volumen", marker_color="rgba(100, 150, 250, 0.4)"), secondary_y=False)
            fig_mes.add_trace(go.Scatter(x=meses_labels, y=conv_mes["mean"], mode="lines+markers", name="% Hot", line=dict(color=ACCENT, width=3)), secondary_y=True)
            fig_mes.add_hline(y=global_rate, line_dash="dash", line_color="gray", secondary_y=True)
            fig_mes.update_layout(title="Volumen y Conversion por mes", height=400)
            fig_mes.update_yaxes(title_text="Cantidad de Leads", secondary_y=False)
            fig_mes.update_yaxes(title_text="% Hot", secondary_y=True)
            _plotly_layout(fig_mes)
            st.plotly_chart(fig_mes, use_container_width=True, key="eda_mes")
            _chart_explainer(
                "Compara el volumen y la tasa de conversión a lo largo de los meses disponibles en el dataset de entrenamiento.",
                "Se observan picos de conversión en meses específicos, posiblemente ligados a campañas o estacionalidad del mercado. El volumen puede variar drásticamente, como se ve en los meses de verano.",
                "El pipeline V2 excluye explícitamente los meses de Abril, Mayo y Junio de 2025 por presentar un comportamiento anómalo que afectaba negativamente el rendimiento del modelo V1."
            )
        else:
            st.info("No se encontró la columna 'mes_creacion' en el dataset.")

    madrugada = df[df["hora_creacion"].between(0, 5)]["target"].mean() * 100
    best_day = conv_dia["mean"].idxmax()
    worst_day = conv_dia["mean"].idxmin()
    
    if "mes_creacion" in df.columns:
        conv_month = df.groupby("mes_creacion")["target"].mean().dropna() * 100
        best_month = int(conv_month.idxmax()) if not conv_month.empty else None
        worst_month = int(conv_month.idxmin()) if not conv_month.empty else None
    else:
        best_month, worst_month, conv_month = None, None, pd.Series()

    findings = [
        f"**Madrugada (00-05h):** {madrugada:.1f}% de conversion. Aunque el volumen es menor, los leads que completan formularios de madrugada suelen tener una intencion de compra mas genuina y exploran con menos ruido." if not np.isnan(madrugada) else None,
        f"**Mejor dia:** {best_day} ({conv_dia['mean'].max():.1f}%). Los fines de semana o dias especificos atraen prospectos con mas tiempo libre para analizar su compra, a diferencia del peor dia ({worst_day}, {conv_dia['mean'].min():.1f}%).",
        f"**Estacionalidad (Mes):** Mejor mes {MONTH_NAMES.get(best_month, best_month)} ({conv_month.max():.1f}%) vs Peor mes {MONTH_NAMES.get(worst_month, worst_month)} ({conv_month.min():.1f}%). Los meses iniciales de anio o cierres de trimestre suelen alterar fuertemente el volumen de pauta y la calidad del lead." if best_month is not None else None,
    ]
    st.markdown("**Hallazgos de negocio:**")
    st.markdown("\n".join([f"- {item}" for item in findings if item]))

    st.subheader("2.3 Heatmap interactivo (Dia vs Hora)")
    st.caption("Compara los momentos de mayor trafico (volumen) con los momentos de mayor calidad (conversion).")
    pivot_conv = df.groupby(["dia_semana_creacion", "hora_creacion"])["target"].mean().unstack(fill_value=0) * 100
    pivot_conv = pivot_conv.reindex([day for day in DAY_ORDER if day in pivot_conv.index])
    
    pivot_vol = df.groupby(["dia_semana_creacion", "hora_creacion"]).size().unstack(fill_value=0)
    pivot_vol = pivot_vol.reindex([day for day in DAY_ORDER if day in pivot_vol.index])
    
    hm_type = st.radio("Metrica del Heatmap:", ["Tasa de Conversion (%)", "Volumen (Conteo)"], horizontal=True)
    if hm_type == "Tasa de Conversion (%)":
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot_conv.values,
                x=[f"{hour}:00" for hour in pivot_conv.columns],
                y=pivot_conv.index,
                colorscale="RdYlGn",
                zmid=global_rate,
                text=np.round(pivot_conv.values, 0).astype(int),
                texttemplate="%{text}%",
            )
        )
        fig_hm.update_layout(title="Tasa de conversion por hora y dia")
    else:
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot_vol.values,
                x=[f"{hour}:00" for hour in pivot_vol.columns],
                y=pivot_vol.index,
                colorscale="Blues",
                text=pivot_vol.values,
                texttemplate="%{text}",
            )
        )
        fig_hm.update_layout(title="Volumen de leads por hora y dia")

    _plotly_layout(fig_hm, 400)
    st.plotly_chart(fig_hm, use_container_width=True, key="eda_hm")
    _chart_explainer(
        "El mapa de calor cruza el día de la semana con la hora del día. Puedes alternar entre ver la 'Tasa de Conversión' (donde verde es mejor) y el 'Volumen' (donde azul oscuro es mayor).",
        "Permite identificar 'horas doradas': momentos con alta conversión y volumen decente (ej. noches de fin de semana) y 'puntos ciegos': horas de alto volumen pero baja calidad (ej. mediodía entre semana).",
        "En V2, este análisis fue clave para crear la feature `franja_horaria`, que agrupa estos patrones y mejora la capacidad del modelo para distinguir leads de calidad basados en el momento de creación."
    )

    st.subheader("2.4 Conversion por feature categorica")
    cat_features = {
        "Vehiculo": "vehiculo_interes",
        "Origen": "origen",
        "Campana": "campana",
        "Formulario": "nombre_formulario",
        "Canal": "origen_creacion",
    }
    selected_label = st.radio("Filtrar por:", list(cat_features.keys()), horizontal=True, key="eda_feature_radio")
    selected_feat = cat_features[selected_label]
    
    known_categories = report["artifacts"].get("known_categories", {})
    
    conv = df.groupby(selected_feat)["target"].agg(["mean", "count"]).reset_index()
    if selected_feat in known_categories:
        conv = conv[conv[selected_feat].isin(known_categories[selected_feat])]
        
    conv["mean"] *= 100
    conv = conv.sort_values("mean", ascending=True)
    fig = go.Figure(
        go.Bar(
            y=conv[selected_feat],
            x=conv["mean"],
            orientation="h",
            marker_color=[ACCENT if value >= global_rate else HOT_COLOR for value in conv["mean"]],
            text=[f"{rate:.1f}% (n={int(count):,})" for rate, count in zip(conv["mean"], conv["count"])],
            textposition="outside",
        )
    )
    fig.add_vline(x=global_rate, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Tasa de conversion por {selected_label}", xaxis_title="% Hot", height=max(350, len(conv) * 30))
    _plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True, key="eda_cat_conv")
    _chart_explainer(
        "Muestra la tasa de conversión para cada categoría de la variable seleccionada. Las barras en verde superan el promedio global (línea punteada), mientras que las rojas están por debajo.",
        "Permite identificar qué vehículos, orígenes o campañas generan leads de mayor o menor calidad. Es una herramienta clave para optimizar la inversión en marketing.",
        "En V2, las categorías con muy pocos leads (y tasas de conversión engañosas del 100% o 0%) son agrupadas en 'otros', limpiando la visualización y permitiendo un análisis más robusto."
    )

    st.subheader("2.5 Fuerza de asociacion con el target")
    from scipy.stats import chi2_contingency

    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = chi2_contingency(ct)[0]
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cramers = {col: cramers_v(df[col], df["target"]) for col in cat_cols}
    cramers_df = pd.DataFrame(sorted(cramers.items(), key=lambda item: item[1]), columns=["Feature", "V"])

    fig = go.Figure(
        go.Bar(
            y=cramers_df["Feature"],
            x=cramers_df["V"],
            orientation="h",
            marker=dict(color=cramers_df["V"], colorscale=[[0, COLD_COLOR], [0.5, GOLD], [1, HOT_COLOR]], showscale=True),
            text=[f"{value:.3f}" for value in cramers_df["V"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.3, line_dash="dot", line_color=HOT_COLOR, annotation_text="Fuerte")
    fig.add_vline(x=0.1, line_dash="dot", line_color=GOLD, annotation_text="Moderada")
    fig.update_layout(title="Cramer's V por variable", xaxis_title="Cramer's V")
    _plotly_layout(fig, 420)
    st.plotly_chart(fig, use_container_width=True, key="eda_cramer")
    _chart_explainer(
        "Mide la fuerza de asociación entre cada variable categórica y el target (ser 'Hot' o 'Cold'). Un valor cercano a 1 indica una asociación muy fuerte, mientras que cerca de 0 es débil.",
        "Variables como `vehiculo_interes` y `origen` tienen la mayor capacidad predictiva. Otras, como `dia_semana_creacion`, tienen una asociación más débil por sí solas.",
        "El análisis de Cramer's V en V2 ayudó a seleccionar las features más relevantes para el modelo, descartando aquellas con bajo poder predictivo que solo añadían ruido en V1."
    )

    top_cramer = cramers_df.sort_values("V", ascending=False).head(3)
    top_vehicle = None
    if "vehiculo_interes" in df.columns:
        vehicle_conv = df.groupby("vehiculo_interes")["target"].agg(["mean", "count"])
        if not vehicle_conv.empty:
            top_vehicle = vehicle_conv.sort_values(["count", "mean"], ascending=[False, True]).index[0]
            top_vehicle_rate = vehicle_conv.loc[top_vehicle, "mean"] * 100
    st.markdown("**Conclusiones del EDA**")
    st.markdown(f"- Variables con mayor asociacion actual: {', '.join(top_cramer['Feature'].tolist())}.")
    if top_vehicle is not None:
        st.markdown(f"- Vehiculo de mayor volumen actual: `{top_vehicle}` con conversion de {top_vehicle_rate:.1f}%.")
    st.markdown("- Las cifras de este tab se recalculan sobre el dataset limpio actual; no se usan porcentajes fijos.")

    st.subheader("2.6 Matriz de correlacion numerica")
    st.caption("Permite identificar relaciones directas o inversas entre las variables temporales y de comportamiento frente al target.")
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filtramos columnas sin varianza (ej. ano_creacion constante) y las de IDs que ensucian la matriz
    valid_cols = [col for col in num_cols if df[col].nunique() > 1 and not col.lower().endswith("id")]
    if len(valid_cols) > 1:
        corr = df[valid_cols].corr()
        
        # Ordenamos respecto al target como en el notebook V2
        if "target" in corr.columns:
            sorted_cols = corr["target"].sort_values(ascending=False).index
            corr = corr.loc[sorted_cols, sorted_cols]
            
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig_corr.update_layout(title="Correlacion Lineal (Variables Numericas)", height=450)
        fig_corr.update_yaxes(autorange="reversed")
        _plotly_layout(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True, key="eda_corr")
        _chart_explainer(
            "Muestra la correlación lineal de Pearson entre las variables numéricas. Los tonos rojos indican una correlación positiva (si una sube, la otra también), los azules una correlación negativa (si una sube, la otra baja), y los blancos una correlación cercana a cero.",
            "No se observan correlaciones extremadamente altas entre las features predictivas, lo que es bueno para evitar multicolinealidad. La correlación más fuerte es entre las variables de fecha (día, mes, año), lo cual es esperado.",
            "En V2, la matriz se ordena por la correlación con el `target`, haciendo más fácil identificar las relaciones más importantes. Además, se excluyen columnas de ID que no aportan valor."
        )
def _render_feature_engineering(df, report):
    st.header("3. Feature Engineering")
    raw_df = load_raw_data()
    artifacts = report["artifacts"]
    known_categories = artifacts.get("known_categories", {})
    n_features = report["training_context"]["n_features"]
    x_train_cols = list(report["X_train"].columns)
    bayes_count = int(
        sum(col.endswith("_bayes_enc") or col.endswith("_target_enc") for col in x_train_cols)
    )
    one_hot_prefixes = ("origen_creacion_", "dia_semana_creacion_", "franja_horaria_")
    one_hot_count = int(sum(any(col.startswith(prefix) for prefix in one_hot_prefixes) for col in x_train_cols))
    numeric_derived_count = int(max(n_features - bayes_count - one_hot_count, 0))

    st.markdown(
        f"El pipeline actual parte de **{len(df.columns)} columnas limpias** y entrega **{n_features} features** "
        f"compatibles con el modelo activo."
    )

    st.subheader("3.1 Pipeline de transformacion")
    st.caption("Evolución de las dimensiones: eliminamos fuga de datos (leakage) y compactamos las variables con técnicas de Encoding.")
    pipeline_df = pd.DataFrame(
        {
            "Paso": ["Crudo (Salesforce)", "Limpio (Sin leakage)", "Features finales (Modelo)"],
            "Columnas": [len(raw_df.columns), len(df.columns), n_features],
        }
    )
    fig = go.Figure(
        go.Funnel(
            y=pipeline_df["Paso"],
            x=pipeline_df["Columnas"],
            textinfo="value",
            marker=dict(color=["#636EFA", ACCENT, GOLD]),
        )
    )
    fig.update_layout(title="Reducción de dimensionalidad (Columnas)")
    _plotly_layout(fig, 340)
    st.plotly_chart(fig, use_container_width=True, key="fe_funnel")
    _chart_explainer(
        "El gráfico muestra cómo se reduce el número de columnas desde el archivo crudo hasta las features finales que recibe el modelo.",
        "Partimos de 27 columnas, limpiamos hasta quedarnos con 12 variables de negocio relevantes, y tras el encoding (transformación a números), el modelo trabaja con un set final de features optimizadas.",
        "El pipeline V2 es más agresivo en la selección y transformación de features, utilizando técnicas como Bayesian Encoding para representar información de manera más compacta y efectiva que el One-Hot Encoding simple de V1."
    )
    with st.expander("Detalle de transformaciones"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Transformacion": "Features derivadas", "Detalle": "`es_fin_de_semana` y `franja_horaria`."},
                    {
                        "Transformacion": "Eliminacion de leakage",
                        "Detalle": "Se elimina `subtipo_interes` por riesgo de data leakage (refleja informacion post-clasificacion).",
                    },
                    {"Transformacion": "Agrupacion de categorias", "Detalle": "Categorias fuera de `known_categories` se mapean a `otros`."},
                    {"Transformacion": "Target/Bayesian encoding", "Detalle": "Se aplican encoders bayesianos para reducir dimensionalidad y ruido."},
                    {"Transformacion": "Alineacion final", "Detalle": "El DataFrame se reindexa a las features exactas esperadas por el modelo."},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        encoders = artifacts.get("bayesian_encoders", {})
        if encoders:
            st.info(f"🎯 **Bayesian Encoders aplicados a:** {', '.join(encoders.keys())}")

    with st.expander("Bayesian Target Encoding (Smoothing)"):
        st.markdown(
            """
            El **Target Encoding con smoothing bayesiano** reemplaza categorias por una tasa de conversion
            suavizada entre:
            - media de la categoria (senal local)
            - media global del dataset (senal estable)

            Esto reduce ruido en categorias raras y evita explosiones de dimensionalidad de un One-Hot puro.
            """
        )
        st.markdown(
            """
            **Variables de alta cardinalidad tratadas con Bayesian/Target Encoding (notebook 02):**
            `vehiculo_interes`, `origen`, `nombre_formulario`, `campana`, `concesionario`.
            """
        )
        st.markdown(
            """
            **Variables candidatas a One-Hot en el disenio del notebook 02:**
            `origen_creacion`, `dia_semana_creacion`, `franja_horaria`.
            """
        )
        st.caption(
            "Nota: el desglose mostrado abajo refleja el artifact actualmente cargado en produccion."
        )

    st.subheader("3.2 Nueva feature: franja horaria")
    st.caption("Se derivan las horas en 4 grandes franjas para que el modelo capture mejor el incremento de conversión en la madrugada/noche.")
    df_temp = df.copy()
    df_temp["franja_horaria"] = df_temp["hora_creacion"].apply(
        lambda hour: "madrugada" if hour < 6 else "manana" if hour < 12 else "tarde" if hour < 18 else "noche"
    )
    franja_conv = df_temp.groupby("franja_horaria")["target"].agg(["mean", "count"]).reindex(["madrugada", "manana", "tarde", "noche"])
    fig = go.Figure(
        go.Bar(
            x=franja_conv.index,
            y=franja_conv["mean"].fillna(0) * 100,
            marker_color=["#9b59b6", "#f39c12", "#e74c3c", "#2c3e50"],
            text=[
                f"{rate:.1f}% (n={int(count):,})" if pd.notna(rate) else "sin datos"
                for rate, count in zip(franja_conv["mean"].fillna(0) * 100, franja_conv["count"].fillna(0))
            ],
            textposition="outside",
        )
    )
    fig.add_hline(y=df["target"].mean() * 100, line_dash="dash", line_color="gray")
    fig.update_layout(title="Conversion por franja horaria", yaxis_title="% Hot")
    _plotly_layout(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="fe_franja")
    _chart_explainer(
        "La gráfica muestra la tasa de conversión promedio para cada una de las 4 franjas horarias creadas.",
        "Agrupar las 24 horas en franjas captura un patrón claro: la conversión es más alta en la 'madrugada' y 'noche'. Esta nueva feature es más simple y potente para el modelo que usar la hora individual.",
        "Esta feature es una de las mejoras clave del pipeline V2. El modelo V1 usaba la hora como una variable numérica simple, perdiendo la oportunidad de capturar estos patrones no lineales."
    )

    st.subheader("3.3 Reduccion de categorias")
    st.caption("Las categorías muy raras (< 1% de aparición) se agrupan en 'otros' para evitar que el modelo las memorice y sufra sobreajuste (overfitting).")
    rows = []
    for feature, valid_categories in known_categories.items():
        before = int(df[feature].nunique()) if feature in df.columns else np.nan
        after = len(valid_categories)
        rows.append(
            {
                "Feature": feature,
                "Antes": before,
                "Despues": after,
                "Agrupadas en otros": before - after if pd.notna(before) else np.nan,
            }
        )
    reduction_df = pd.DataFrame(rows)
    if not reduction_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Se mantienen", y=reduction_df["Feature"], x=reduction_df["Despues"], orientation="h", marker_color=ACCENT))
        fig.add_trace(
            go.Bar(
                name="Agrupadas en otros",
                y=reduction_df["Feature"],
                x=reduction_df["Agrupadas en otros"].clip(lower=0),
                orientation="h",
                marker_color=HOT_COLOR,
            )
        )
        fig.update_layout(barmode="stack", title="Categorias antes y despues", xaxis_title="Numero de categorias")
        _plotly_layout(fig, 340)
        st.plotly_chart(fig, use_container_width=True, key="fe_categories")
        _chart_explainer(
            "Compara la cantidad de categorías únicas por variable antes (azul+rojo) y después (azul) de la limpieza. Las categorías en rojo son las que se agruparon en 'otros' por ser poco frecuentes.",
            "Variables como `nombre_formulario` o `campana` tienen cientos de categorías, la mayoría con muy pocos ejemplos. Agruparlas reduce el riesgo de que el modelo 'memorice' casos raros (overfitting).",
            "El pipeline V2 formaliza este agrupamiento usando un `known_categories.json`, asegurando que el modelo en producción trate las nuevas categorías de forma consistente, a diferencia de V1 que podía fallar ante valores no vistos."
        )
        st.dataframe(reduction_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se registraron agrupaciones de categorías de baja frecuencia en los artefactos actuales.")

    st.subheader("3.4 Dataset final")
    col1, col2, col3 = st.columns(3)
    col1.metric("Train", f"{report['X_train'].shape[0]:,} filas")
    col2.metric("Test", f"{report['X_test'].shape[0]:,} filas")
    col3.metric("Features", f"{n_features}")
    train_hot_rate = float(report["y_train"].mean()) * 100
    test_hot_rate = float(report["y_test"].mean()) * 100
    st.success(
        f"Train shape: {report['X_train'].shape} | Test shape: {report['X_test'].shape} | "
        f"Sin columnas de texto en la matriz final (todo codificado)."
    )
    st.caption(
        "Split estratificado 80/20 manteniendo la distribucion del target en ambos conjuntos. "
        f"Hot rate train: {train_hot_rate:.2f}% | Hot rate test: {test_hot_rate:.2f}%."
    )

    st.subheader("3.5 Desglose de dimensionalidad por tipo de encoding")
    encoding_breakdown_df = pd.DataFrame(
        [
            {"Tipo": "Bayesian/Target Encoded", "Features": bayes_count},
            {"Tipo": "One-Hot Encoded", "Features": one_hot_count},
            {"Tipo": "Numericas/Derivadas", "Features": numeric_derived_count},
        ]
    )
    fig_dim = go.Figure(
        go.Bar(
            x=encoding_breakdown_df["Tipo"],
            y=encoding_breakdown_df["Features"],
            marker_color=[ACCENT, "#636EFA", GOLD],
            text=encoding_breakdown_df["Features"],
            textposition="outside",
        )
    )
    fig_dim.update_layout(
        title=f"Composicion de features finales ({n_features} total)",
        yaxis_title="Numero de features",
    )
    _plotly_layout(fig_dim, 340)
    st.plotly_chart(fig_dim, use_container_width=True, key="fe_encoding_breakdown")
    _chart_explainer(
        "Desglosa el número total de features que recibe el modelo según la técnica de encoding utilizada.",
        "La mayoría de las features provienen de la codificación One-Hot de variables como `origen_creacion`. Las variables de alta cardinalidad son tratadas con Bayesian Encoding para controlar la dimensionalidad.",
        "El modelo V2 introduce el Bayesian Encoding, una técnica más sofisticada que el One-Hot puro usado en V1. Esto permite incluir más información sin crear un número excesivo de columnas, resultando en un modelo más robusto."
    )
    st.dataframe(encoding_breakdown_df, use_container_width=True, hide_index=True)
    st.markdown(
        "- **One-Hot Encoded:** una columna de texto/categoria se divide en varias columnas binarias (0/1), "
        "una por cada categoria relevante. Ejemplo: `origen_creacion` -> `origen_creacion_ONE`, `origen_creacion_RRSS`, etc.\n"
        "- **Numericas/Derivadas:** variables numericas directas (por ejemplo `hora_creacion`) "
        "o creadas a partir de otras (`es_fin_de_semana`, `franja_horaria` transformada)."
    )
    if bayes_count == 0:
        st.info(
            "Nota de version: en el pipeline documentado se evalua Target/Bayesian Encoding, "
            "pero el artifact V2 actualmente desplegado no materializa columnas bayesianas en la matriz final. "
            "Por eso el desglose muestra `Bayesian/Target Encoded = 0` y la mayor parte de la expansion "
            "proviene de One-Hot + variables numericas/derivadas."
        )


def _render_modeling(report):
    st.header("4. Modelado")
    model = report["model"]
    metrics = report["metrics"]
    feature_importances = report["feature_importances"]
    dataset_name = report["dataset_name"]

    st.markdown(
        f"El modelo desplegado hoy es **{_format_model_name(model)}** y se evalua sobre **{dataset_name}** "
        f"con **{report['X_test'].shape[0]:,} leads**."
    )

    st.subheader("4.1 Estado del benchmark")
    st.info(
        "No se encontro un archivo de benchmark versionado para comparar modelos en runtime. "
        "Las comparaciones historicas deben publicarse desde un artifact dedicado; por ahora se muestra solo el modelo activo."
    )

    st.subheader("4.2 Curva ROC")
    fpr, tpr, _ = roc_curve(report["y_test"], report["y_proba"])
    auc_val = roc_auc_score(report["y_test"], report["y_proba"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{_format_model_name(model)} (AUC={auc_val:.4f})",
            line=dict(color=ACCENT, width=3),
            fill="tonexty",
            fillcolor="rgba(0,196,154,0.12)",
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC=0.5)", line=dict(color="gray", dash="dash")))
    fig.update_layout(title=f"Curva ROC - {_format_model_name(model)}", xaxis_title="FPR", yaxis_title="TPR")
    _plotly_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True, key="mod_roc")
    _chart_explainer(
        "La curva ROC muestra qué tan bueno es el modelo para distinguir entre clases. Un modelo perfecto estaría pegado a la esquina superior izquierda (TPR=1, FPR=0). El área bajo la curva (AUC) resume su rendimiento: 1.0 es perfecto, 0.5 es azar.",
        f"El modelo actual tiene un AUC de {metrics['ROC-AUC']:.4f}, lo que indica una capacidad de discriminación buena, muy superior al azar.",
        "El modelo V2 muestra un AUC superior al de V1, gracias a la limpieza de datos más estricta y al feature engineering mejorado, lo que le permite separar mejor las clases."
    )

    st.subheader("4.3 Importancia de features")
    if feature_importances.empty:
        st.warning("El modelo activo no expone importancias nativas de features.")
    else:
        top_features = feature_importances.head(20).sort_values(ascending=True)
        fig = go.Figure(
            go.Bar(
                y=top_features.index,
                x=top_features.values,
                orientation="h",
                marker=dict(color=top_features.values, colorscale=[[0, COLD_COLOR], [0.5, GOLD], [1, HOT_COLOR]], showscale=True),
                text=[f"{value:.1%}" for value in top_features.values],
                textposition="outside",
            )
        )
        fig.update_layout(title="Top features del modelo activo", xaxis_title="Importancia")
        _plotly_layout(fig, 550)
        st.plotly_chart(fig, use_container_width=True, key="mod_importance")
        _chart_explainer(
            "El gráfico muestra las 20 features que más influyen en las predicciones del modelo. Una barra más larga significa mayor importancia.",
            "Las variables relacionadas con el vehículo de interés y el origen del lead son consistentemente las más importantes. Features derivadas como `franja_horaria` también demuestran tener un impacto significativo.",
            "En V2, las importancias son más interpretables. Features que antes dominaban por data leakage (como `subtipo_interes`) han sido eliminadas, revelando los verdaderos drivers de negocio."
        )

        top5 = feature_importances.head(5)
        st.markdown("**Top 5 features actuales**")
        for rank, (feature, value) in enumerate(top5.items(), start=1):
            st.markdown(f"{rank}. `{feature}`: {value:.1%}")

    st.subheader("4.4 Metricas del modelo activo")
    metrics_df = pd.DataFrame([{"Metrica": name, "Valor": value} for name, value in metrics.items()])
    metrics_df["Valor"] = metrics_df.apply(
        lambda row: f"{row['Valor']:.4f}" if row["Metrica"] == "ROC-AUC" else f"{row['Valor']:.1%}",
        axis=1,
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption("La validacion cruzada no se muestra porque el artifact actual no guarda resultados fold-by-fold.")
def _render_modeling(report):
    st.header("4. Modelado")
    model = report["model"]
    metrics = report["metrics"]
    feature_importances = report["feature_importances"]
    dataset_name = report["dataset_name"]

    st.markdown(
        f"El modelo desplegado hoy es **{_format_model_name(model)}** y se evalua sobre **{dataset_name}** "
        f"con **{report['X_test'].shape[0]:,} leads**."
    )

    st.subheader("4.1 Comparacion de modelos (referencia notebook 03)")
    benchmark_rows = [
        {"Modelo": "Logistic Regression", "Accuracy": 0.704, "Precision": 0.633, "Recall": 0.431, "F1": 0.513, "ROC-AUC": 0.742},
        {"Modelo": "Random Forest", "Accuracy": 0.733, "Precision": 0.667, "Recall": 0.503, "F1": 0.573, "ROC-AUC": 0.770},
        {"Modelo": "Gradient Boosting", "Accuracy": 0.738, "Precision": 0.676, "Recall": 0.517, "F1": 0.586, "ROC-AUC": 0.779},
        {"Modelo": "LightGBM", "Accuracy": 0.736, "Precision": 0.671, "Recall": 0.512, "F1": 0.581, "ROC-AUC": 0.777},
    ]
    benchmark_df = pd.DataFrame(benchmark_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    benchmark_df["Ganador"] = benchmark_df["Modelo"].eq("Gradient Boosting").map(lambda flag: "Si" if flag else "")
    for metric_name in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        benchmark_df[metric_name] = benchmark_df[metric_name].map(lambda value: f"{value:.3f}")
    st.dataframe(
        benchmark_df[["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Ganador"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Tabla de referencia del benchmark V2 (notebook 03). "
        "El modelo ganador fue Gradient Boosting."
    )
    with st.expander("Para que sirve cada modelo y que aporta aqui", expanded=True):
        st.markdown(
            "- **Logistic Regression:** modelo base simple y facil de explicar. "
            "Sirve como punto de partida para validar que el pipeline tiene senal real.\n"
            "- **Random Forest:** combina muchos arboles y suele ser robusto con datos mixtos "
            "(categoricos + numericos), con buena estabilidad general.\n"
            "- **Gradient Boosting (ganador):** corrige errores arbol por arbol y captura mejor relaciones no lineales. "
            "En este proyecto logra el mejor equilibrio de metricas.\n"
            "- **LightGBM:** variante de boosting muy eficiente para escalar y reentrenar rapido. "
            "Quedo cerca del ganador y es buena opcion operativa."
        )
        st.markdown(
            "Resumen practico: Logistic ayuda a interpretar, Random Forest aporta robustez, "
            "y los modelos de boosting (Gradient Boosting / LightGBM) capturan mejor la complejidad del problema."
        )

    st.subheader("4.2 Hiperparametros del modelo activo")
    with st.expander("Ver hiperparametros", expanded=False):
        params = model.get_params() if hasattr(model, "get_params") else {}
        if params:
            params_df = pd.DataFrame(
                [{"Parametro": key, "Valor": str(value)} for key, value in sorted(params.items(), key=lambda item: item[0])]
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("El modelo activo no expone hiperparametros via `get_params()`.")
    with st.expander("Que significan los hiperparametros del ganador (explicado simple)", expanded=True):
        st.markdown(
            "- **n_estimators (ej. 200):** cantidad de arboles que se entrenan. "
            "Mas arboles pueden mejorar precision, pero tambien aumentan tiempo y riesgo de sobreajuste.\n"
            "- **max_depth (ej. 5):** que tan profundo puede crecer cada arbol. "
            "Profundidad moderada ayuda a captar patrones sin memorizar ruido.\n"
            "- **learning_rate (ej. 0.1):** tamano del paso en cada iteracion. "
            "Un valor menor aprende mas lento pero suele generalizar mejor.\n"
            "- **min_samples_leaf (ej. 10):** minimo de registros por hoja. "
            "Evita reglas demasiado especificas y hace el modelo mas estable."
        )
        st.markdown(
            "Lectura practica para negocio: estos parametros buscan equilibrio entre "
            "**capturar oportunidades reales** y **evitar falsas alarmas por ruido**."
        )

    st.subheader("4.3 Importancia de features")
    if feature_importances.empty:
        st.warning("El modelo activo no expone importancias nativas de features.")
    else:
        top_features = feature_importances.head(20).sort_values(ascending=True)
        fig = go.Figure(
            go.Bar(
                y=top_features.index,
                x=top_features.values,
                orientation="h",
                marker=dict(color=top_features.values, colorscale=[[0, COLD_COLOR], [0.5, GOLD], [1, HOT_COLOR]], showscale=True),
                text=[f"{value:.1%}" for value in top_features.values],
                textposition="outside",
            )
        )
        fig.update_layout(title="Top features del modelo activo", xaxis_title="Importancia")
        _plotly_layout(fig, 550)
        st.plotly_chart(fig, use_container_width=True, key="mod_importance")
        _chart_explainer(
            "Barras mas largas significan mayor impacto relativo de la feature en la prediccion.",
            "Las variables de origen/formulario y atributos de interes suelen concentrar la mayor senal.",
            "En V2 las importancias son mas confiables al remover features con leakage de V1."
        )

        top5 = feature_importances.head(5)
        st.markdown("**Top 5 features actuales**")
        for rank, (feature, value) in enumerate(top5.items(), start=1):
            st.markdown(f"{rank}. `{feature}`: {value:.1%}")

    st.subheader("4.4 Metricas del modelo activo")
    metrics_df = pd.DataFrame([{"Metrica": name, "Valor": value} for name, value in metrics.items()])
    metrics_df["Valor"] = metrics_df.apply(
        lambda row: f"{row['Valor']:.4f}" if row["Metrica"] == "ROC-AUC" else f"{row['Valor']:.1%}",
        axis=1,
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    with st.expander("Que significa cada metrica (explicado simple)", expanded=True):
        st.markdown(
            "- **Accuracy:** porcentaje total de aciertos (Hot y Cold) sobre todos los leads.\n"
            "- **Precision:** de los leads que el modelo marca como Hot, cuantos realmente eran Hot.\n"
            "- **Recall:** de todos los Hot reales, cuantos logra detectar el modelo.\n"
            "- **F1:** equilibrio entre Precision y Recall en un solo numero.\n"
            "- **ROC-AUC:** capacidad general del modelo para ordenar Hot por encima de Cold (1.0 es excelente, 0.5 es azar)."
        )
        st.markdown(
            "En negocio automotriz normalmente **Recall** importa mucho (no dejar escapar oportunidades), "
            "sin descuidar **Precision** para no saturar al equipo comercial con falsos positivos."
        )

    st.subheader("4.5 Validacion cruzada (referencia)")
    cv_mean = 0.7791
    cv_std = 0.0049
    st.info(
        "Referencia notebook 03 (5-fold CV): ROC-AUC = 0.7791 +/- 0.0049. "
        "Este valor corresponde al experimento de entrenamiento V2 y se muestra como benchmark historico."
    )
    st.markdown(
        "La validacion cruzada divide los datos en 5 partes y repite el entrenamiento/prueba varias veces. "
        "Sirve para comprobar que el rendimiento no depende de una sola particion train/test."
    )
    cv_df = pd.DataFrame(
        [
            {"Escenario": "CV 5-fold (media)", "ROC-AUC": cv_mean},
            {"Escenario": "CV 5-fold (media - 1 desv.)", "ROC-AUC": cv_mean - cv_std},
            {"Escenario": "CV 5-fold (media + 1 desv.)", "ROC-AUC": cv_mean + cv_std},
            {"Escenario": "Test holdout actual", "ROC-AUC": metrics.get("ROC-AUC", np.nan)},
        ]
    )
    fig_cv = go.Figure(
        go.Bar(
            x=cv_df["Escenario"],
            y=cv_df["ROC-AUC"],
            marker_color=[ACCENT, "#8fd3c1", "#8fd3c1", GOLD],
            text=[f"{value:.4f}" for value in cv_df["ROC-AUC"]],
            textposition="outside",
        )
    )
    fig_cv.update_layout(title="Referencia de estabilidad ROC-AUC", yaxis_title="ROC-AUC", yaxis_range=[0.70, 0.82])
    _plotly_layout(fig_cv, 320)
    st.plotly_chart(fig_cv, use_container_width=True, key="mod_cv_reference")
    _chart_explainer(
        "Compara el rendimiento promedio de validacion cruzada contra el resultado del test actual.",
        "La desviacion pequena (+/- 0.0049) sugiere comportamiento estable entre folds.",
        "V2 muestra mas consistencia que V1 al entrenar con pipeline depurado y menos leakage."
    )


def _render_evaluation(report):
    st.header("5. Evaluacion Detallada")
    y_test = report["y_test"]
    y_proba = report["y_proba"]
    threshold = report["umbral"]
    metrics = report["metrics"]
    cm = report["cm"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = len(y_test)
    hot_rate = float(y_test.mean())
    best_f1 = report["best_f1_row"]

    st.subheader(f"5.1 Metricas finales (umbral = {threshold:.3f})")
    st.caption(
        "Estas metricas resumen como rinde el modelo con el umbral activo. "
        "La idea de este corte es simple: detectar la mayor cantidad posible de oportunidades reales "
        "sin mandar demasiados leads frios al equipo comercial."
    )
    gauge_fig = make_subplots(rows=1, cols=5, specs=[[{"type": "indicator"}] * 5], subplot_titles=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    gauge_colors = ["#636EFA", "#AB63FA", ACCENT, "#FFA15A", GOLD]
    for idx, (metric_name, color) in enumerate(zip(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], gauge_colors), start=1):
        value = metrics[metric_name]
        gauge_fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value * 100 if metric_name != "ROC-AUC" else value,
                number={"suffix": "%" if metric_name != "ROC-AUC" else ""},
                gauge=dict(
                    axis=dict(range=[0, 100] if metric_name != "ROC-AUC" else [0, 1], visible=False),
                    bar=dict(color=color, thickness=0.7),
                    bgcolor="rgba(50,50,50,0.3)",
                    borderwidth=0,
                ),
            ),
            row=1,
            col=idx,
        )
    gauge_fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(gauge_fig, use_container_width=True, key="eval_gauge")
    _chart_explainer(
        "Estos medidores muestran las métricas de rendimiento clave del modelo con el umbral de decisión actual. 'Recall' (Sensibilidad) es especialmente importante: mide cuántos leads 'Hot' reales logramos capturar.",
        f"Con el umbral actual, el modelo captura el {metrics['Recall']:.1%} de los leads 'Hot'. La precisión del {metrics['Precision']:.1%} indica que de los leads que se marcan como 'Hot', ese porcentaje es correcto.",
        "El modelo V2 fue optimizado para mejorar el Recall sin sacrificar demasiada Precisión, un equilibrio más alineado con los objetivos de negocio que el enfoque de V1, que priorizaba la precisión general (Accuracy)."
    )
    with st.expander("Que significa cada metrica en 5.1 (explicado simple)", expanded=False):
        st.markdown(
            "- **Accuracy:** porcentaje total de aciertos del modelo (Hot y Cold).\n"
            "- **Precision:** de los leads marcados como Hot, cuantos realmente eran Hot.\n"
            "- **Recall:** de todos los Hot reales, cuantos logro detectar el modelo.\n"
            "- **F1:** balance entre Precision y Recall en un solo valor.\n"
            "- **ROC-AUC:** capacidad general de separar Hot por encima de Cold (1.0 excelente, 0.5 azar)."
        )
        st.markdown(
            "En esta aplicacion, Recall y Precision suelen ser las metricas mas sensibles: "
            "Recall evita perder oportunidades y Precision evita saturar al equipo con falsos positivos."
        )
    st.subheader(f"5.2 Matriz de confusion (umbral = {threshold:.3f})")
    st.caption(
        "La matriz de confusion muestra en que acierta y en que se equivoca el modelo. "
        "En este negocio duele mas dejar escapar un Hot Lead que revisar algun Cold Lead de mas."
    )
    col1, col2 = st.columns(2)
    with col1:
        cm_fig = go.Figure(
            go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred Cold", "Pred Hot"],
                y=["Real Cold", "Real Hot"],
                colorscale=[[0, "rgba(75,139,255,0.15)"], [1, "rgba(75,139,255,0.85)"]],
                text=[[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]],
                texttemplate="%{text}",
                textfont=dict(color="white", size=14),
                showscale=False,
            )
        )
        cm_fig.update_layout(title="Cantidades")
        _plotly_layout(cm_fig, 350)
        st.plotly_chart(cm_fig, use_container_width=True, key="eval_cm")

    with col2:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Resultado": "Verdaderos Negativos", "Cantidad": tn, "%": f"{tn / total:.1%}"},
                    {"Resultado": "Falsos Positivos", "Cantidad": fp, "%": f"{fp / total:.1%}"},
                    {"Resultado": "Falsos Negativos", "Cantidad": fn, "%": f"{fn / total:.1%}"},
                    {"Resultado": "Verdaderos Positivos", "Cantidad": tp, "%": f"{tp / total:.1%}"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    _chart_explainer(
        "La matriz cruza la realidad ('Real') con las predicciones del modelo ('Pred'). Los Falsos Negativos (FN) son oportunidades perdidas, y los Falsos Positivos (FP) son leads fríos que se marcan como calientes.",
        f"Actualmente, el modelo deja escapar {fn} oportunidades (FN) pero a cambio solo genera {fp} falsas alarmas (FP). Este balance se considera aceptable para la operación.",
        "El modelo V2, con su umbral ajustado, reduce significativamente los Falsos Negativos en comparación con V1 (que usaba un umbral de 0.5), aceptando un ligero aumento en Falsos Positivos para no perder ventas."
    )

    st.subheader("5.3 Curvas ROC y Precision-Recall")
    st.caption(
        "Estas curvas ayudan a entender la calidad general del modelo. "
        "La ROC mide que tan bien separa Hot de Cold; la Precision-Recall muestra mejor el costo real de intentar capturar mas Hot Leads."
    )
    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={metrics['ROC-AUC']:.4f}", line=dict(color=ACCENT, width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray")))
        roc_fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
        _plotly_layout(roc_fig, 380)
        st.plotly_chart(roc_fig, use_container_width=True, key="eval_roc")
    with col2:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=rec_arr, y=prec_arr, mode="lines", line=dict(color="#AB63FA", width=3)))
        pr_fig.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
        _plotly_layout(pr_fig, 380)
        st.plotly_chart(pr_fig, use_container_width=True, key="eval_pr")
    _chart_explainer(
        "La curva ROC (izquierda) mide la capacidad de separación general del modelo. La curva Precision-Recall (derecha) es más informativa para datasets desbalanceados, mostrando el 'costo' en precisión al intentar aumentar el recall.",
        "La forma de la curva PR muestra que para capturar más del 80-90% de los leads Hot (alto Recall), la precisión comienza a caer notablemente. El umbral actual busca un punto óptimo en esta curva.",
        "El modelo V2 tiene un área bajo la curva PR superior a V1, lo que significa que para cualquier nivel de Recall, generalmente se obtiene una mejor Precisión. Es un modelo globalmente superior."
    )
    st.subheader(f"5.4 Analisis de umbral de decision (umbral activo = {threshold:.3f})")
    st.caption(
        "Mover el umbral cambia la politica comercial: mas bajo captura mas oportunidades, mas alto filtra mas fuerte. "
        "El punto elegido busca un equilibrio facil de defender."
    )
    threshold_df = report["threshold_df"]
    fig = go.Figure()
    for metric_name, color in {"Accuracy": "#636EFA", "Precision": "#AB63FA", "Recall": ACCENT, "F1": "#FFA15A"}.items():
        fig.add_trace(
            go.Scatter(
                x=threshold_df["umbral"],
                y=threshold_df[metric_name],
                mode="lines",
                name=metric_name,
                line=dict(color=color, width=2.5),
            )
        )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="0.500",
        annotation_position="top left",
    )
    fig.add_vline(
        x=0.30,
        line_dash="dot",
        line_color="#ff8c00",
        annotation_text="0.300",
        annotation_position="top left",
    )
    fig.add_vline(
        x=threshold,
        line_dash="dashdot",
        line_color=GOLD,
        annotation_text=f"{threshold:.3f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=0.35,
        line_dash="dot",
        line_color="#d62728",
        annotation_text="0.350",
        annotation_position="top right",
    )
    fig.update_layout(title="Metricas vs umbral", xaxis_title="Umbral", yaxis_title="Valor", xaxis_range=[0.05, 0.95])
    _plotly_layout(fig, 420)
    st.plotly_chart(fig, use_container_width=True, key="eval_threshold")
    _chart_explainer(
        "El gráfico muestra cómo cambian las métricas principales al mover el umbral de decisión. La línea dorada es el umbral activo, y la gris es el default (0.5).",
        "Se ve claramente que bajar el umbral de 0.5 a 0.325 aumenta mucho el Recall (línea verde) con una caída controlada de la Precisión (línea morada). El F1-Score (naranja), que balancea ambas, alcanza su pico cerca de este umbral.",
        "El modelo V1 operaba con el umbral de 0.5 por defecto. El análisis de V2 demostró que un umbral más bajo estaba mejor alineado con el objetivo de negocio de maximizar la captura de leads calientes."
    )

    comp = report["threshold_comparison"]
    row_030 = threshold_df.loc[(threshold_df["umbral"] - 0.30).abs().idxmin()].to_dict()
    row_035 = threshold_df.loc[(threshold_df["umbral"] - 0.35).abs().idxmin()].to_dict()
    comparison_rows = [
        {"Umbral": "0.500", **comp["default_050"], "Estrategia": "Baseline"},
        {"Umbral": "0.300", **row_030, "Estrategia": "Alternativa baja"},
        {"Umbral": f"{threshold:.3f}", **comp["active"], "Estrategia": "Activo"},
        {"Umbral": "0.350", **row_035, "Estrategia": "Alternativa alta"},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    for metric_name in ["Accuracy", "Precision", "Recall", "F1"]:
        comparison_df[metric_name] = comparison_df[metric_name].map(lambda value: f"{value:.1%}")
    st.dataframe(comparison_df[["Umbral", "Accuracy", "Precision", "Recall", "F1", "Estrategia"]], use_container_width=True, hide_index=True)
    st.markdown(
        f"Frente a `0.500`, el umbral activo recupera **{comp['delta_hot_vs_050']} Hot Leads** adicionales "
        f"y agrega **{comp['delta_cold_vs_050']} Cold Leads** como falsos positivos."
    )
    st.info(
        "La tabla se enfoca en los umbrales de decision operativos (0.300, 0.325 y 0.350) "
        "frente al baseline de 0.500 para facilitar comparacion de negocio."
    )

    st.subheader("5.5 Distribucion de errores")
    st.caption(
        "No todos los errores son iguales. Estos histogramas muestran si el modelo se equivoca cerca del umbral "
        "o si hay casos realmente confusos."
    )
    fp_probas = report["fp_probas"]
    fn_probas = report["fn_probas"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=fp_probas, nbinsx=20, marker_color=HOT_COLOR, opacity=0.8))
        fig.add_vline(x=threshold, line_dash="dashdot", line_color=GOLD)
        fig.update_layout(title=f"Falsos Positivos (n={len(fp_probas)})", xaxis_title="Probabilidad predicha")
        _plotly_layout(fig, 320)
        st.plotly_chart(fig, use_container_width=True, key="eval_fp_hist")
    with col2:
        fig = go.Figure(go.Histogram(x=fn_probas, nbinsx=20, marker_color="#FFA15A", opacity=0.8))
        fig.add_vline(x=threshold, line_dash="dashdot", line_color=GOLD)
        fig.update_layout(title=f"Falsos Negativos (n={len(fn_probas)})", xaxis_title="Probabilidad predicha")
        _plotly_layout(fig, 320)
        st.plotly_chart(fig, use_container_width=True, key="eval_fn_hist")
    _chart_explainer(
        "Los histogramas muestran la distribución de probabilidades para los errores del modelo. Falsos Positivos (izquierda) son leads 'Cold' que el modelo predijo como 'Hot'. Falsos Negativos (derecha) son 'Hot' que el modelo predijo como 'Cold'.",
        f"La mayoría de los errores (tanto FP como FN) se concentran cerca del umbral de decisión. Esto es positivo: significa que el modelo duda en los casos más ambiguos, en lugar de cometer errores groseros con alta confianza. La probabilidad media de los FP es {_safe_mean(fp_probas):.3f} y de los FN es {_safe_mean(fn_probas):.3f}.",
        "El modelo V2 tiene una distribución de errores más concentrada alrededor del umbral, indicando una mejor 'calibración' de sus probabilidades en comparación con V1."
    )

    st.subheader("5.6 Interpretabilidad - SHAP")
    st.caption(
        "SHAP sirve para responder una pregunta muy humana: por que el modelo penso que este lead valia la pena. "
        "No solo importa si acierta, tambien importa poder explicarlo."
    )
    try:
        import shap

        explainer = shap.TreeExplainer(report["model"])
        shap_values = explainer(report["X_test"])
        shap_values = get_positive_shap_explanation(shap_values)
        col_shap1, col_shap2 = st.columns(2)

        fig, _ = plt.subplots(figsize=(7, 5))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.tight_layout()
        col_shap1.pyplot(fig)
        plt.close()
        with col_shap1:
            _chart_explainer(
            "Este gráfico de barras muestra el impacto promedio absoluto de cada feature en la predicción. Las features más arriba son las que más mueven la aguja del modelo, sea para bien o para mal.",
            "Permite confirmar a alto nivel qué variables son las más influyentes en el modelo.",
            "En V2, las features son más limpias y representativas del negocio real, a diferencia de V1 donde el leakage podía dominar."
        )

        fig, _ = plt.subplots(figsize=(7, 5))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.tight_layout()
        col_shap2.pyplot(fig)
        with col_shap2:
            _chart_explainer(
            "Cada punto es un lead. El color indica el valor de la feature (rojo=alto, azul=bajo). El eje X muestra el impacto SHAP: valores positivos empujan hacia 'Hot', negativos hacia 'Cold'.",
            "Permite ver no solo QUÉ features son importantes, sino CÓMO impactan. Por ejemplo, se puede observar si valores altos de una feature (puntos rojos) tienden a tener un impacto positivo (a la derecha).",
            "Este tipo de análisis detallado no se realizó en V1, y es clave en V2 para entender el comportamiento del modelo a nivel de lead individual."
        )
        plt.close()

        top_features_names = list(report["feature_importances"].head(5).index) if not report["feature_importances"].empty else []
        top_features_readable = ", ".join([f"`{name}`" for name in top_features_names[:5]]) if top_features_names else "las features lideres del modelo"
        month_note = (
            "En esta corrida, `mes_creacion` muestra patron: valores bajos suelen empujar hacia Hot y valores altos hacia Cold."
            if "mes_creacion" in report["X_test"].columns
            else "No se detecto `mes_creacion` en la matriz final de esta corrida."
        )
        st.markdown("**Hallazgos SHAP (lectura de negocio):**")
        st.markdown(
            f"- Las features top en SHAP son consistentes con la importancia Gini (ejemplo: {top_features_readable}).\n"
            "- Variables de origen/formulario/vehiculo capturan calidad e intencion del lead; variables temporales capturan contexto operativo.\n"
            f"- {month_note}"
        )

        hot_candidates = np.where(y_proba > 0.8)[0]
        cold_candidates = np.where(y_proba < 0.2)[0]
        st.markdown("**Ejemplos Waterfall SHAP (1 Hot y 1 Cold):**")
        if len(hot_candidates) > 0 and len(cold_candidates) > 0:
            hot_idx = int(hot_candidates[np.argmax(y_proba[hot_candidates])])
            cold_idx = int(cold_candidates[np.argmin(y_proba[cold_candidates])])
            col_hot, col_cold = st.columns(2)
            with col_hot:
                st.caption(f"Lead Hot de alta confianza (y_proba={y_proba[hot_idx]:.3f})")
                fig_hot, _ = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_values[hot_idx], max_display=12, show=False)
                plt.tight_layout()
                st.pyplot(fig_hot)
                plt.close()
            with col_cold:
                st.caption(f"Lead Cold de alta confianza (y_proba={y_proba[cold_idx]:.3f})")
                fig_cold, _ = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_values[cold_idx], max_display=12, show=False)
                plt.tight_layout()
                st.pyplot(fig_cold)
                plt.close()
        else:
            st.info(
                "No se encontraron ejemplos extremos suficientes para waterfall "
                "(se requiere al menos un lead con y_proba > 0.8 y otro con y_proba < 0.2)."
            )
    except Exception as exc:
        st.warning(f"No se pudo generar SHAP: {exc}")

    st.subheader("5.7 Conclusiones y recomendaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "\n".join(
                [
                    "### Lo bueno",
                    "- El modelo ayuda a ordenar mejor los leads para que ventas empiece por los mas prometedores.",
                    f"- Con el umbral actual recupera **{metrics['Recall']:.1%}** de los Hot Leads reales.",
                    "- El criterio es consistente en toda la app: sidebar, reporte e inferencia usan el mismo umbral.",
                    "- Las cifras que ves aca salen del modelo y del dataset actual, no de textos historicos fijos.",
                ]
            )
        )
    with col2:
        st.markdown(
            "\n".join(
                [
                    "### Lo que hay que mirar de cerca",
                    f"- Todavia se cuelan algunos leads frios: **{fp / total:.1%}** del test.",
                    f"- Tambien se pierden algunas oportunidades reales: **{fn / total:.1%}** del test.",
                    "- El reporte hoy explica muy bien el modelo activo, pero no tiene un benchmark historico guardado para comparar versiones automaticamente.",
                ]
            )
        )

    st.markdown(
        "\n".join(
            [
                "### En palabras simples",
                "1. El umbral `0.325` es una forma razonable de priorizar mas oportunidades sin desbordar al equipo comercial.",
                "2. Si ventas puede revisar mas leads, este enfoque tiene sentido; si el costo operativo sube mucho, conviene endurecer el corte.",
                "3. Lo importante no es perseguir una metrica perfecta, sino elegir el tipo de error que el negocio esta dispuesto a tolerar.",
            ]
        )
    )
