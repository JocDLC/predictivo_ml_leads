"""
Pipeline de inferencia programático.
Reutiliza la lógica de src/predict.py adaptada para Streamlit.
"""

import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "models", "preprocessing_config.joblib")
DEFAULT_UMBRAL = 0.325
LEGACY_UMBRAL = 0.35

DIAS_SEMANA_EN_ES = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miércoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sábado",
    "Sunday": "domingo",
}


def fix_encoding(text):
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clasificar_franja(hora):
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "manana"
    elif 12 <= hora < 18:
        return "tarde"
    return "noche"


def find_column(columns, keywords, exclude_keywords=None):
    for col in columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            if exclude_keywords and any(ek.lower() in col_lower for ek in exclude_keywords):
                continue
            return col
    return None


def load_model_and_artifacts():
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    return model, artifacts


def resolve_umbral(artifacts):
    stored = artifacts.get("umbral")
    if stored is None:
        return DEFAULT_UMBRAL

    try:
        value = float(stored)
    except (TypeError, ValueError):
        return DEFAULT_UMBRAL

    return DEFAULT_UMBRAL if np.isclose(value, LEGACY_UMBRAL) else value


def get_expected_features(model, artifacts):
    expected_features = list(getattr(model, "feature_names_in_", []))
    if expected_features:
        return expected_features

    training_columns = artifacts.get("training_columns", [])
    if training_columns:
        return training_columns

    raise ValueError("No fue posible determinar las columnas esperadas por el modelo.")


def extract_date_features(df):
    date_col = find_column(
        df.columns,
        keywords=["fecha", "creación"],
        exclude_keywords=["cliente", "reasign"],
    )
    if date_col is None:
        date_col = find_column(df.columns, keywords=["fecha", "creacion"])
    if date_col is None:
        raise ValueError(
            f"No se encontró columna 'Fecha de creación'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["mes_creacion"] = dates.dt.month.astype("Int64")
    df["dia_creacion"] = dates.dt.day.astype("Int64")
    df["hora_creacion"] = dates.dt.hour.astype("Int64")
    df["dia_semana_creacion"] = dates.dt.day_name().map(DIAS_SEMANA_EN_ES)

    df = df[dates.notna()].copy()
    return df


def select_columns(df):
    col_map = {}
    mappings = [
        (["nombre", "formulario"], [], "nombre_formulario"),
        (["campaña"], [], "campana"),
        (["origen", "creación"], [], "origen_creacion"),
        (["origen", "creacion"], [], "origen_creacion"),
        (["vehículo", "interés"], [], "vehiculo_interes"),
        (["vehiculo", "interes"], [], "vehiculo_interes"),
        (["nombre corto", "concesión"], [], "concesionario"),
        (["nombre corto", "concesion"], [], "concesionario"),
        (["cualificación"], [], "cualificacion"),
        (["cualificacion"], [], "cualificacion"),
        (["lead id"], [], "lead_id"),
    ]

    for keywords, exclude, target_name in mappings:
        if target_name not in col_map.values():
            found = find_column(df.columns, keywords, exclude)
            if found:
                col_map[found] = target_name

    for c in df.columns:
        if c.strip().lower() == "origen" and c not in col_map:
            col_map[c] = "origen"
            break

    df = df.rename(columns=col_map)

    needed = [
        "mes_creacion", "dia_creacion", "hora_creacion", "dia_semana_creacion",
        "nombre_formulario", "campana", "origen_creacion",
        "vehiculo_interes", "concesionario", "origen",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas no encontradas: {missing}")

    optional = ["lead_id", "cualificacion"]
    keep = needed + [c for c in optional if c in df.columns]
    return df[keep].copy()


def clean_nulls(df):
    null_rules = {
        "campana": "sin_campana",
        "origen": "desconocido",
        "vehiculo_interes": "sin_vehiculo",
        "concesionario": "sin_concesionario",
    }
    for col, fill_value in null_rules.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("desconocido")
    for col in df.select_dtypes(include=["int64", "float64", "Int64"]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def transform_with_encoder(encoder, series):
    transformed = encoder.transform(series)

    if isinstance(transformed, pd.DataFrame):
        if series.name in transformed.columns:
            transformed = transformed[series.name]
        else:
            transformed = transformed.iloc[:, 0]

    if isinstance(transformed, pd.Series):
        return pd.to_numeric(transformed, errors="coerce")

    return pd.to_numeric(pd.Series(transformed, index=series.index), errors="coerce")


def apply_feature_engineering(df, artifacts, expected_features):
    known_cats = artifacts["known_categories"]
    conc_means = artifacts["concesionario_means"]
    global_mean = artifacts["global_mean"]
    bayesian_encoders = artifacts.get("bayesian_encoders", {})

    df = df.copy()
    df_model = pd.DataFrame(index=df.index)

    df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(
        ["sábado", "domingo"]
    ).astype(int)
    df["franja_horaria"] = df["hora_creacion"].apply(clasificar_franja)

    for col, valid_cats in known_cats.items():
        if col in df.columns:
            df.loc[~df[col].isin(valid_cats), col] = "otros"

    df_model["mes_creacion"] = pd.to_numeric(df["mes_creacion"], errors="coerce")
    df_model["dia_creacion"] = pd.to_numeric(df["dia_creacion"], errors="coerce")
    df_model["hora_creacion"] = pd.to_numeric(df["hora_creacion"], errors="coerce")
    df_model["es_fin_de_semana"] = df["es_fin_de_semana"]

    concesionario_encoded = df["concesionario"].map(conc_means).fillna(global_mean)
    if "concesionario" in expected_features:
        df_model["concesionario"] = concesionario_encoded
    if "concesionario_target_enc" in expected_features:
        df_model["concesionario_target_enc"] = concesionario_encoded

    encoded_columns = ["nombre_formulario", "campana", "vehiculo_interes", "origen"]
    for col in encoded_columns:
        encoder = bayesian_encoders.get(col)
        if encoder is None:
            continue

        encoded_values = transform_with_encoder(encoder, df[col])
        if col in expected_features:
            df_model[col] = encoded_values
        if f"{col}_bayes_enc" in expected_features:
            df_model[f"{col}_bayes_enc"] = encoded_values

    one_hot_specs = {
        "origen_creacion": "origen_creacion",
        "dia_semana_creacion": "dia_semana_creacion",
        "franja_horaria": "franja_horaria",
    }

    for source_col, prefix in one_hot_specs.items():
        prefixed_expected = [feature for feature in expected_features if feature.startswith(f"{prefix}_")]
        if prefixed_expected:
            dummies = pd.get_dummies(df[source_col], prefix=prefix, drop_first=True, dtype=int)
            for feature in prefixed_expected:
                df_model[feature] = dummies.get(feature, pd.Series(0, index=df.index, dtype=int))

        encoder = bayesian_encoders.get(source_col)
        bayes_feature_name = f"{source_col}_bayes_enc"
        if encoder is not None and bayes_feature_name in expected_features:
            df_model[bayes_feature_name] = transform_with_encoder(encoder, df[source_col])

    df_model = df_model.reindex(columns=expected_features, fill_value=0)

    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

    return df_model


def run_inference(uploaded_file, umbral=None):
    """
    Pipeline completo: archivo Excel → DataFrame con predicciones.

    Args:
        uploaded_file: Archivo Excel subido por el usuario.
        umbral: Umbral de decisión (default: usa el guardado en artifacts).

    Retorna:
        results: DataFrame con columnas [lead_id, prediccion, probabilidad_hot, ...]
        df_model: DataFrame con features procesadas (para SHAP)
        stats: dict con estadísticas del proceso
    """
    model, artifacts = load_model_and_artifacts()
    expected_features = get_expected_features(model, artifacts)
    if umbral is None:
        umbral = resolve_umbral(artifacts)

    df = pd.read_excel(uploaded_file)
    df.columns = [fix_encoding(c) for c in df.columns]
    total_raw = len(df)

    df = extract_date_features(df)
    df = select_columns(df)

    lead_ids = df.pop("lead_id") if "lead_id" in df.columns else None

    y_real = None
    n_bot = 0
    if "cualificacion" in df.columns:
        mask_bot = df["cualificacion"].isna()
        n_bot = mask_bot.sum()
        if n_bot > 0:
            df = df[~mask_bot].copy()
            if lead_ids is not None:
                lead_ids = lead_ids[~mask_bot].copy()
        y_real = (df["cualificacion"] == "Contacto interesado").astype(int)
        df = df.drop(columns=["cualificacion"])

    # Guardar datos originales para mostrar en grilla
    df_display = df.copy()

    df = clean_nulls(df)
    df_model = apply_feature_engineering(df, artifacts, expected_features)

    y_proba = model.predict_proba(df_model)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    results = pd.DataFrame({
        "prediccion": np.where(y_pred == 1, "Hot 🔥", "Cold ❄️"),
        "probabilidad_hot": np.round(y_proba * 100, 1),
    })

    if lead_ids is not None:
        results.insert(0, "lead_id", lead_ids.values[:len(results)])

    # Agregar columnas de contexto para la grilla
    for col in ["vehiculo_interes", "concesionario", "campana",
                 "nombre_formulario", "origen", "origen_creacion"]:
        if col in df_display.columns:
            results[col] = df_display[col].values[:len(results)]

    if y_real is not None and len(y_real) == len(y_pred):
        results["real"] = np.where(y_real.values == 1, "Hot 🔥", "Cold ❄️")
        accuracy = (y_pred == y_real.values).mean() * 100
    else:
        accuracy = None

    stats = {
        "total_raw": total_raw,
        "n_bot": n_bot,
        "total_evaluados": len(results),
        "n_hot": int(y_pred.sum()),
        "n_cold": int(len(y_pred) - y_pred.sum()),
        "accuracy": accuracy,
        "umbral": umbral,
    }

    return results, df_model, stats
