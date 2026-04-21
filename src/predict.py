"""
Predicción de Hot/Cold Leads a partir de un archivo Excel de Salesforce.

Uso:
    python src/predict.py "ruta/al/archivo.xlsx"
    python src/predict.py "ruta/al/archivo.xlsx" -o resultados.csv

El script:
  1. Lee el Excel crudo (mismo formato que Salesforce exporta)
  2. Aplica todas las transformaciones del pipeline de data engineering + feature engineering
  3. Genera predicciones con el modelo entrenado (Random Forest, umbral 0.35)
  4. Exporta un CSV con: lead_id, predicción, probabilidad

Si el archivo tiene la columna 'Cualificación' con datos, también compara
las predicciones con la realidad y reporta accuracy.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd


# --- Rutas ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "models", "preprocessing_config.joblib")

DIAS_SEMANA_EN_ES = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miércoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sábado",
    "Sunday": "domingo",
}


# ─────────────────────────── Utilidades ───────────────────────────


def fix_encoding(text):
    """Corrige encoding latino corrupto en nombres de columna."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clasificar_franja(hora):
    """Clasifica hora en franja horaria."""
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "manana"
    elif 12 <= hora < 18:
        return "tarde"
    return "noche"


def find_column(columns, keywords, exclude_keywords=None):
    """Busca una columna por palabras clave (case-insensitive)."""
    for col in columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            if exclude_keywords and any(ek.lower() in col_lower for ek in exclude_keywords):
                continue
            return col
    return None


# ─────────────────────────── Pipeline ───────────────────────────


def read_excel(path):
    """Lee el Excel de Salesforce y corrige encoding de columnas."""
    df = pd.read_excel(path)
    df.columns = [fix_encoding(c) for c in df.columns]
    print(f"  Leído: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def extract_date_features(df):
    """Extrae mes, día, hora y día de semana desde 'Fecha de creación'."""
    date_col = find_column(
        df.columns,
        keywords=["fecha", "creación"],
        exclude_keywords=["cliente", "reasign"],
    )
    if date_col is None:
        date_col = find_column(df.columns, keywords=["fecha", "creacion"])

    if date_col is None:
        raise ValueError(
            "No se encontró columna 'Fecha de creación'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["mes_creacion"] = dates.dt.month.astype("Int64")
    df["dia_creacion"] = dates.dt.day.astype("Int64")
    df["hora_creacion"] = dates.dt.hour.astype("Int64")
    df["dia_semana_creacion"] = dates.dt.day_name().map(DIAS_SEMANA_EN_ES)

    nulos_fecha = dates.isna().sum()
    if nulos_fecha > 0:
        print(f"  AVISO: {nulos_fecha} filas con fecha inválida → se eliminarán")
        df = df[dates.notna()].copy()

    return df


def select_columns(df):
    """Selecciona y renombra las columnas necesarias para el modelo."""
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

    # "Origen" sin más (columna exacta, no "Origen de creación")
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
    """Imputa valores nulos con reglas de negocio."""
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


def apply_feature_engineering(df, artifacts):
    """Aplica feature engineering: derivadas, agrupación, encoding."""
    known_cats = artifacts["known_categories"]
    conc_means = artifacts["concesionario_means"]
    global_mean = artifacts["global_mean"]
    training_cols = artifacts["training_columns"]

    # Features derivadas
    df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(
        ["sábado", "domingo"]
    ).astype(int)
    df["franja_horaria"] = df["hora_creacion"].apply(clasificar_franja)

    # Agrupar categorías desconocidas en "otros"
    for col, valid_cats in known_cats.items():
        if col in df.columns:
            df.loc[~df[col].isin(valid_cats), col] = "otros"

    # Target encoding para concesionario
    df["concesionario_target_enc"] = (
        df["concesionario"].map(conc_means).fillna(global_mean)
    )
    df = df.drop(columns=["concesionario"])

    # One-hot encoding
    onehot_cols = list(df.select_dtypes(include="object").columns)
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)

    # Alinear con columnas de entrenamiento
    df = df.reindex(columns=training_cols, fill_value=0)

    return df


# ─────────────────────────── Main ───────────────────────────


def predict(input_path, output_path=None):
    """Pipeline completo: Excel crudo → predicciones."""
    print(f"\n{'='*60}")
    print(f"  PREDICCIÓN DE HOT/COLD LEADS")
    print(f"{'='*60}")

    print(f"\nCargando modelo y artefactos...")
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    umbral = artifacts["umbral"]
    print(f"  Modelo: {type(model).__name__}")
    print(f"  Umbral: {umbral}")

    print(f"\nLeyendo archivo: {os.path.basename(input_path)}")
    df = read_excel(input_path)

    print(f"\nExtrayendo features de fecha...")
    df = extract_date_features(df)

    print(f"\nSeleccionando columnas...")
    df = select_columns(df)

    # Separar lead_id para el output
    lead_ids = df.pop("lead_id") if "lead_id" in df.columns else None

    # Separar cualificación real (si existe) para validación
    y_real = None
    if "cualificacion" in df.columns:
        mask_con_cualif = df["cualificacion"].notna()
        mask_bot = df["cualificacion"].isna()
        n_bot = mask_bot.sum()
        n_con = mask_con_cualif.sum()

        print(f"  Leads con cualificación: {n_con:,}")
        print(f"  Leads sin cualificación (bot): {n_bot:,} → se excluyen")

        if n_bot > 0:
            df = df[~mask_bot].copy()
            if lead_ids is not None:
                lead_ids = lead_ids[~mask_bot].copy()

        y_real = (df["cualificacion"] == "Contacto interesado").astype(int)
        df = df.drop(columns=["cualificacion"])

    print(f"\nLimpiando datos...")
    df = clean_nulls(df)
    print(f"  Leads a evaluar: {len(df):,}")

    print(f"\nAplicando feature engineering...")
    df_model = apply_feature_engineering(df, artifacts)

    print(f"\nGenerando predicciones...")
    y_proba = model.predict_proba(df_model)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    # Construir resultados
    results = pd.DataFrame({
        "prediccion": np.where(y_pred == 1, "Hot", "Cold"),
        "probabilidad_hot": np.round(y_proba, 4),
    })

    if lead_ids is not None:
        results.insert(0, "lead_id", lead_ids.values[:len(results)])

    # Validación contra datos reales
    if y_real is not None and len(y_real) == len(y_pred):
        results["real"] = np.where(y_real.values == 1, "Hot", "Cold")
        results["acierto"] = np.where(
            y_pred == y_real.values, "✔", "✘"
        )

        correct = (y_pred == y_real.values).sum()
        total = len(y_pred)
        hot_real = y_real.sum()
        hot_pred = y_pred.sum()

        print(f"\n  Validación vs datos reales:")
        print(f"    Accuracy: {correct/total*100:.1f}% ({correct:,}/{total:,})")
        print(f"    Hot reales: {hot_real:,} | Hot predichos: {hot_pred:,}")
        from sklearn.metrics import precision_score, recall_score, f1_score
        print(f"    Precision: {precision_score(y_real, y_pred):.4f}")
        print(f"    Recall:    {recall_score(y_real, y_pred):.4f}")
        print(f"    F1-Score:  {f1_score(y_real, y_pred):.4f}")

    # Resumen
    n_hot = y_pred.sum()
    n_cold = len(y_pred) - n_hot
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  Total evaluados: {len(results):,}")
    print(f"  Hot (enviar al concesionario): {n_hot:,} ({n_hot/len(y_pred)*100:.1f}%)")
    print(f"  Cold (no enviar):              {n_cold:,} ({n_cold/len(y_pred)*100:.1f}%)")

    # Exportar
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predicciones.csv"

    results.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n  Exportado a: {output_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicción de Hot/Cold Leads desde Excel de Salesforce"
    )
    parser.add_argument("input", help="Ruta al archivo Excel de Salesforce (.xlsx)")
    parser.add_argument("-o", "--output", help="Ruta del CSV de salida (opcional)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Archivo no encontrado: {args.input}")
        sys.exit(1)

    predict(args.input, args.output)
