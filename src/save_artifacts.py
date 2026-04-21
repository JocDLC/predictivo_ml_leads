"""
Guarda los artefactos de preprocesamiento necesarios para inferencia.
Ejecutar UNA VEZ después de entrenar el modelo.

Artefactos guardados en models/preprocessing_config.joblib:
  - known_categories: categorías válidas por columna (las demás → "otros")
  - concesionario_means: target encoding del concesionario
  - global_mean: media global del target (fallback)
  - training_columns: lista exacta de 48 features del modelo
  - umbral: umbral de decisión (0.35)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

CLEANED_PATH = "data/processed/leads_cleaned.csv"
OUTPUT_PATH = "models/preprocessing_config.joblib"
UMBRAL_DECISION = 0.35
UMBRAL_FRECUENCIA_PCT = 1.0


def main():
    df = pd.read_csv(CLEANED_PATH)
    df = df.drop(columns=["anio_creacion", "subtipo_interes"])

    df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(
        ["sábado", "domingo"]
    ).astype(int)

    df["franja_horaria"] = df["hora_creacion"].apply(clasificar_franja)

    umbral_abs = int(len(df) * UMBRAL_FRECUENCIA_PCT / 100)
    cols_agrupar = [
        "nombre_formulario", "vehiculo_interes",
        "origen", "campana", "concesionario",
    ]

    known_categories = {}
    for col in cols_agrupar:
        vc = df[col].value_counts()
        known_categories[col] = vc[vc >= umbral_abs].index.tolist()
        df.loc[~df[col].isin(known_categories[col]), col] = "otros"

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_temp = X_train.copy()
    train_temp["target"] = y_train.values
    concesionario_means = (
        train_temp.groupby("concesionario")["target"].mean().to_dict()
    )
    global_mean = float(y_train.mean())

    X_train["concesionario_target_enc"] = (
        X_train["concesionario"].map(concesionario_means).fillna(global_mean)
    )
    X_train = X_train.drop(columns=["concesionario"])

    onehot_cols = list(X_train.select_dtypes(include="object").columns)
    X_train = pd.get_dummies(
        X_train, columns=onehot_cols, drop_first=True, dtype=int
    )

    if "plataforma_MX_LEAD_QUALIF" in X_train.columns:
        X_train = X_train.drop(columns=["plataforma_MX_LEAD_QUALIF"])

    training_columns = X_train.columns.tolist()

    artifacts = {
        "known_categories": known_categories,
        "concesionario_means": concesionario_means,
        "global_mean": global_mean,
        "training_columns": training_columns,
        "umbral": UMBRAL_DECISION,
    }

    joblib.dump(artifacts, OUTPUT_PATH)

    print(f"Artefactos guardados en: {OUTPUT_PATH}")
    print(f"  Features del modelo: {len(training_columns)}")
    print(f"  Concesionarios mapeados: {len(concesionario_means)}")
    print(f"  Umbral de decisión: {UMBRAL_DECISION}")
    for col, cats in known_categories.items():
        print(f"  {col}: {len(cats)} categorías válidas")


def clasificar_franja(hora):
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "manana"
    elif 12 <= hora < 18:
        return "tarde"
    return "noche"


if __name__ == "__main__":
    main()
