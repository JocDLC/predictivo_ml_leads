"""
Guarda los artefactos de preprocesamiento necesarios para inferencia.
Ejecutar UNA VEZ después de entrenar el modelo.

Artefactos guardados en models/preprocessing_config.joblib:
  - known_categories: categorías válidas por columna (las demás → "otros")
  - concesionario_means: target encoding del concesionario
  - global_mean: media global del target (fallback)
  - bayesian_encoders: diccionario de codificadores Bayesianos entrenados (v2)
  - training_columns: lista exacta de 11 features del modelo (v2)
  - umbral: umbral de decisión (0.35)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from category_encoders import TargetEncoder # Asumo que usas esta o similar

# --- Constantes para Bayesian Encoding ---
SMOOTH_M = 100 # Factor de suavizado para Bayesian Encoding


CLEANED_PATH = "data/processed/leads_cleaned.csv"
OUTPUT_PATH = "models/preprocessing_config.joblib"
UMBRAL_DECISION = 0.35
UMBRAL_FRECUENCIA_PCT = 1.0
SMOOTH_M = 100
ENCODE_2FEAT_COLS = ["vehiculo_interes", "origen", "nombre_formulario", "campana"]
BAYESIAN_COLS = [
    "nombre_formulario",
    "campana",
    "origen_creacion",
    "vehiculo_interes",
    "origen",
    "franja_horaria",
]

def main():
    df = pd.read_csv(CLEANED_PATH)
    df = df.drop(columns=["anio_creacion", "subtipo_interes", "plataforma"])

    df["es_fin_de_semana"] = df["dia_semana_creacion"].isin(
        ["sábado", "domingo"]
    ).astype(int)
    df = df.drop(columns=["dia_semana_creacion"])

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

    # Target encoding para concesionario
    X_train["concesionario_target_enc"] = (
        X_train["concesionario"].map(concesionario_means).fillna(global_mean)
    )
    X_train = X_train.drop(columns=["concesionario"])

    # --- MODELO v2: Bayesian Encoding ---
    bayesian_encoders = {}
    for col in BAYESIAN_COLS:
        # Usamos TargetEncoder de category_encoders para Bayesian Encoding
        # con suavizado (smoothing)
        encoder = TargetEncoder(cols=[col], smoothing=SMOOTH_M, min_samples_leaf=1)
        encoder.fit(X_train[col], y_train)
        bayesian_encoders[col] = encoder
        X_train[f"{col}_bayes_enc"] = encoder.transform(X_train[col])
        X_train = X_train.drop(columns=[col])

    # Asegurarse de que no queden columnas 'object' sin codificar
    remaining_object_cols = X_train.select_dtypes(include="object").columns
    if not remaining_object_cols.empty:
        print(f"  AVISO: Columnas categóricas restantes sin codificar: {list(remaining_object_cols)}")
        # Si hay, se podrían eliminar o manejar de otra forma,
        # pero para v2 asumimos que todas las relevantes están en BAYESIAN_COLS
        X_train = X_train.drop(columns=remaining_object_cols)

    training_columns = X_train.columns.tolist()

    artifacts = {
        "known_categories": known_categories,
        "concesionario_means": concesionario_means,
        "global_mean": global_mean,
        "bayesian_encoders": bayesian_encoders, # Guardar los encoders entrenados
        "training_columns": training_columns,
        "umbral": UMBRAL_DECISION,
    }

    joblib.dump(artifacts, OUTPUT_PATH)

    print(f"Artefactos guardados en: {OUTPUT_PATH}")
    print(f"  Features del modelo (v2): {len(training_columns)}")
    print(f"  Concesionarios mapeados: {len(concesionario_means)}")
    print(f"  Encoders Bayesianos guardados: {list(bayesian_encoders.keys())}")
    print(f"  Factor suavizado (m): {SMOOTH_M}")
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
