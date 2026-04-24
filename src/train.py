"""
Script para entrenar y guardar el modelo predictivo (v2).
Uso: python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Importamos la función de feature engineering desde predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import apply_feature_engineering

CLEANED_PATH = "data/processed/leads_cleaned.csv"
ARTIFACTS_PATH = "models/preprocessing_config.joblib"
MODEL_PATH = "models/best_model.joblib"

def main():
    print("1. Cargando datos y artefactos...")
    df = pd.read_csv(CLEANED_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)

    # Mismas variables eliminadas en el preprocesamiento
    df = df.drop(columns=["anio_creacion", "subtipo_interes", "plataforma"])

    X = df.drop(columns=["target"])
    y = df["target"]

    # Mismo split que en save_artifacts (VITAL para mantener consistencia)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Aplicando Feature Engineering (Target & Bayesian Encoding)...")
    X_train_processed = apply_feature_engineering(X_train.copy(), artifacts)
    X_test_processed = apply_feature_engineering(X_test.copy(), artifacts)

    print(f"   Features finales: {X_train_processed.shape[1]} columnas")

    print("\n3. Entrenando el modelo (Random Forest)...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)

    print("\n4. Evaluando en Test...")
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= artifacts["umbral"]).astype(int)
    print(f"   ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"   Reporte de clasificación (umbral {artifacts['umbral']}):\n")
    print(classification_report(y_test, y_pred))

    print("5. Guardando el modelo y datasets para el reporte...")
    joblib.dump(model, MODEL_PATH)
    X_train_processed.to_csv("data/processed/X_train.csv", index=False)
    X_test_processed.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_frame().to_csv("data/processed/y_train.csv", index=False)
    y_test.to_frame().to_csv("data/processed/y_test.csv", index=False)
    print(f"   ¡Éxito! Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()