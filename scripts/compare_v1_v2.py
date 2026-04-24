"""Compare OLD model (v1, one-hot) vs NEW model (v2, contribution + bayesian)."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data/processed"
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print(f"Features v2: {X_train.shape[1]}")
print(f"Columnas: {list(X_train.columns)}\n")

# --- OLD METRICS (from 03_modeling.ipynb output, v1 with 48 one-hot features) ---
old_metrics = {
    "Logistic Regression": {"ACC": 0.8979, "PREC": 0.9386, "REC": 0.9110, "F1": 0.9246, "AUC": 0.9441},
    "Random Forest":       {"ACC": 0.9003, "PREC": 0.9443, "REC": 0.9084, "F1": 0.9260, "AUC": 0.9476},
    "Gradient Boosting":   {"ACC": 0.9015, "PREC": 0.9525, "REC": 0.9015, "F1": 0.9263, "AUC": 0.9464},
}

# --- TRAIN NEW MODELS on v2 features ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=10, random_state=42
    ),
}

sep = "=" * 90
print(sep)
header = f"{'Modelo':25s} | {'Metrica':>9s} | {'OLD v1':>8s} | {'NEW v2':>8s} | {'Cambio':>8s}"
print(header)
print(sep)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    new = {
        "ACC": accuracy_score(y_test, y_pred),
        "PREC": precision_score(y_test, y_pred),
        "REC": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
    }
    old = old_metrics[name]

    for metric in ["ACC", "PREC", "REC", "F1", "AUC"]:
        diff = new[metric] - old[metric]
        arrow = "+" if diff >= 0 else ""
        print(f"  {name:23s} | {metric:>9s} | {old[metric]:>8.4f} | {new[metric]:>8.4f} | {arrow}{diff:>7.4f}")
    print("-" * 90)

# --- Feature importances for Random Forest v2 ---
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print()
print("Top 10 features - Random Forest v2:")
for feat, imp in importances.head(10).items():
    print(f"  {feat:45s} {imp:.4f}")

# --- Key comparison: KWID behavior ---
print()
print("=" * 60)
print("COMPARACION CLAVE: Sesgo de vehiculos")
print("=" * 60)
print()
print("OLD v1 (one-hot): vehiculo_interes_KWID tenia importancia 0.2429")
print("  -> El modelo penalizaba KWID porque su tasa (53%) < global (69%)")
print()
kwid_features = [c for c in X_train.columns if "vehiculo" in c]
print(f"NEW v2 features de vehiculo: {kwid_features}")
for f in kwid_features:
    print(f"  {f}: importancia = {importances.get(f, 0):.4f}")
