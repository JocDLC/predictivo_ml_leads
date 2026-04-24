"""
Genera explicaciones SHAP para leads individuales.
Usa TreeExplainer (optimizado para Random Forest).
"""

import shap
import numpy as np
import pandas as pd

from .inference import load_model_and_artifacts


_explainer = None


def get_explainer():
    """Carga el explainer SHAP (singleton para no recalcular)."""
    global _explainer
    if _explainer is None:
        model, _ = load_model_and_artifacts()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_lead(df_model_row, top_n=10):
    """
    Genera explicación SHAP para un lead individual.

    Args:
        df_model_row: Serie o DataFrame de 1 fila con las 48 features.
        top_n: Número de features más importantes a mostrar.

    Returns:
        explanation_df: DataFrame con columnas [feature, valor, impacto, direccion]
        base_value: Valor base del modelo (probabilidad promedio)
        predicted_proba: Probabilidad predicha para este lead
    """
    explainer = get_explainer()

    if isinstance(df_model_row, pd.Series):
        df_model_row = df_model_row.to_frame().T

    shap_values = explainer.shap_values(df_model_row)

    # Para clasificación binaria, clase positiva (Hot) = índice 1
    ev = explainer.expected_value
    if isinstance(shap_values, list):
        # SHAP < 0.42: lista de arrays [clase_0, clase_1]
        sv = np.array(shap_values[1]).flatten()
        base_value = float(ev[1])
    elif np.ndim(shap_values) == 3:
        # SHAP >= 0.44: array 3D (n_samples, n_features, n_classes)
        sv = shap_values[0, :, 1]
        base_value = float(ev[1])
    else:
        sv = np.array(shap_values).flatten()
        base_value = float(ev) if np.ndim(ev) == 0 else float(ev[1])

    feature_names = df_model_row.columns.tolist()
    feature_values = df_model_row.values[0]

    indices_sorted = np.argsort(np.abs(sv))[::-1][:top_n]

    rows = []
    for idx in indices_sorted:
        idx = int(idx)
        impact = sv[idx]
        name = feature_names[idx]
        value = feature_values[idx]

        # Traducir nombre técnico a descripción legible
        readable_name = translate_feature_name(name)
        direction = "Favorece Hot 🔥" if impact > 0 else "Favorece Cold ❄️"

        rows.append({
            "feature": readable_name,
            "feature_raw": name,
            "valor": value,
            "impacto": round(impact, 4),
            "impacto_abs": round(abs(impact), 4),
            "direccion": direction,
        })

    explanation_df = pd.DataFrame(rows)
    predicted_proba = base_value + sv.sum()

    return explanation_df, base_value, predicted_proba


def translate_feature_name(name):
    """Traduce nombres técnicos de features a lenguaje de negocio."""
    translations = {
        "mes_creacion": "Mes de creación",
        "dia_creacion": "Día del mes",
        "hora_creacion": "Hora de creación",
        "es_fin_de_semana": "¿Fin de semana?",
        "concesionario_target_enc": "Concesionario (Target Enc.)",
        "nombre_formulario_bayes_enc": "Formulario (Bayesian Enc.)",
        "campana_bayes_enc": "Campaña (Bayesian Enc.)",
        "origen_creacion_bayes_enc": "Origen Creación (Bayesian Enc.)",
        "vehiculo_interes_bayes_enc": "Vehículo (Bayesian Enc.)",
        "origen_bayes_enc": "Canal (Bayesian Enc.)",
        "franja_horaria_bayes_enc": "Franja Horaria (Bayesian Enc.)",
    }

    if name in translations:
        return translations[name]

    # Fallback para cualquier otra columna que no esté en el diccionario
    name = name.replace("_", " ").replace("bayes enc", "(Bayesian Enc.)").strip()
    return name.title()
