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

    # Para clasificación binaria, shap_values[1] = clase positiva (Hot)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    feature_names = df_model_row.columns.tolist()
    feature_values = df_model_row.values[0]

    indices_sorted = np.argsort(np.abs(sv))[::-1][:top_n]

    rows = []
    for idx in indices_sorted:
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
        "concesionario_target_enc": "Concesionario (tasa histórica)",
    }

    if name in translations:
        return translations[name]

    prefixes = {
        "dia_semana_creacion_": "Día: ",
        "nombre_formulario_": "Formulario: ",
        "campana_": "Campaña: ",
        "origen_creacion_": "Origen creación: ",
        "vehiculo_interes_": "Vehículo: ",
        "origen_": "Canal: ",
        "franja_horaria_": "Franja: ",
    }

    for prefix, label in prefixes.items():
        if name.startswith(prefix):
            return label + name[len(prefix):]

    return name
