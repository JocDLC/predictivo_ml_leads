"""
Genera explicaciones SHAP para leads individuales.
Usa TreeExplainer (optimizado para Random Forest).
"""

import shap
import numpy as np
import pandas as pd

from .inference import load_model_and_artifacts


_explainer = None
_model = None


def get_explainer():
    """Carga el explainer SHAP (singleton para no recalcular)."""
    global _explainer, _model
    if _explainer is None:
        model, _ = load_model_and_artifacts()
        _model = model
        _explainer = shap.TreeExplainer(model)
    return _explainer


def get_model():
    """Retorna el modelo cargado asociado al explainer."""
    global _model
    if _model is None:
        get_explainer()
    return _model


def get_positive_output(values):
    """Extrae la salida positiva o única desde la estructura devuelta por SHAP."""
    if isinstance(values, list):
        selected = values[1] if len(values) > 1 else values[0]
        return np.asarray(selected)

    array_values = np.asarray(values)

    if array_values.ndim == 3:
        class_index = 1 if array_values.shape[2] > 1 else 0
        return array_values[:, :, class_index]

    return array_values


def get_expected_base_value(expected_value):
    """Obtiene el valor base correcto aunque SHAP devuelva un solo output."""
    array_expected = np.asarray(expected_value)

    if array_expected.ndim == 0:
        return float(array_expected)

    flattened = array_expected.flatten()
    class_index = 1 if flattened.size > 1 else 0
    return float(flattened[class_index])


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

    model = get_model()
    shap_values = explainer.shap_values(df_model_row)
    sv = get_positive_output(shap_values)
    if sv.ndim == 2:
        sv = sv[0]
    sv = np.asarray(sv).flatten()

    base_value = get_expected_base_value(explainer.expected_value)

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
    predicted_proba = float(model.predict_proba(df_model_row)[0, 1])

    return explanation_df, base_value, predicted_proba


def translate_feature_name(name):
    """Traduce nombres técnicos de features a lenguaje de negocio."""
    translations = {
        "mes_creacion": "Mes de creación",
        "dia_creacion": "Día del mes",
        "hora_creacion": "Hora de creación",
        "es_fin_de_semana": "¿Fin de semana?",
        "concesionario": "Concesionario (Target Enc.)",
        "concesionario_target_enc": "Concesionario (Target Enc.)",
        "nombre_formulario": "Formulario (Bayesian Enc.)",
        "nombre_formulario_bayes_enc": "Formulario (Bayesian Enc.)",
        "campana": "Campaña (Bayesian Enc.)",
        "campana_bayes_enc": "Campaña (Bayesian Enc.)",
        "vehiculo_interes": "Vehículo (Bayesian Enc.)",
        "vehiculo_interes_bayes_enc": "Vehículo (Bayesian Enc.)",
        "origen": "Canal (Bayesian Enc.)",
        "origen_bayes_enc": "Canal (Bayesian Enc.)",
        "origen_creacion_bayes_enc": "Origen Creación (Bayesian Enc.)",
        "franja_horaria_bayes_enc": "Franja Horaria (Bayesian Enc.)",
    }

    if name in translations:
        return translations[name]

    # Fallback para cualquier otra columna que no esté en el diccionario
    name = name.replace("_", " ").replace("bayes enc", "(Bayesian Enc.)").strip()
    return name.title()
