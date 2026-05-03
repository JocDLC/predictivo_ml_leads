# Graph Report - predictivo_ml_leads  (2026-04-30)

## Corpus Check
- 23 files · ~73,340,020 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 152 nodes · 209 edges · 18 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]

## God Nodes (most connected - your core abstractions)
1. `run_inference()` - 12 edges
2. `render_prediction_page()` - 9 edges
3. `render_model_report()` - 9 edges
4. `_get_conn()` - 9 edges
5. `predict()` - 8 edges
6. `init_db()` - 8 edges
7. `explain_lead()` - 8 edges
8. `render_history_page()` - 7 edges
9. `load_report_artifacts()` - 7 edges
10. `read_excel()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `read_excel()`  [INFERRED]
  src\data_engineering_2025.py → src\predict.py
- `run_inference()` --calls--> `read_excel()`  [INFERRED]
  streamlit_app\core\inference.py → src\predict.py
- `main()` --calls--> `apply_feature_engineering()`  [INFERRED]
  src\train.py → streamlit_app\core\inference.py
- `render_prediction_page()` --calls--> `run_inference()`  [INFERRED]
  streamlit_app\app.py → streamlit_app\core\inference.py
- `render_prediction_page()` --calls--> `save_session()`  [INFERRED]
  streamlit_app\app.py → streamlit_app\core\memory.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (24): main(), apply_feature_engineering(), clasificar_franja(), clean_nulls(), extract_date_features(), find_column(), fix_encoding(), get_expected_features() (+16 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (17): Componente de detalle individual con explicación SHAP., Permite seleccionar un lead por búsqueda de ID (todos) o dropdown (filtrados)., Renderiza gráfico de barras horizontal con impactos SHAP estilo UI moderna., Renderiza el detalle de un lead con explicación SHAP., render_lead_detail(), render_lead_selector(), render_shap_chart(), Componente de grilla de resultados con filtros. (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (20): _feature_importance_series(), _find_column(), _format_model_name(), get_positive_shap_explanation(), load_clean_data(), load_raw_data(), load_report_artifacts(), load_train_test() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.19
Nodes (17): delete_session(), _ensure_db_dir(), _get_conn(), get_session_predictions(), get_sessions(), get_trend_data(), init_db(), Sistema de memoria persistente con SQLite.  Guarda sesiones de predicción y re (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.25
Nodes (12): apply_feature_engineering(), clean_nulls(), extract_date_features(), find_column(), fix_encoding(), get_expected_features(), Pipeline de inferencia programático. Reutiliza la lógica de src/predict.py adap, Pipeline completo: archivo Excel → DataFrame con predicciones.      Args: (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.21
Nodes (13): explain_lead(), get_expected_base_value(), get_explainer(), get_model(), get_positive_output(), Genera explicaciones SHAP para leads individuales. Usa TreeExplainer (optimizad, Traduce nombres técnicos de features a lenguaje de negocio., Carga el explainer SHAP (singleton para no recalcular). (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.27
Nodes (9): Componente de historial de sesiones — Sistema de memoria SQLite.  Renderiza el, Renderiza gráfico de % Hot leads a lo largo del tiempo., Página completa del historial de sesiones., Renderiza la tabla de sesiones históricas.      Returns:         session_id s, Permite seleccionar una sesión y ver sus predicciones., render_history_page(), render_session_detail(), render_sessions_table() (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.5
Nodes (1): Guarda los artefactos de preprocesamiento necesarios para inferencia. Ejecutar

### Community 8 - "Community 8"
Cohesion: 0.67
Nodes (2): apply_theme(), Inyecta CSS personalizado compatible con los temas nativo de Streamlit.      No

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Compare OLD model (v1, one-hot) vs NEW model (v2, contribution + bayesian).

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Aplica feature engineering: derivadas, agrupación, encoding.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Pipeline completo: Excel crudo → predicciones.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Regenera X_train/X_test aplicando el pipeline V2 desde leads_cleaned.csv.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Aplica márgenes y altura consistentes a todas las figuras Plotly.

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Carga el explainer SHAP (singleton para no recalcular).

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Genera explicación SHAP para un lead individual.      Args:         df_model_

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Traduce nombres técnicos de features a lenguaje de negocio.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Aplica estilos CSS globales a la aplicación Streamlit, con toggle personalizado.

## Knowledge Gaps
- **56 isolated node(s):** `Compare OLD model (v1, one-hot) vs NEW model (v2, contribution + bayesian).`, `Predicción de Hot/Cold Leads a partir de un archivo Excel de Salesforce.  Uso:`, `Corrige encoding latino corrupto en nombres de columna.`, `Clasifica hora en franja horaria.`, `Busca una columna por palabras clave (case-insensitive).` (+51 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (4 nodes): `clasificar_franja()`, `main()`, `save_artifacts.py`, `Guarda los artefactos de preprocesamiento necesarios para inferencia. Ejecutar`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (3 nodes): `apply_theme()`, `Inyecta CSS personalizado compatible con los temas nativo de Streamlit.      No`, `theme.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (2 nodes): `compare_v1_v2.py`, `Compare OLD model (v1, one-hot) vs NEW model (v2, contribution + bayesian).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Aplica feature engineering: derivadas, agrupación, encoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Pipeline completo: Excel crudo → predicciones.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Regenera X_train/X_test aplicando el pipeline V2 desde leads_cleaned.csv.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Aplica márgenes y altura consistentes a todas las figuras Plotly.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Carga el explainer SHAP (singleton para no recalcular).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Genera explicación SHAP para un lead individual.      Args:         df_model_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Traduce nombres técnicos de features a lenguaje de negocio.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Aplica estilos CSS globales a la aplicación Streamlit, con toggle personalizado.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `render_prediction_page()` connect `Community 1` to `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.408) - this node is a cross-community bridge._
- **Why does `run_inference()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.380) - this node is a cross-community bridge._
- **Why does `save_session()` connect `Community 3` to `Community 1`?**
  _High betweenness centrality (0.236) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `run_inference()` (e.g. with `render_prediction_page()` and `read_excel()`) actually correct?**
  _`run_inference()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `render_prediction_page()` (e.g. with `render_upload()` and `run_inference()`) actually correct?**
  _`render_prediction_page()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Compare OLD model (v1, one-hot) vs NEW model (v2, contribution + bayesian).`, `Predicción de Hot/Cold Leads a partir de un archivo Excel de Salesforce.  Uso:`, `Corrige encoding latino corrupto en nombres de columna.` to the rest of the system?**
  _56 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._python -m streamlit run streamlit_app/app.py
  