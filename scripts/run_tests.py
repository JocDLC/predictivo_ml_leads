"""
Test suite completo del proyecto predictivo_ml_leads.

Ejecutar con el interprete correcto (que tiene sklearn/scipy):
  & "C:\Program Files\WindowsApps\PythonSoftwareFoundation.PythonManager_26.2.240.0_x64__qbz5n2kfra8p0\python.exe" scripts/run_tests.py
"""
import sys
import io
import json

sys.path.insert(0, "streamlit_app")

errors = []

def ok(msg):
    print(f"  [OK] {msg}")

def err(msg, e):
    print(f"  [ERR] {msg}: {e}")
    errors.append((msg, e))


print("=" * 60)
print("TEST SUITE — predictivo_ml_leads")
print("=" * 60)

# ---- Test 1: Compilación -----------------------------------------------
print("\n[1] Compilacion de modulos")
import py_compile
for f in [
    "streamlit_app/core/inference.py",
    "streamlit_app/core/memory.py",
    "streamlit_app/core/shap_explainer.py",
    "streamlit_app/components/model_report.py",
    "streamlit_app/components/results_grid.py",
    "streamlit_app/components/upload.py",
    "streamlit_app/components/history.py",
    "streamlit_app/components/lead_detail.py",
    "streamlit_app/app.py",
]:
    try:
        py_compile.compile(f, doraise=True)
        ok(f)
    except py_compile.PyCompileError as e:
        err(f, e)


# ---- Test 2: Carga de datos y reporte ----------------------------------
print("\n[2] Carga de datos y reporte")
from components.model_report import (
    load_clean_data,
    load_raw_data,
    load_train_test,
    load_report_artifacts,
)

try:
    df = load_clean_data()
    assert df.shape[0] > 0
    ok(f"load_clean_data: {df.shape}")
except Exception as e:
    err("load_clean_data", e)

try:
    raw = load_raw_data()
    assert raw.shape[0] > 0
    ok(f"load_raw_data: {raw.shape}")
except Exception as e:
    err("load_raw_data", e)

try:
    ds = load_train_test()
    assert ds["X_test"].shape[0] > 0
    ok(f"load_train_test: train={ds['X_train'].shape} test={ds['X_test'].shape}")
except Exception as e:
    err("load_train_test", e)

try:
    rep = load_report_artifacts()
    assert rep["umbral"] > 0
    ok(f"load_report_artifacts: umbral={rep['umbral']}, features={rep['training_context']['n_features']}")
except Exception as e:
    err("load_report_artifacts", e)


# ---- Test 3: Inferencia ------------------------------------------------
print("\n[3] Inferencia con archivos de prueba")
from core.inference import run_inference

test_files = [
    "data/test/test_50_leads.xlsx",
    "data/test/test_100_leads_2025.xlsx",
    "data/test/test_1000_leads_2025.xlsx",
]

for path in test_files:
    try:
        with open(path, "rb") as f:
            uploaded = io.BytesIO(f.read())
        uploaded.name = path
        results, df_model, stats = run_inference(uploaded)
        json.dumps(stats)
        assert len(results) > 0
        assert df_model.shape[0] == len(results)
        labels = set(results["prediccion"].unique())
        assert labels <= {"Hot \U0001f525", "Cold \u2744\ufe0f"}
        ok(f"{path}: n={len(results)}, hot={stats['n_hot']}, cold={stats['n_cold']}, umbral={stats['umbral']}")
    except Exception as e:
        err(path, e)


# ---- Test 4: Memoria SQLite -------------------------------------------
print("\n[4] Memoria SQLite")
from core.memory import init_db, save_session, get_sessions, delete_session

try:
    init_db()
    ok("init_db")
except Exception as e:
    err("init_db", e)

try:
    with open("data/test/test_50_leads.xlsx", "rb") as f:
        uploaded = io.BytesIO(f.read())
    uploaded.name = "test_50_leads.xlsx"
    results, df_model, stats = run_inference(uploaded)
    sid = save_session("test_50_leads.xlsx", results, stats)
    sessions = get_sessions()
    assert len(sessions) > 0
    delete_session(sid)
    ok(f"save/get/delete_session: session_id={sid}")
except Exception as e:
    err("memory_session", e)


# ---- Resultado final ---------------------------------------------------
print()
print("=" * 60)
if not errors:
    print("RESULTADO FINAL: TODOS LOS TESTS PASARON")
else:
    print(f"RESULTADO FINAL: {len(errors)} ERRORES")
    for name, exc in errors:
        print(f"  - {name}: {exc}")
print("=" * 60)
sys.exit(0 if not errors else 1)
