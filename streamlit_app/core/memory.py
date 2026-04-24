"""
Sistema de memoria persistente con SQLite.

Guarda sesiones de predicción y resultados entre reinicios del servidor.
"""

import os
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

import pandas as pd

# Ruta de la base de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "memory", "history.db")


def _ensure_db_dir():
    """Crea el directorio de la DB si no existe."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def _get_conn():
    """Context manager para conexiones SQLite."""
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Inicializa las tablas de la base de datos si no existen."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                file_name   TEXT NOT NULL,
                total_leads INTEGER NOT NULL,
                n_hot       INTEGER NOT NULL,
                n_cold      INTEGER NOT NULL,
                pct_hot     REAL NOT NULL,
                umbral      REAL NOT NULL,
                accuracy    REAL,
                stats_json  TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                lead_id     TEXT,
                prediccion  TEXT NOT NULL,
                prob_hot    REAL NOT NULL,
                vehiculo    TEXT,
                campana     TEXT,
                concesionario TEXT,
                origen      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_session
                ON predictions(session_id);
        """)


def save_session(file_name: str, results: pd.DataFrame, stats: dict) -> int:
    """
    Guarda una sesión completa (metadatos + predicciones individuales).

    Args:
        file_name: Nombre del archivo Excel procesado.
        results:   DataFrame con resultados de run_inference().
        stats:     Dict con estadísticas del proceso.

    Returns:
        session_id: ID de la sesión guardada.
    """
    init_db()

    n_hot   = stats.get("n_hot", 0)
    n_cold  = stats.get("n_cold", 0)
    total   = n_hot + n_cold
    pct_hot = round((n_hot / total * 100) if total > 0 else 0, 1)

    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (created_at, file_name, total_leads, n_hot, n_cold, pct_hot, umbral, accuracy, stats_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_name,
                total,
                n_hot,
                n_cold,
                pct_hot,
                stats.get("umbral", 0.35),
                stats.get("accuracy"),
                json.dumps(stats, default=str),
            ),
        )
        session_id = cur.lastrowid

        # Insertar predicciones individuales (batch)
        rows = []
        for _, row in results.iterrows():
            rows.append((
                session_id,
                str(row.get("lead_id", "")) if "lead_id" in results.columns else None,
                row.get("prediccion", ""),
                float(row.get("probabilidad_hot", 0)),
                row.get("vehiculo_interes", None),
                row.get("campana", None),
                row.get("concesionario", None),
                row.get("origen", None),
            ))

        conn.executemany(
            """INSERT INTO predictions
               (session_id, lead_id, prediccion, prob_hot, vehiculo, campana, concesionario, origen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    return session_id


def get_sessions(limit: int = 50) -> pd.DataFrame:
    """
    Retorna el historial de sesiones ordenadas por fecha descendente.

    Args:
        limit: Máximo de sesiones a retornar.

    Returns:
        DataFrame con columnas de metadatos de sesión.
    """
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT id, created_at, file_name, total_leads,
                      n_hot, n_cold, pct_hot, umbral, accuracy
               FROM sessions
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def get_session_predictions(session_id: int) -> pd.DataFrame:
    """
    Retorna todas las predicciones de una sesión específica.

    Args:
        session_id: ID de la sesión.

    Returns:
        DataFrame con predicciones individuales.
    """
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT lead_id, prediccion, prob_hot, vehiculo, campana, concesionario, origen
               FROM predictions
               WHERE session_id = ?
               ORDER BY prob_hot DESC""",
            (session_id,),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def get_trend_data() -> pd.DataFrame:
    """
    Retorna datos agregados por sesión para gráficos de tendencia.

    Returns:
        DataFrame con created_at, pct_hot, total_leads, file_name.
    """
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT created_at, file_name, total_leads, n_hot, pct_hot, umbral
               FROM sessions
               ORDER BY id ASC
               LIMIT 30"""
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def delete_session(session_id: int):
    """Elimina una sesión y sus predicciones asociadas."""
    init_db()
    with _get_conn() as conn:
        conn.execute("DELETE FROM predictions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
