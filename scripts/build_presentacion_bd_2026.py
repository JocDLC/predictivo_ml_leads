"""Genera la presentación HTML (autocontenida, con gráficos Plotly) para la
reunión de validación de la BD MX 2026 con el encargado de datos.

Lee la BD completa de Salesforce y el exporte de ventas R-Solve, calcula SOLO
agregados (nunca PII) y arma un deck navegable con hallazgos, inconsistencias,
dudas, plan de limpieza y próximos pasos.

Uso:
    python scripts/build_presentacion_bd_2026.py [--bd RUTA.csv] [--rsolve RUTA.csv] [--out salida.html]
"""

from __future__ import annotations

import argparse
import html
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

BASE_DIR = Path(
    r"C:\Users\ax24611\OneDrive - Alliance\@ IA - SOFASA\# SOLUCIONES CON IA"
    r"\PREDICTIVO_MEXICO\BD_HASTA_09-2026"
)
DEFAULT_BD = BASE_DIR / "FULL BD MX 2026 .csv"
DEFAULT_RSOLVE = (
    BASE_DIR / "BD-R-SOLVE-ENE_A_OCT_2026-leads_exportofmatchedsales_2026-10-20 22_30_54.csv"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "presentaciones" / "BD_MX_2026_validacion.html"

HOT = "Contacto interesado"
COL_FECHA = "Fecha de creación por el cliente"
COL_PLAT = "Nombre de la plataforma"
COL_FORM = "Nombre del formulario lead"
COL_ORIGEN = "Origen de creación"
COL_CUAL = "Cualificación"
COL_SUB = "Sub-Cualificación"
COL_ESTADO = "Estado de candidato"
COL_TEL = "Teléfono Móvil"
COL_CONC = "Nombre corto de la Concesión"
COL_VEH = "vehículo de interés"
COL_ID = "Lead ID"

# Paleta
C_HOT, C_COLD, C_NULL = "#F5B700", "#4A4A4A", "#B8B8B8"
C_ALERT, C_OK, C_SALE, C_INFO = "#D7263D", "#2E8B57", "#1B998B", "#2A6F97"

# Campos que se completan después de cualificar → nunca como feature
POST_EVENTO = [
    "Fecha reasignación del lead cualificado", "Primera fecha de acción",
    "Tiempo procesado hasta primer contacto", "Qualified lead reassignment user",
    "Último alias modificado", "Propietario del candidato", "Estado de candidato",
    "Sub-Cualificación", "Comentario", "Otra información",
]
PII = ["Nombre", "Apellidos", "Teléfono Móvil", "Correo electrónico", "Otra información",
       "Descripción", "Comentario", "Numero de matricula"]

# Conocidas por el modelo 2025 (para medir categorías nuevas)
CONCESIONARIOS_2025 = 42


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def load_bd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8", encoding_errors="replace", low_memory=False)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    df["_fecha"] = pd.to_datetime(df[COL_FECHA], dayfirst=True, errors="coerce")
    df["_mes"] = df["_fecha"].dt.to_period("M").astype(str)
    df["_hot"] = df[COL_CUAL].eq(HOT)
    df["_cualificado"] = df[COL_CUAL].notna()
    return df


def load_rsolve(path: Path) -> pd.DataFrame:
    # El archivo trae ';' final en cada fila → index_col=False evita el corrimiento de columnas
    df = pd.read_csv(path, sep=";", encoding="utf-8", encoding_errors="replace",
                     low_memory=False, index_col=False)
    df["_venta"] = pd.to_datetime(df["sales_date_de_vente"], errors="coerce")
    df["_lead"] = pd.to_datetime(df["leads_createddate"], errors="coerce")
    return df


# ----------------------------------------------------------------------------
# Helpers de layout
# ----------------------------------------------------------------------------
def base_layout(fig: go.Figure, title: str | None = None, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=title, height=height, template="plotly_white",
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        font=dict(family="Segoe UI, Arial", size=13),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def kpi(value: str, label: str, tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def fig_volumen_mensual(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("_mes").agg(total=("_hot", "size"), hot=("_hot", "sum"),
                               cual=("_cualificado", "sum")).reset_index()
    g["cold"] = g["cual"] - g["hot"]
    g["sin"] = g["total"] - g["cual"]
    fig = go.Figure([
        go.Bar(name="Hot (Contacto interesado)", x=g["_mes"], y=g["hot"], marker_color=C_HOT),
        go.Bar(name="Cold (Rechazo)", x=g["_mes"], y=g["cold"], marker_color=C_COLD),
        go.Bar(name="Sin cualificar", x=g["_mes"], y=g["sin"], marker_color=C_NULL),
    ])
    fig.update_layout(barmode="stack", yaxis_title="Leads")
    return base_layout(fig, "Volumen mensual de leads 2026 por resultado")


def fig_target_global(df: pd.DataFrame) -> go.Figure:
    vc = df[COL_CUAL].fillna("Sin cualificar").value_counts()
    colors = {HOT: C_HOT, "Rechazo argumentado": C_COLD, "Rechazo no argumentado": "#7A7A7A",
              "Sin cualificar": C_NULL}
    fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.5,
                           marker_colors=[colors.get(k, C_INFO) for k in vc.index],
                           textinfo="label+percent", sort=False))
    fig.update_layout(showlegend=False)
    return base_layout(fig, "Distribución del target (BD completa)")


def fig_hot_por_plataforma(df: pd.DataFrame) -> go.Figure:
    q = df[df["_cualificado"]]
    g = q.groupby(q[COL_PLAT].fillna("(sin plataforma) FIDELE")).agg(
        n=("_hot", "size"), hot=("_hot", "sum")).reset_index()
    g["cold"] = g["n"] - g["hot"]
    g["rate"] = g["hot"] / g["n"]
    fig = go.Figure([
        go.Bar(name="Hot", y=g[COL_PLAT], x=g["hot"], orientation="h", marker_color=C_HOT,
               text=[f"Hot rate {pct(r)}" for r in g["rate"]], textposition="inside"),
        go.Bar(name="Cold", y=g[COL_PLAT], x=g["cold"], orientation="h", marker_color=C_COLD),
    ])
    fig.update_layout(barmode="stack", xaxis_title="Leads cualificados")
    return base_layout(fig, "Hot rate por plataforma de cualificación", height=340)


def fig_hot_por_formulario(df: pd.DataFrame, min_n: int = 200) -> go.Figure:
    q = df[df["_cualificado"]]
    g = q.groupby(COL_FORM).agg(n=("_hot", "size"), hot=("_hot", "mean")).reset_index()
    g = g[g["n"] >= min_n].sort_values("hot")
    colors = [C_ALERT if r >= 0.95 else C_OK for r in g["hot"]]
    fig = go.Figure(go.Bar(
        y=g[COL_FORM], x=g["hot"], orientation="h", marker_color=colors,
        text=[f"{pct(r)}  (n={fmt(n)})" for r, n in zip(g["hot"], g["n"])], textposition="outside",
    ))
    fig.add_vline(x=0.95, line_dash="dash", line_color=C_ALERT,
                  annotation_text="umbral 95% = pre-clasificado", annotation_position="top left")
    fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1.25]), xaxis_title="Hot rate")
    return base_layout(fig, f"Hot rate por formulario (solo cualificados, n ≥ {min_n})", height=560)


def fig_hot_por_origen(df: pd.DataFrame) -> go.Figure:
    q = df[df["_cualificado"]]
    es_bot = q[COL_PLAT].eq("MX_LEAD_CHATBOT_QUALIF")
    n_origen = q.groupby(COL_ORIGEN).size()
    origenes = n_origen[n_origen >= 50].sort_values(ascending=False).index.tolist()

    def resumen(mask: pd.Series) -> pd.DataFrame:
        return q[mask].groupby(COL_ORIGEN)["_hot"].agg(["size", "mean"]).reindex(origenes).fillna(0)

    def barra(nombre: str, g: pd.DataFrame) -> go.Bar:
        return go.Bar(
            name=nombre, x=origenes, y=g["mean"],
            marker_color=[C_ALERT if r >= 0.95 else C_INFO for r in g["mean"]],
            text=[f"{pct(r)}<br>n={fmt(n_)}" if n_ else "" for r, n_ in zip(g["mean"], g["size"])],
            textposition="outside",
        )

    fig = go.Figure([
        barra("Otras plataformas (cualificación humana, FIDELE, sin asignar)", resumen(~es_bot)),
        barra("MX_LEAD_CHATBOT_QUALIF (etiqueta automática)", resumen(es_bot)),
    ])
    fig.update_layout(barmode="group",
                      yaxis=dict(tickformat=".0%", range=[0, 1.25]), yaxis_title="Hot rate")
    return base_layout(fig, "Hot rate por origen de creación, separado por plataforma", height=440)


def fig_deriva_mensual(df: pd.DataFrame) -> go.Figure:
    q = df[df["_cualificado"]]
    g_all = q.groupby("_mes")["_hot"].mean()
    hq = q[q[COL_PLAT].eq("MX_LEAD_QUALIF")]
    g_h = hq.groupby("_mes").agg(rate=("_hot", "mean"), n=("_hot", "size"))
    fig = go.Figure()
    fig.add_bar(name="Leads MX_LEAD_QUALIF (volumen)", x=g_h.index, y=g_h["n"],
                marker_color="#D9E2EC", yaxis="y2")
    fig.add_scatter(name="Hot rate BD completa", x=g_all.index, y=g_all.values,
                    mode="lines+markers+text", text=[pct(v, 0) for v in g_all.values],
                    textposition="top center", line=dict(color=C_ALERT, width=3))
    fig.add_scatter(name="Hot rate MX_LEAD_QUALIF (cualificación humana)", x=g_h.index, y=g_h["rate"],
                    mode="lines+markers+text", text=[pct(v, 0) for v in g_h["rate"]],
                    textposition="bottom center", line=dict(color=C_INFO, width=3))
    fig.update_layout(
        yaxis=dict(title="Hot rate", tickformat=".0%", range=[0, 1.05]),
        yaxis2=dict(title="Leads", overlaying="y", side="right", showgrid=False),
    )
    return base_layout(fig, "Deriva temporal del target: BD completa vs cualificación humana", height=460)


def fig_telefonos(df: pd.DataFrame) -> go.Figure:
    tel = df[COL_TEL].astype(str)
    sci = tel.str.contains(r"E\+", regex=True)
    nulo = df[COL_TEL].isna()
    ok = ~sci & ~nulo
    fig = go.Figure(go.Bar(
        x=["Notación científica<br>(dígitos perdidos)", "Formato válido", "Nulo"],
        y=[sci.sum(), ok.sum(), nulo.sum()], marker_color=[C_ALERT, C_OK, C_NULL],
        text=[pct(sci.mean()), pct(ok.mean()), pct(nulo.mean())], textposition="outside",
    ))
    fig.update_layout(yaxis_title="Leads")
    return base_layout(fig, "Calidad del campo Teléfono Móvil", height=380)


def fig_subcualificacion(df: pd.DataFrame) -> go.Figure:
    vc = df[COL_SUB].value_counts().head(12).sort_values()
    colors = [C_ALERT if k in ("Duplicate", "Duplicado", "Número falso") else C_COLD for k in vc.index]
    fig = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h", marker_color=colors,
                           text=[fmt(v) for v in vc.values], textposition="outside"))
    fig.update_layout(xaxis_title="Leads (solo Rechazo)")
    return base_layout(fig, "Sub-Cualificación: motivos de rechazo (rojo = no son Cold reales)", height=460)


def fig_origen_utm(df: pd.DataFrame) -> go.Figure:
    vc = df["Origen"].value_counts().head(14).sort_values()
    fig = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h", marker_color=C_INFO,
                           text=[fmt(v) for v in vc.values], textposition="outside"))
    fig.update_layout(xaxis_title="Leads")
    return base_layout(fig, "Campo Origen (UTM): sinónimos y mayúsculas sin normalizar", height=460)


def fig_rsolve_conversion_cualificacion(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(df[COL_CUAL].fillna("Sin cualificar")).agg(
        n=("_vendido", "size"), ventas=("_vendido", "sum")).reset_index()
    g["conv"] = g["ventas"] / g["n"]
    order = [HOT, "Rechazo argumentado", "Rechazo no argumentado", "Sin cualificar"]
    g = g.set_index(COL_CUAL).reindex(order).reset_index()
    fig = go.Figure(go.Bar(
        x=g[COL_CUAL], y=g["conv"], marker_color=[C_HOT, C_COLD, "#7A7A7A", C_NULL],
        text=[f"{pct(c, 2)}<br>{fmt(v)} ventas / {fmt(n)} leads" for c, v, n in zip(g["conv"], g["ventas"], g["n"])],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(tickformat=".1%", range=[0, g["conv"].max() * 1.6]),
                      yaxis_title="Tasa de venta real (R-Solve)")
    return base_layout(fig, "Venta real según la Cualificación del equipo", height=430)


def fig_rsolve_conversion_plataforma(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(df[COL_PLAT].fillna("(sin plataforma) FIDELE")).agg(
        n=("_vendido", "size"), ventas=("_vendido", "sum")).reset_index()
    g["conv"] = g["ventas"] / g["n"]
    fig = go.Figure(go.Bar(
        x=g[COL_PLAT], y=g["conv"], marker_color=C_SALE,
        text=[f"{pct(c, 2)}<br>{fmt(v)} / {fmt(n)}" for c, v, n in zip(g["conv"], g["ventas"], g["n"])],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(tickformat=".1%", range=[0, g["conv"].max() * 1.6]),
                      yaxis_title="Tasa de venta real")
    return base_layout(fig, "Venta real por plataforma", height=430)


def fig_rsolve_mensual(df: pd.DataFrame, rs: pd.DataFrame) -> go.Figure:
    leads = df.groupby("_mes").size()
    ventas = df.groupby("_mes")["_vendido"].sum()
    conv = (ventas / leads).fillna(0)
    fig = go.Figure()
    fig.add_bar(name="Ventas R-Solve (por mes de creación del lead)", x=ventas.index, y=ventas.values,
                marker_color=C_SALE)
    fig.add_scatter(name="Tasa de venta", x=conv.index, y=conv.values, yaxis="y2",
                    mode="lines+markers+text", text=[pct(v, 1) for v in conv.values],
                    textposition="top center", line=dict(color=C_ALERT, width=3))
    fig.update_layout(yaxis_title="Ventas",
                      yaxis2=dict(title="Tasa de venta", overlaying="y", side="right",
                                  tickformat=".1%", showgrid=False, range=[0, conv.max() * 1.8]))
    return base_layout(fig, "Ventas reales por mes de creación del lead (los meses recientes aún maduran)", height=430)


def fig_funnel(steps: list[tuple[str, int]]) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=[s for s, _ in steps], x=[n for _, n in steps],
        textinfo="value+percent initial", marker_color=[C_INFO] * (len(steps) - 1) + [C_OK],
    ))
    return base_layout(fig, "Plan de limpieza: leads que quedan después de cada paso", height=520)


def fig_categorias_nuevas(df: pd.DataFrame, forms_2025: set[str]) -> go.Figure:
    n_conc = df[COL_CONC].nunique()
    conc_nuevos = max(n_conc - CONCESIONARIOS_2025, 0)
    form_nuevo = ~df[COL_FORM].isin(forms_2025)
    boreal = df[COL_VEH].astype(str).str.contains("BOREAL", case=False).mean()
    fig = go.Figure(go.Bar(
        x=["Concesionarios nuevos<br>(vs 42 conocidos)", "Leads con formulario<br>desconocido por el modelo",
           "Leads de Boreal<br>(vehículo nuevo)"],
        y=[conc_nuevos / n_conc, form_nuevo.mean(), boreal], marker_color=C_ALERT,
        text=[f"{conc_nuevos} de {n_conc}", pct(form_nuevo.mean()), pct(boreal)], textposition="outside",
    ))
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.15], title="Proporción"))
    return base_layout(fig, "Lo que el modelo 2025 no conoce de la BD 2026", height=400)


# ----------------------------------------------------------------------------
# Contenido estático (dudas, pulido, pasos)
# ----------------------------------------------------------------------------
DUDAS = [
    ("Q1", "¿Desde cuándo y por qué los formularios MX_Renault_2026_* (Facebook) y el chatbot se envían directo como “Contacto interesado” sin cualificación humana? ¿Es un flow de Salesforce o decisión comercial? ¿Sigue así?", "Define si 2/3 de la BD sirven como target."),
    ("Q2", "¿Qué es MX_LEAD_CHATBOT_QUALIF vs MX_LEAD_QUALIF? ¿Colas? ¿Quién las trabaja? ¿Un lead puede pasar de una a otra?", "Es la variable que separa lo confiable de lo no confiable."),
    ("Q3", "El Hot rate humano cayó de ~42% a 18% en septiembre con el volumen triplicado. ¿Cambió el criterio, el equipo o la calidad de leads?", "Deriva del target; septiembre podría estar sub-cualificado."),
    ("Q4", "Ilocalizable / Permanece ilocalizable (13.346): ¿cuántos intentos antes de marcarlo? ¿Cold definitivo o pendiente?", "Decide si es Cold, clase aparte o se excluye."),
    ("Q5", "“Interesado en precio o condición” (9.906) figura como Rechazo. ¿Es Cold por precio o un Hot tibio mal clasificado?", "9.9k leads cambian de clase."),
    ("Q6", "FIDELE / FIDELEHOT / FIDELERSF (10.267, sin plataforma): ¿clientes actuales (recompra)? ¿Entran al predictivo de leads nuevos?", "Población distinta; mezclarla sesga."),
    ("Q7", "¿Criterio para cerrar un lead como Duplicate? ¿Mismo teléfono, email, persona en N días? ¿Regla automática?", "Replicar la detección en la app y excluir bien."),
    ("Q8", "¿Pueden re-exportar directo desde Salesforce (CSV UTF-8, sin Excel) y agregar “Fecha de cualificación”?", "Teléfonos destruidos; falta fecha del resultado."),
    ("Q9", "R-Solve: ¿la llave Lead ID es la oficial? ¿Qué ventana de atribución (30/60/90 días)? ¿Qué significa attributed vs contributed? ¿Por qué 340 leads vendidos no están en la BD 2026?", "Define el target de venta real."),
    ("Q10", "¿Qué son ONE-PR, ONE-BATD, ONE-GR, ONE-FLEET, ONE-REVEAL-BOREAL? ¿“-copy” en Kwid_Lead-copy es un error de Facebook Ads?", "Agrupar por canal real; detectar formularios fantasma."),
    ("Q11", "¿Por qué los concesionarios pasaron de 42 a 103? ¿Nuevos o cambio de nomenclatura?", "Si es nomenclatura, mapear para no perder histórico."),
    ("Q12", "Abr–jun 2025 se excluyeron por anómalos. ¿Hubo algo similar en 2026 (migración, campaña masiva, cambio de CRM)?", "Replicar la exclusión con criterio."),
]

PULIR = [
    ("Teléfonos en notación científica (92,7%)", "Re-exporte directo desde Salesforce, columna como texto. Irrecuperable desde este archivo.", "MX / TI", "Bloqueante"),
    ("Bytes inválidos en el archivo (no es UTF-8 limpio)", "Re-exporte UTF-8 o lectura tolerante documentada.", "MX / TI", "Bloqueante"),
    ("Target inflado por MX_LEAD_CHATBOT_QUALIF (99,9% Hot)", "Confirmar semántica (Q1-Q2). Default: excluir del training, mantener para scoring.", "MX", "Decisión"),
    ("Formularios pre-clasificados (≥95% Hot)", "Regla automática y re-evaluable en cada reentrenamiento; confirmar con Q1.", "Dev + MX", "Decisión"),
    ("24.907 leads sin cualificar (Nuevo / En curso)", "Excluir del training; no son Cold. Reusar como set de scoring.", "Dev", "Definido"),
    ("Duplicate / Duplicado (8.368) y Número falso (805)", "Normalizar etiquetas; sacar del training principal; base para módulos dedup / fake number.", "Dev + MX (Q7)", "Decisión"),
    ("Ilocalizable (13.346) e Interesado en precio (9.906)", "Definir clase (Q4, Q5). Default: Cold.", "MX", "Decisión"),
    ("FIDELE / FIDELEHOT (10.267) sin plataforma", "Definir alcance (Q6). Default: excluir.", "MX", "Decisión"),
    ("Campo Origen (UTM) con sinónimos y mayúsculas", "Normalizar a minúsculas + diccionario de sinónimos.", "Dev", "Definido"),
    ("Campos post-evento (10 columnas)", "Eliminar de las features; conservar solo para auditoría.", "Dev", "Definido"),
    ("PII (8 columnas)", "Separar a tabla hasheada solo para dedup / join; nunca al repo.", "Dev", "Definido"),
    ("61 concesionarios nuevos, Boreal, formularios 2026", "Mapear nomenclatura (Q11) y reentrenar con categorías actuales.", "Dev + MX", "Decisión"),
    ("2 leads Cualificado/Cerrado sin Cualificación", "Error de carga; excluir.", "Dev", "Definido"),
    ("Falta Fecha de cualificación", "Pedir en el re-exporte (Q8) para split temporal correcto.", "MX", "Pedido"),
    ("R-Solve: ';' final, columnas 100% vacías, 410 leads sin match", "Confirmar formato oficial y llave (Q9).", "MX", "Pedido"),
    ("Sin mes como feature + split temporal", "Train ene–jul, valid ago, test sep.", "Dev", "Definido"),
]

PASOS = [
    ("BD 2026 recibida", "done", "06-oct"),
    ("Auditoría BD (hallazgos)", "done", "18-oct"),
    ("Ventas R-Solve recibidas", "done", "21-oct"),
    ("Validación con MX (Q1–Q12)", "now", "hoy"),
    ("Re-exporte limpio", "todo", ""),
    ("Limpieza v3 + reporte", "todo", ""),
    ("Target venta real", "todo", ""),
    ("Reentrenamiento", "todo", ""),
    ("Evaluación", "todo", ""),
    ("Producción", "todo", ""),
]

PROXIMOS = [
    ("Cerrar Q1–Q12 en esta reunión y registrar decisiones", "MX + Asdrúbal", "Hoy"),
    ("Re-exporte Salesforce sin Excel + Fecha de cualificación", "MX / TI", "Esta semana"),
    ("Confirmar llave y ventana de atribución R-Solve; exporte oficial", "MX", "Esta semana"),
    ("Script de limpieza v3 con reporte de calidad por paso", "Dev", "Tras re-exporte"),
    ("Construir target de venta real (Lead ID ↔ R-Solve) y compararlo con Cualificación", "Dev", "Tras Q9"),
    ("Corregir pipeline (bugs B1–B7) y unificar train / inferencia", "Dev", "En paralelo"),
    ("Reentrenar con split temporal; champion vs challenger", "Dev", "Noviembre"),
    ("Monitoreo de deriva mensual y regla automática de leakage", "Dev", "Diciembre"),
]


# ----------------------------------------------------------------------------
# Armado
# ----------------------------------------------------------------------------
def compute_funnel(df: pd.DataFrame) -> list[tuple[str, int]]:
    steps = [("BD completa", len(df))]
    d = df[df["_cualificado"] & ~df[COL_ESTADO].isin(["Nuevo", "En curso"])]
    steps.append(("Sin leads pendientes (Nuevo / En curso / sin cualificar)", len(d)))
    d = d[~d[COL_PLAT].eq("MX_LEAD_CHATBOT_QUALIF")]
    steps.append(("Sin plataforma CHATBOT_QUALIF", len(d)))
    rate = d.groupby(COL_FORM)["_hot"].agg(["mean", "size"])
    pre = set(rate[(rate["mean"] >= 0.95) & (rate["size"] > 100)].index)
    d = d[~d[COL_FORM].isin(pre)]
    steps.append(("Sin formularios pre-clasificados (≥95% Hot)", len(d)))
    d = d[~d[COL_SUB].isin(["Duplicate", "Duplicado", "Número falso"])]
    steps.append(("Sin Duplicate / Número falso", len(d)))
    d = d[~d[COL_ORIGEN].astype(str).str.upper().str.startswith("FIDELE")]
    steps.append(("Sin FIDELE (default, Q6)", len(d)))
    steps.append((f"Dataset entrenable — Hot rate {pct(d['_hot'].mean())}", len(d)))
    return steps


def build(df: pd.DataFrame, rs: pd.DataFrame | None, out: Path) -> dict:
    stats: dict = {}
    n = len(df)
    tel_sci = df[COL_TEL].astype(str).str.contains(r"E\+", regex=True).mean()
    sin_cual = (~df["_cualificado"]).sum()
    q = df[df["_cualificado"]]
    hot_chatbot = q[q[COL_PLAT].eq("MX_LEAD_CHATBOT_QUALIF")]["_hot"].mean()
    hot_humano = q[q[COL_PLAT].eq("MX_LEAD_QUALIF")]["_hot"].mean()
    n_humano = q[COL_PLAT].eq("MX_LEAD_QUALIF").sum()
    forms_2025 = {"ONE-PR", "ONE-GR", "ONE-FLEET", "MX_Renault_0125_Arkana_Lead", "MX_Renault_0125_Oroch_Lead"}
    stats.update(n=n, tel_sci=tel_sci, sin_cual=sin_cual, hot_chatbot=hot_chatbot,
                 hot_humano=hot_humano, n_humano=n_humano)

    hum_origen = q[~q[COL_PLAT].eq("MX_LEAD_CHATBOT_QUALIF")].groupby(COL_ORIGEN)["_hot"].mean()

    def hot_hum(o: str) -> str:
        return pct(hum_origen[o]) if o in hum_origen.index else "sin leads"

    # R-Solve
    rs_block = ""
    if rs is not None:
        vendidos = set(rs["leads_id"].dropna().astype(str))
        df["_vendido"] = df[COL_ID].astype(str).isin(vendidos)
        n_rs_leads = rs["leads_id"].nunique()
        n_match = int(df["_vendido"].sum())
        sin_match = n_rs_leads - n_match
        conv_hot = df.loc[df["_hot"], "_vendido"].mean()
        conv_cold = df.loc[df["_cualificado"] & ~df["_hot"], "_vendido"].mean()
        attrib = rs["contrib_or_attrib"].value_counts()
        stats.update(n_rs_leads=n_rs_leads, n_match=n_match, sin_match=sin_match,
                     conv_hot=conv_hot, conv_cold=conv_cold, n_vin=rs["sales_vin"].nunique())
        rs_block = f"""
<section class="slide" id="rsolve">
  <h2>Hallazgo 5 — Ventas reales (R-Solve): la Cualificación casi no predice la venta</h2>
  <div class="kpis">
    {kpi(fmt(n_rs_leads), "leads con venta en R-Solve (ene–oct 2026)", "sale")}
    {kpi(fmt(n_match), f"cruzan por Lead ID con la BD ({pct(n_match / n)} de los leads)", "sale")}
    {kpi(fmt(rs["sales_vin"].nunique()), f"VIN únicos · {fmt(attrib.get('attributed', 0))} attributed / {fmt(attrib.get('contributed', 0))} contributed")}
    {kpi(fmt(sin_match), "leads vendidos que NO están en la BD 2026 (anteriores a 2026 o faltantes)", "warn")}
  </div>
  <div class="grid2">{to_div(fig_rsolve_conversion_cualificacion(df))}{to_div(fig_rsolve_conversion_plataforma(df))}</div>
  <div class="callout alert"><b>Un lead marcado Hot compra {conv_hot / conv_cold:.1f}× más que uno marcado Cold</b>
  ({pct(conv_hot, 2)} vs {pct(conv_cold, 2)}). Con un target útil esa relación debería ser mucho mayor.
  Confirma que el target definitivo debe ser la <b>venta real</b>, no la cualificación.
  Con venta real, los {fmt(df[COL_PLAT].eq('MX_LEAD_CHATBOT_QUALIF').sum())} leads del chatbot vuelven a ser útiles.</div>
  {to_div(fig_rsolve_mensual(df, rs))}
  <div class="callout info">Inconsistencias del exporte R-Solve: cada fila termina en <code>;</code> (desalinea columnas al leer),
  5 columnas 100% vacías (<code>sales_type</code>, <code>sales_code_model</code>, <code>sales_dealer_id</code>…),
  leads creados desde jul-2025 (fuera del rango de la BD), 70 leads con más de una venta. Ver Q9.</div>
</section>"""

    funnel = compute_funnel(df)
    stats["entrenable"] = funnel[-1][1]

    pasos_html = "".join(
        f'<div class="step {st}"><div class="dot"></div><div class="t">{html.escape(t)}</div><div class="d">{d}</div></div>'
        for t, st, d in PASOS
    )
    dudas_rows = [[f"<b>{q}</b>", html.escape(p), html.escape(w)] for q, p, w in DUDAS]
    pulir_rows = [[html.escape(a), html.escape(b), c, f'<span class="tag {d.lower()}">{d}</span>'] for a, b, c, d in PULIR]
    prox_rows = [[html.escape(a), b, c] for a, b, c in PROXIMOS]
    post_rows = [[c, "Se completa después de cualificar → fuga de información"] for c in POST_EVENTO]

    hoy = date.today().strftime("%d/%m/%Y")
    doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Validación BD MX 2026 — Modelo predictivo de leads</title>
<script>{get_plotlyjs()}</script>
<style>
:root{{--y:#F5B700;--k:#111;--g:#4A4A4A;--bg:#F7F7F5;--alert:#D7263D;--ok:#2E8B57;--sale:#1B998B;--info:#2A6F97}}
*{{box-sizing:border-box}}
html{{scroll-snap-type:y mandatory;scroll-behavior:smooth}}
body{{margin:0;font-family:"Segoe UI",Arial,sans-serif;color:var(--k);background:var(--bg)}}
.slide{{min-height:100vh;scroll-snap-align:start;padding:48px 64px 64px;border-bottom:1px solid #e4e4e0;position:relative}}
.slide:nth-child(even){{background:#fff}}
h1{{font-size:44px;margin:0 0 12px}} h2{{font-size:28px;margin:0 0 20px;border-left:8px solid var(--y);padding-left:14px}}
h3{{margin:18px 0 8px;font-size:18px}}
.sub{{color:var(--g);font-size:18px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0 20px}}
.kpi{{flex:1 1 180px;background:#fff;border:1px solid #e4e4e0;border-top:5px solid var(--y);border-radius:8px;padding:14px 16px}}
.kpi .v{{font-size:30px;font-weight:700}} .kpi .l{{color:var(--g);font-size:13px;margin-top:4px}}
.kpi.warn{{border-top-color:var(--alert)}} .kpi.ok{{border-top-color:var(--ok)}} .kpi.sale{{border-top-color:var(--sale)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.callout{{border-radius:8px;padding:14px 18px;margin:14px 0;font-size:16px;line-height:1.45}}
.callout.alert{{background:#fdecee;border-left:6px solid var(--alert)}}
.callout.info{{background:#e8f1f7;border-left:6px solid var(--info)}}
.callout.ok{{background:#e9f5ee;border-left:6px solid var(--ok)}}
table{{width:100%;border-collapse:collapse;font-size:14px;background:#fff}}
th{{background:var(--k);color:#fff;text-align:left;padding:8px 10px}} td{{padding:7px 10px;border-bottom:1px solid #eee;vertical-align:top}}
tr:hover td{{background:#fffbe8}}
.tag{{padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff;background:var(--g);white-space:nowrap}}
.tag.bloqueante{{background:var(--alert)}} .tag.decisión{{background:var(--y);color:#111}} .tag.definido{{background:var(--ok)}} .tag.pedido{{background:var(--info)}}
.steps{{display:flex;gap:6px;margin:24px 0 36px;align-items:flex-start}}
.step{{flex:1;text-align:center;position:relative}}
.step .dot{{width:22px;height:22px;border-radius:50%;margin:0 auto 8px;background:#ccc;border:4px solid #fff;box-shadow:0 0 0 2px #ccc}}
.step.done .dot{{background:var(--ok);box-shadow:0 0 0 2px var(--ok)}}
.step.now .dot{{background:var(--y);box-shadow:0 0 0 4px rgba(245,183,0,.35);animation:pulse 1.6s infinite}}
.step .t{{font-size:13px;font-weight:600}} .step .d{{font-size:12px;color:var(--g)}}
.step:not(:last-child)::after{{content:"";position:absolute;top:9px;left:calc(50% + 14px);width:calc(100% - 28px);height:3px;background:#ddd}}
.step.done:not(:last-child)::after{{background:var(--ok)}}
@keyframes pulse{{0%{{box-shadow:0 0 0 2px rgba(245,183,0,.9)}}70%{{box-shadow:0 0 0 10px rgba(245,183,0,0)}}100%{{box-shadow:0 0 0 2px rgba(245,183,0,0)}}}}
.nav{{position:fixed;right:18px;bottom:14px;background:#111;color:#fff;padding:6px 12px;border-radius:14px;font-size:13px;opacity:.85}}
.foot{{position:absolute;bottom:16px;left:64px;font-size:12px;color:#999}}
ul.big li{{font-size:17px;margin:8px 0}}
@media print{{html{{scroll-snap-type:none}} .slide{{page-break-after:always;min-height:auto}} .nav{{display:none}}}}
</style></head><body>

<section class="slide" id="portada">
  <div style="margin-top:14vh">
  <div class="sub">Renault México · Modelo predictivo de leads</div>
  <h1>Validación de la BD 2026 para el reentrenamiento</h1>
  <p class="sub">Estado de los datos, inconsistencias encontradas y lo que necesitamos definir juntos.<br>Reunión de validación · {hoy}</p>
  <div class="kpis" style="max-width:900px">
    {kpi(fmt(n), "leads en la BD 2026 (ene → 3 oct)")}
    {kpi(pct(df["_hot"].mean()), "marcados “Contacto interesado”", "warn")}
    {kpi(pct(tel_sci), "teléfonos ilegibles", "warn")}
    {kpi(fmt(stats["entrenable"]), "leads realmente entrenables hoy", "ok")}
  </div></div>
  <div class="foot">Fuente: FULL BD MX 2026 .csv (3-oct-2026) · R-Solve matched sales (20-oct-2026). Solo agregados, sin datos personales.</div>
</section>

<section class="slide" id="pasos">
  <h2>Dónde estamos</h2>
  <div class="steps">{pasos_html}</div>
  <div class="grid2">
    <div class="callout ok"><b>Avances</b><ul class="big">
      <li>Modelo 2025 en producción (Gradient Boosting, ROC-AUC 0,78) con app de scoring y explicación por lead.</li>
      <li>BD 2026 auditada completa: 185 k leads, 30 columnas, 10 hallazgos documentados.</li>
      <li>Ventas reales R-Solve recibidas y cruzadas por Lead ID.</li>
      <li>Plan de limpieza v3 y pipeline de reentrenamiento diseñados.</li></ul></div>
    <div class="callout alert"><b>Qué frena el reentrenamiento</b><ul class="big">
      <li>El target “Cualificación” cambió de significado en 2026 (hallazgos 1–3).</li>
      <li>El exporte pasó por Excel: teléfonos destruidos y bytes inválidos.</li>
      <li>12 definiciones de negocio pendientes (Q1–Q12).</li>
      <li>Llave y ventana de atribución de R-Solve sin confirmar.</li></ul></div>
  </div>
</section>

<section class="slide" id="ficha">
  <h2>Ficha técnica de la BD 2026</h2>
  <div class="kpis">
    {kpi(fmt(n), "leads")} {kpi("30", "columnas")} {kpi(fmt(df[COL_CONC].nunique()), "concesionarios (el modelo conocía 42)", "warn")}
    {kpi("0", "Lead ID duplicados o nulos", "ok")} {kpi(fmt(sin_cual), f"sin cualificar ({pct(sin_cual / n)})")}
    {kpi(df["_fecha"].min().strftime("%d/%m") + " → " + df["_fecha"].max().strftime("%d/%m/%Y"), "rango de fechas")}
  </div>
  {to_div(fig_volumen_mensual(df))}
  <div class="callout info">El volumen se <b>triplica desde junio</b>. Hay que entender si es crecimiento real de campañas o cambio de proceso (Q1, Q3).</div>
</section>

<section class="slide" id="h1">
  <h2>Hallazgo 1 — El target se infló: 69% “Contacto interesado”</h2>
  <div class="grid2">{to_div(fig_target_global(df))}{to_div(fig_hot_por_plataforma(df))}</div>
  <div class="callout alert">En el dataset 2025 limpio el Hot rate era 35–40%. <b>MX_LEAD_CHATBOT_QUALIF marca Hot al {pct(hot_chatbot)}</b> de sus leads: no cualifica, etiqueta.
  Solo <b>MX_LEAD_QUALIF</b> ({fmt(n_humano)} leads, {pct(hot_humano)} Hot) refleja una cualificación humana. En 2025 ya se habían excluido los leads del chatbot por el mismo motivo; en 2026 son 2 de cada 3.</div>
</section>

<section class="slide" id="h2">
  <h2>Hallazgo 2 — Todos los formularios 2026 de Facebook y el chatbot entran pre-clasificados como Hot</h2>
  {to_div(fig_hot_por_formulario(df))}
  <div class="callout alert">En 2025 el problema era solo Arkana y Oroch (se corrigió a mano). En 2026 cubre <b>todos los MX_Renault_2026_*</b> y CHATBOT-WHATSAPP.
  Si se entrena así, el modelo aprende <i>por qué formulario entró el lead</i>, no si quiere comprar.</div>
</section>

<section class="slide" id="h3">
  <h2>Hallazgo 3 — Origen de creación: el ~97% de Facebook y WhatsApp era el chatbot</h2>
  {to_div(fig_hot_por_origen(df))}
  <div class="callout alert">Separando por plataforma se ve que <b>MX_LEAD_CHATBOT_QUALIF etiqueta ~{pct(hot_chatbot)} Hot en todos los orígenes</b>: no es conversión real.
  Con cualificación humana el panorama cambia — ONE {hot_hum("ONE")} · Facebook {hot_hum("Facebook")} · LOCAL MX {hot_hum("LOCAL MX")} · WhatsApp {hot_hum("WhatsApp")} con cualificación humana.
  FIDELE / FIDELEHOT no tienen plataforma asignada y parecen un flujo de recompra (Q6).</div>
</section>

<section class="slide" id="h4">
  <h2>Hallazgo 4 — Deriva temporal: la cualificación humana cae de 42% a 18%</h2>
  {to_div(fig_deriva_mensual(df))}
  <div class="callout alert">La caída de jul → sep coincide con el pico de volumen. Hipótesis: sobrecarga del equipo, cambio de criterio o peor calidad de leads (Q3).
  Según la respuesta, septiembre se pondera, se excluye o se usa tal cual.</div>
</section>
{rs_block}
<section class="slide" id="calidad">
  <h2>Inconsistencias de calidad del archivo</h2>
  <div class="grid2">{to_div(fig_telefonos(df))}{to_div(fig_origen_utm(df))}</div>
  <div class="callout alert"><b>Teléfonos:</b> <code>5,25523E+11</code> en el {pct(tel_sci)} de las filas. Excel convirtió el número y perdió los últimos dígitos. <b>Irrecuperable</b>: imposible deduplicar, detectar números falsos o cruzar por teléfono. Hace falta re-exportar directo desde Salesforce (Q8).</div>
  <div class="callout info"><b>Encoding:</b> el archivo tiene bytes inválidos sueltos; una lectura estricta UTF-8 falla. <b>Origen (UTM):</b> <code>Social-paid</code> vs <code>social_media_paid</code>, <code>Organic_search</code> vs <code>organic</code>, aparece <code>chatgpt.com</code> como fuente nueva. <b>Estado:</b> 2 leads “Cualificado/Cerrado” sin Cualificación.</div>
</section>

<section class="slide" id="sub">
  <h2>Sub-Cualificación: no todo “Rechazo” es un Cold real</h2>
  {to_div(fig_subcualificacion(df))}
  <div class="callout info"><b>Duplicate + Duplicado</b> (dos etiquetas para lo mismo) y <b>Número falso</b> son datos inválidos, no clientes desinteresados → fuera del training, pero son la base perfecta para los módulos de duplicados y números falsos de la app.
  <b>Ilocalizable</b> (Q4) e <b>Interesado en precio</b> (Q5) necesitan definición de negocio.</div>
</section>

<section class="slide" id="nuevas">
  <h2>El modelo 2025 ya está degradado en producción</h2>
  <div class="grid2">
    {to_div(fig_categorias_nuevas(df, forms_2025))}
    <div><h3>Campos post-evento (fuga de información)</h3>{table(["Columna", "Motivo"], post_rows)}
    <p class="sub" style="font-size:14px;margin-top:10px">PII a separar (hasheada, nunca al repo): {", ".join(PII)}.</p></div>
  </div>
  <div class="callout alert">No da error, pero una fracción grande de los leads 2026 se predice con categoría “otros” / media global: concesionarios nuevos, Boreal, formularios 2026, orígenes FIDELE/WhatsApp.</div>
</section>

<section class="slide" id="pulir">
  <h2>Qué necesitamos pulir en los datos</h2>
  {table(["Problema", "Acción propuesta", "Responsable", "Estado"], pulir_rows)}
</section>

<section class="slide" id="dudas">
  <h2>Dudas a resolver en esta reunión (Q1–Q12)</h2>
  {table(["#", "Pregunta", "Por qué importa"], dudas_rows)}
</section>

<section class="slide" id="plan">
  <h2>Plan de limpieza propuesto (con los defaults actuales)</h2>
  <div class="grid2">
    {to_div(fig_funnel(funnel))}
    <div><h3>Reglas</h3><ul class="big">
      <li>Re-exportar sin Excel; separar PII hasheada; quitar columnas constantes y post-evento.</li>
      <li>Leads pendientes (Nuevo / En curso) → set de scoring, no de training.</li>
      <li>Excluir CHATBOT_QUALIF y formularios ≥95% Hot con <b>regla automática</b> en cada reentrenamiento.</li>
      <li>Duplicate / Número falso → datasets aparte para dedup y fake number.</li>
      <li>Normalizar Origen, concesionarios y formularios (<code>-copy</code>).</li>
      <li><b>Split temporal</b> (train ene–jul · valid ago · test sep), sin “mes” como feature.</li>
      <li>Versionar dataset v3 + reporte de limpieza con conteos por paso.</li></ul>
    <div class="callout ok">Quedan ~{fmt(stats["entrenable"])} leads con Hot rate {funnel[-1][0].split('Hot rate ')[1]}: comparable al 2025 (58.546). <b>El crecimiento 2026 fue todo en población no cualificada.</b> Con R-Solve como target, esa población vuelve a servir.</div></div>
  </div>
</section>

<section class="slide" id="prox">
  <h2>Próximos pasos</h2>
  {table(["Acción", "Responsable", "Cuándo"], prox_rows)}
  <h3>Decisiones que necesitamos hoy</h3>
  <ul class="big">
    <li>Target del modelo: <b>venta real R-Solve</b> (recomendado) vs cualificación humana (MX_LEAD_QUALIF) como proxy interino.</li>
    <li>Defaults de clase: Ilocalizable = Cold · Interesado en precio = Cold · FIDELE = fuera · Duplicate / Número falso = fuera.</li>
    <li>Quién y cuándo genera el re-exporte de Salesforce y el exporte oficial de R-Solve (proceso repetible, mensual).</li>
  </ul>
  <div class="foot">Documentos de respaldo: docs/CASO_BD_MX_2026_LIMPIEZA.md · docs/ANALISIS_ESTRATEGICO_2026-10.md · tablero Gantt (proyecto mx-model).</div>
</section>

<div class="nav" id="nav">1 / 1</div>
<script>
const slides=[...document.querySelectorAll('.slide')],nav=document.getElementById('nav');
function cur(){{const y=window.scrollY+innerHeight/2;return Math.max(0,slides.findIndex(s=>s.offsetTop+s.offsetHeight>y));}}
function upd(){{nav.textContent=(cur()+1)+' / '+slides.length;}}
addEventListener('scroll',upd);upd();
addEventListener('keydown',e=>{{const i=cur();
 if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)&&slides[i+1]){{e.preventDefault();slides[i+1].scrollIntoView();}}
 if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)&&slides[i-1]){{e.preventDefault();slides[i-1].scrollIntoView();}}}});
</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bd", type=Path, default=DEFAULT_BD)
    ap.add_argument("--rsolve", type=Path, default=DEFAULT_RSOLVE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = load_bd(args.bd)
    rs = load_rsolve(args.rsolve) if args.rsolve.exists() else None
    stats = build(df, rs, args.out)
    print(f"Presentación generada: {args.out}")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
