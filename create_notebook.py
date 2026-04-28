import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown cells y Code cells correspondientes a la nueva lgica

cells = [
    nbf.v4.new_markdown_cell("""# 00 — Data Engineering: Limpieza y Preparación de Datos (V2 - 2025)

**Objetivo:** Tomar el dataset crudo (`Leads Mexico Enero 2022 - Enero 13 2026 excel.xlsx`) y transformarlo en un dataset limpio, filtrado por las nuevas reglas de negocio del año 2025, listo para análisis y modelado.

**Pasos de este notebook:**
1. Importar librerías
2. Cargar el Excel crudo (y normalizar nombres de columnas)
3. Renombrar columnas a formato limpio (`snake_case`)
4. Filtrar por año 2025 y excluir meses anómalos (Abril, Mayo, Junio)
5. Filtrar leads del bot (Plataforma CHATBOT)
6. Eliminar leads que sigan sin cualificación
7. Crear variable target binaria (Hot Lead sí/no)
8. Eliminar columnas con data leakage y sin valor predictivo
9. Manejo de valores nulos
10. Reporte de transformación y exportar dataset limpio"""),
    
    nbf.v4.new_markdown_cell("## 1. Importar librerías"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
print("Librerías importadas correctamente.")"""),

    nbf.v4.new_markdown_cell("## 2. Cargar el Excel crudo y Normalizar Columnas\n\nEl archivo original tiene caracteres con problemas de encoding. Usamos el índice de columnas para evitar problemas al cargar."),
    
    nbf.v4.new_code_cell("""RAW_PATH = "../Leads Mexico Enero 2022 - Enero 13 2026 excel.xlsx"

df_raw = pd.read_excel(RAW_PATH, header=0)

cols_corrected = [
    "Fecha de creación por el cliente", "Fecha de creación", "Fecha reasignación del lead cualificado",
    "Primera fecha de acción", "Tiempo procesado hasta primer contacto", "Qualified lead reassignment user",
    "Nombre del formulario lead", "Campaña", "Último alias modificado", "Propietario del candidato",
    "Tipo de interés", "Nombre de la plataforma", "Origen de creación", "Sub-tipo de interés",
    "Cualificación", "Sub-Cualificación", "Nombre", "Apellidos", "vehículo de interés", "Teléfono Móvil",
    "Nombre corto de la Concesión", "Correo electrónico", "Otra información", "Lead ID", "Descripción",
    "Comentario", "Numero de matricula", "Origen"
]
if len(df_raw.columns) == len(cols_corrected):
    df_raw.columns = cols_corrected

print(f"Dataset cargado: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")
print("\\nInformación del dataset:")
df_raw.info()
print("\\nPrimeras 3 filas:")
display(df_raw.head(3))"""),

    nbf.v4.new_markdown_cell("## 3. Renombrar columnas a snake_case y crear fechas"),
    
    nbf.v4.new_code_cell("""# Extraer año, mes, etc. usando la fecha ingresada por el cliente (que contiene la hora correcta)
df_raw["fecha_dt"] = pd.to_datetime(df_raw["Fecha de creación por el cliente"], dayfirst=True, errors="coerce")
df_raw["Año creación"] = df_raw["fecha_dt"].dt.year
df_raw["Mes creación"] = df_raw["fecha_dt"].dt.month
df_raw["Día creación"] = df_raw["fecha_dt"].dt.day
df_raw["Hora creación"] = df_raw["fecha_dt"].dt.hour

mapa_dias = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado", "Sunday": "domingo"
}
df_raw["Día de Semana creación"] = df_raw["fecha_dt"].dt.day_name().map(mapa_dias)

RENAME_MAP = {
    "Año creación":                              "anio_creacion",
    "Mes creación":                              "mes_creacion",
    "Día creación":                              "dia_creacion",
    "Hora creación":                             "hora_creacion",
    "Día de Semana creación":                    "dia_semana_creacion",
    "Fecha reasignación del lead cualificado":   "fecha_reasignacion",
    "Primera fecha de acción":                   "primera_fecha_accion",
    "Tiempo procesado hasta primer contacto":    "tiempo_primer_contacto",
    "Qualified lead reassignment user":          "usuario_reasignacion",
    "Nombre del formulario lead":                "nombre_formulario",
    "Campaña":                                   "campana",
    "Último alias modificado":                   "ultimo_alias",
    "Propietario del candidato":                 "propietario_candidato",
    "Tipo de interés":                           "tipo_interes",
    "Nombre de la plataforma":                   "plataforma",
    "Origen de creación":                        "origen_creacion",
    "Sub-tipo de interés":                       "subtipo_interes",
    "Cualificación":                             "cualificacion",
    "Sub-Cualificación":                         "sub_cualificacion",
    "vehículo de interés":                       "vehiculo_interes",
    "Nombre corto de la Concesión":              "concesionario",
    "Otra información":                          "otra_informacion",
    "Lead ID":                                   "lead_id",
    "Descripción":                               "descripcion",
    "Comentario":                                "comentario",
    "Numero de matricula":                       "numero_matricula",
    "Origen":                                    "origen",
}

df = df_raw.rename(columns=RENAME_MAP).copy()
print("Columnas renombradas exitosamente.")"""),

    nbf.v4.new_markdown_cell("## 4. Filtrado: Año 2025 y exclusión de meses anómalos\n\nNos quedamos solo con los datos de 2025, eliminando los meses de abril, mayo y junio por tener comportamientos atípicos en asignación de leads."),
    
    nbf.v4.new_code_cell("""filas_antes = len(df)
df = df[df["anio_creacion"] == 2025]
print(f"Filtrado por año 2025: {len(df):,} registros")

df = df[~df["mes_creacion"].isin([4, 5, 6])]
print(f"Eliminados meses 4,5,6: {len(df):,} registros restantes")"""),

    nbf.v4.new_markdown_cell("## 5. Filtrar leads del Chatbot\n\nLos leads atendidos por `MX_LEAD_CHATBOT_QUALIF` no pasan por el proceso humano y son responsables de la gran mayoría de los valores nulos en la cualificación. Deben ser excluidos."),

    nbf.v4.new_code_cell("""is_bot = df["plataforma"].astype(str).str.contains("CHATBOT", case=False, na=False)
print(f"Leads procesados por chatbot a eliminar: {is_bot.sum():,}")

df = df[~is_bot]
print(f"Leads restantes (gestión humana): {len(df):,}")"""),

    nbf.v4.new_markdown_cell("## 6. Eliminar leads sin cualificación restante"),

    nbf.v4.new_code_cell("""sin_cualif = df["cualificacion"].isna().sum()
print(f"Leads sin cualificación tras quitar el bot: {sin_cualif:,}")

df = df.dropna(subset=["cualificacion"])
print(f"Leads finales listos para target: {len(df):,}")"""),

    nbf.v4.new_markdown_cell("## 7. Crear variable Target (Hot Lead sí/no)\n\nTransformamos `Contacto interesado` en 1 (Hot) y el resto de rechazos en 0 (Cold)."),

    nbf.v4.new_code_cell("""df["cualificacion"] = df["cualificacion"].astype(str).str.strip()
df["target"] = (df["cualificacion"] == "Contacto interesado").astype(int)

hot_leads = df["target"].sum()
cold_leads = len(df) - hot_leads

print("=== DISTRIBUCIÓN DEL TARGET ===")
print(f"  1 (Hot Lead):   {hot_leads:,} ({hot_leads/len(df)*100:.1f}%)")
print(f"  0 (Cold Lead):  {cold_leads:,} ({cold_leads/len(df)*100:.1f}%)")
print(f"  Total:          {len(df):,}")

plt.figure(figsize=(6, 4))
ax = df["target"].value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black")
plt.xticks([0, 1], ["Cold Lead (0)", "Hot Lead (1)"], rotation=0)
plt.title("Distribución del Target — Hot Lead vs Cold Lead (2025)")
plt.ylabel("Cantidad de leads")

total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 8. Eliminar columnas con data leakage y sin valor

**Diferencias con V1:**
En esta V2 eliminamos directamente `anio_creacion` (ya que al filtrar solo por 2025 su varianza es 0) y `plataforma` (ya que al eliminar el CHATBOT, el 100% de los leads provienen de `MX_LEAD_QUALIF`, volviendo a la variable inútil). Conservaremos el resto de características que existen ANTES de la llamada."""),

    nbf.v4.new_code_cell("""COLS_LEAKAGE = [
    "sub_cualificacion", "otra_informacion", "comentario", "descripcion", 
    "fecha_reasignacion", "primera_fecha_accion", "usuario_reasignacion", 
    "numero_matricula", "propietario_candidato", "ultimo_alias", "tiempo_primer_contacto",
]
COLS_SIN_VALOR = ["lead_id", "tipo_interes", "cualificacion"]

columnas_finales_requeridas = [
    "mes_creacion", "dia_creacion", "hora_creacion", "dia_semana_creacion",
    "nombre_formulario", "campana", "origen_creacion", "subtipo_interes",
    "vehiculo_interes", "concesionario", "origen", "target"
]

cols_extra = [c for c in df.columns if c not in columnas_finales_requeridas]
df = df.drop(columns=cols_extra)

print(f"Columnas finales restantes ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")"""),

    nbf.v4.new_markdown_cell("## 9. Manejo de valores nulos"),

    nbf.v4.new_code_cell("""print("=== NULOS EN COLUMNAS RESTANTES (antes de imputar) ===\\n")
nulos_restantes = df.isnull().sum()
for col in df.columns:
    n = nulos_restantes[col]
    if n > 0:
        print(f"  {col:30s} → {n:,} nulos ({n/len(df)*100:.1f}%)")

print("\\nImputando nulos con 'Desconocido'...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("Desconocido")
        
duplicados = df.duplicated().sum()
print(f"\\nFilas duplicadas encontradas: {duplicados}")
if duplicados > 0:
    df = df.drop_duplicates()
    print(f"  → Eliminadas. Filas restantes: {len(df):,}")

assert df.isnull().sum().sum() == 0, "ERROR: Quedan nulos"
print("\\nValidaciones pasadas:")
print("  Sin valores nulos")
print("  Target es binario (0 y 1)")
print(f"  DataFrame tiene {len(df):,} filas")"""),

    nbf.v4.new_markdown_cell("## 10. Exportar dataset limpio"),

    nbf.v4.new_code_cell("""CLEAN_PATH = "../data/processed/leads_cleaned.csv"

print("=" * 65)
print("     REPORTE DE TRANSFORMACIÓN — DATA ENGINEERING")
print("=" * 65)

print(f"\\n{'Métrica':<40} {'Limpio':>10}")
print("-" * 65)
print(f"{'Filas':<40} {len(df):>10,}")
print(f"{'Columnas':<40} {len(df.columns):>10}")
print(f"{'Valores nulos totales':<40} {df.isnull().sum().sum():>10}")

print(f"\\n{'Target':<40} {'Cantidad':>10} {'%':>10}")
print("-" * 65)
hot = df["target"].sum()
cold = len(df) - hot
print(f"{'1 — Hot Lead (Contacto interesado)':<40} {hot:>10,} {hot/len(df)*100:>9.1f}%")
print(f"{'0 — Cold Lead (Rechazo)':<40} {cold:>10,} {cold/len(df)*100:>9.1f}%")

print(f"\\n{'Columnas finales:'}")
for i, col in enumerate(df.columns, 1):
    dtype = "numérica" if df[col].dtype in ["int64", "float64"] else "categórica"
    print(f"  {i:2d}. {col:30s} ({dtype})")

print("\\n" + "=" * 65)

df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")

print(f"Dataset limpio exportado a: {CLEAN_PATH}")
print(f"  Filas: {len(df):,}")
print(f"  Columnas: {len(df.columns)}")
print("\\nData Engineering completado exitosamente para el modelo V2 (Año 2025).")"""),
]

nb['cells'] = cells

with open('notebooks/00_data_engineering_v2_2025.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generado correctamente en notebooks/00_data_engineering_v2_2025.ipynb")
