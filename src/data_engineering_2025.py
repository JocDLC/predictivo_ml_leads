import pandas as pd
import os

RAW_EXCEL_PATH = "Leads Mexico Enero 2022 - Enero 13 2026 excel.xlsx"
CLEANED_CSV_PATH = "data/processed/leads_cleaned.csv"

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

COLS_LEAKAGE = [
    "sub_cualificacion",        
    "otra_informacion",         
    "comentario",               
    "descripcion",              
    "fecha_reasignacion",       
    "primera_fecha_accion",     
    "usuario_reasignacion",     
    "numero_matricula",         
    "propietario_candidato",    
    "ultimo_alias",             
    "tiempo_primer_contacto",   
]

COLS_SIN_VALOR = [
    "lead_id",                  
    "tipo_interes",             
    "cualificacion",            
]

def main():
    print("1. Cargando Excel original (esto tomará unos minutos)...")
    # Usamos las posiciones de columnas para evitar problemas de encoding con los acentos
    df_raw = pd.read_excel(RAW_EXCEL_PATH, header=0)
    
    # Normalizar los nombres de columnas para que coincidan con RENAME_MAP ignorando acentos raros
    cols_corrected = [
        "Fecha de creación por el cliente", "Fecha de creación", "Fecha reasignación del lead cualificado",
        "Primera fecha de acción", "Tiempo procesado hasta primer contacto", "Qualified lead reassignment user",
        "Nombre del formulario lead", "Campaña", "Último alias modificado", "Propietario del candidato",
        "Tipo de interés", "Nombre de la plataforma", "Origen de creación", "Sub-tipo de interés",
        "Cualificación", "Sub-Cualificación", "Nombre", "Apellidos", "vehículo de interés", "Teléfono Móvil",
        "Nombre corto de la Concesión", "Correo electrónico", "Otra información", "Lead ID", "Descripción",
        "Comentario", "Numero de matricula", "Origen"
    ]
    
    # Asignamos temporalmente columnas si el count coincide (28)
    if len(df_raw.columns) == len(cols_corrected):
        df_raw.columns = cols_corrected
        
    print(f"   Filas iniciales: {len(df_raw):,}")

    # Extraer año y mes usando "Fecha de creación"
    df_raw["fecha_dt"] = pd.to_datetime(df_raw["Fecha de creación"], dayfirst=True, errors="coerce")
    df_raw["Año creación"] = df_raw["fecha_dt"].dt.year
    df_raw["Mes creación"] = df_raw["fecha_dt"].dt.month
    df_raw["Día creación"] = df_raw["fecha_dt"].dt.day
    df_raw["Hora creación"] = df_raw["fecha_dt"].dt.hour
    df_raw["Día de Semana creación"] = df_raw["fecha_dt"].dt.day_name()

    print("\n2. Renombrando columnas a snake_case...")
    df = df_raw.rename(columns=RENAME_MAP).copy()
    
    # Asegurar que existan anio_creacion y mes_creacion
    print("\n3. Filtrando dataset...")
    # a. Solo año 2025
    filas_antes = len(df)
    df = df[df["anio_creacion"] == 2025]
    print(f"   - Filtrado año 2025: pasamos de {filas_antes:,} a {len(df):,}")
    
    # b. Eliminar abril, mayo, junio (meses atípicos)
    filas_antes = len(df)
    df = df[~df["mes_creacion"].isin([4, 5, 6])]
    print(f"   - Eliminado abr/may/jun: pasamos de {filas_antes:,} a {len(df):,}")
    
    # c. Eliminar leads procesados por Chatbot
    filas_antes = len(df)
    is_bot = df["plataforma"].astype(str).str.contains("CHATBOT", case=False, na=False)
    df = df[~is_bot]
    print(f"   - Eliminado Chatbot: pasamos de {filas_antes:,} a {len(df):,}")
    
    # d. Eliminar leads que sigan sin cualificación
    filas_antes = len(df)
    df = df.dropna(subset=["cualificacion"])
    print(f"   - Eliminado sin cualificacion: pasamos de {filas_antes:,} a {len(df):,}")

    print("\n4. Creando variable Target (1 = Hot, 0 = Cold)...")
    df["cualificacion"] = df["cualificacion"].astype(str).str.strip()
    df["target"] = (df["cualificacion"] == "Contacto interesado").astype(int)
    
    hot_leads = df["target"].sum()
    cold_leads = len(df) - hot_leads
    print(f"   - Hot Leads: {hot_leads:,} ({hot_leads/len(df)*100:.1f}%)")
    print(f"   - Cold Leads: {cold_leads:,} ({cold_leads/len(df)*100:.1f}%)")

    print("\n5. Eliminando columnas de data leakage y sin valor...")
    # Solo mantener las columnas a borrar si existen
    cols_to_drop = [c for c in COLS_LEAKAGE + COLS_SIN_VALOR if c in df.columns]
    
    # También eliminar las columnas que no vamos a usar (como Nombre, Apellidos, etc.)
    columnas_finales_requeridas = [
        "anio_creacion", "mes_creacion", "dia_creacion", "hora_creacion", "dia_semana_creacion",
        "nombre_formulario", "campana", "plataforma", "origen_creacion", "subtipo_interes",
        "vehiculo_interes", "concesionario", "origen", "target"
    ]
    
    cols_extra = [c for c in df.columns if c not in columnas_finales_requeridas]
    df = df.drop(columns=cols_extra)
    print(f"   - Columnas finales: {len(df.columns)}")
    print(f"   - Lista: {list(df.columns)}")
    
    print("\n6. Manejo de Nulos en predictoras...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Desconocido")

    print("\n7. Guardando dataset procesado...")
    os.makedirs(os.path.dirname(CLEANED_CSV_PATH), exist_ok=True)
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"   ¡Éxito! Dataset final guardado en {CLEANED_CSV_PATH} ({len(df):,} filas)")

if __name__ == "__main__":
    main()
