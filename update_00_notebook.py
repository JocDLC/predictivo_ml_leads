import nbformat as nbf

notebook_path = "notebooks/00_data_engineering_v2_2025.ipynb"

# Leer el notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        if "## 1. Importar librerías" in cell.source:
            cell.source = "## 1. Importar librerías\n\n**Explicación:** Se importan las librerías necesarias. `pandas` para la manipulación de los datos en forma de DataFrames, y `matplotlib`/`seaborn` para la generación de gráficas."
        elif "## 2. Cargar el Excel crudo y Normalizar Columnas" in cell.source:
            cell.source = "## 2. Cargar el Excel crudo y Normalizar Columnas\n\n**Explicación:** El archivo original extraído de Salesforce puede contener saltos de línea extraños o caracteres especiales en los nombres de las columnas que rompen el análisis. En lugar de lidiar con esos caracteres, sobreescribimos los nombres de las columnas usando un arreglo estandarizado con los 28 nombres esperados."
        elif "## 3. Renombrar columnas a snake_case y crear fechas" in cell.source:
            cell.source = "## 3. Renombrar columnas a `snake_case` y crear fechas\n\n**Explicación:** Para evitar errores de sintaxis en Python y facilitar el acceso a las variables (por ejemplo, poder usar `df.campana` en vez de `df['Campaña']`), renombramos todas las columnas a formato `snake_case` (minúsculas y guiones bajos).\nAdemás, a partir de la fecha de creación en texto, generamos columnas numéricas útiles para el modelo predictivo (año, mes, día, hora y día de la semana)."
        elif "## 4. Filtrado: Año 2025 y exclusión de meses anómalos" in cell.source:
            cell.source = "## 4. Filtrado: Año 2025 y exclusión de meses anómalos\n\n**Explicación:** En la Versión 1 mezclábamos varios años. Aquí en la Versión 2 nos quedamos estrictamente con el **año 2025** para modelar el comportamiento más reciente.\nPor recomendación de negocio, eliminamos los meses 4 (Abril), 5 (Mayo) y 6 (Junio) debido a que en esos meses se ingresaron leads masivamente como 'Hot' por un proceso atípico, lo cual engañaría al modelo."
        elif "## 5. Filtrar leads del Chatbot" in cell.source:
            cell.source = "## 5. Filtrar leads del Chatbot\n\n**Explicación:** Los leads cuya plataforma contiene la palabra `CHATBOT` no son gestionados por un vendedor humano, por lo que nunca cambian de estado ni reciben cualificación final. Si los dejamos (como se hizo en la Versión 1), introduciríamos ruido enorme y desbalance al modelo. Aquí los removemos de raíz."
        elif "## 6. Eliminar leads sin cualificación restante" in cell.source:
            cell.source = "## 6. Eliminar leads sin cualificación restante\n\n**Explicación:** A pesar de haber removido el bot, puede haber registros huérfanos que el equipo de ventas aún no ha calificado. Eliminamos aquellos registros nulos en la columna `cualificacion` porque no nos sirven para entrenar (el modelo necesita saber si fue Hot o Cold)."
        elif "## 7. Crear variable Target (Hot Lead sí/no)" in cell.source:
            cell.source = "## 7. Crear variable Target (Hot Lead sí/no)\n\n**Explicación:** Convertimos nuestra variable objetivo categórica en un formato binario que los algoritmos entienden. Asignamos un `1` a todo lo que sea `Contacto interesado` (Hot Lead) y un `0` a cualquier otra cosa (Cold Lead, es decir, rechazos, números equivocados, etc.). Esta celda también imprime el estado del balance final de las clases."
        elif "## 8. Eliminar columnas con data leakage y sin valor" in cell.source:
            cell.source = "## 8. Eliminar columnas con data leakage y sin valor predictivo\n\n**Explicación:** En la Versión 1 se colaron columnas que ocurren *después* del contacto inicial (por ejemplo, el tiempo de primer contacto o alias de quién modificó el registro). Esto se conoce como fuga de datos o **Data Leakage**: si el modelo usa variables del futuro, hará trampa.\nPor eso, definimos `columnas_finales_requeridas` únicamente con datos que existen al instante 0 de crearse el lead."
        elif "## 9. Manejo de valores nulos" in cell.source:
            cell.source = "## 9. Manejo de valores nulos\n\n**Explicación:** Varios algoritmos (como el Random Forest estándar en Scikit-Learn) no soportan campos vacíos (`NaN`). En lugar de eliminar filas completas (perdiendo valiosa información), rellenamos los espacios vacíos con la etiqueta `'Desconocido'`."
        elif "## 10. Exportar dataset limpio" in cell.source:
            cell.source = "## 10. Exportar dataset limpio\n\n**Explicación:** Finalmente, guardamos nuestro dataset purgado y transformado en formato CSV. Este archivo resultante (`leads_cleaned.csv`) será consumido tanto para la Ingeniería de Características (Bayesian Encoding) como para el Entrenamiento del modelo, garantizando que el punto de partida esté totalmente libre de impurezas."
            
    elif cell.cell_type == 'code':
        # Agregar comentarios dentro de los bloques de código
        if "import pandas as pd" in cell.source:
            cell.source = cell.source.replace("import pandas as pd", "# Pandas para manipulación de tablas (DataFrames)\nimport pandas as pd")
            cell.source = cell.source.replace("import matplotlib.pyplot as plt", "# Matplotlib y Seaborn para gráficos\nimport matplotlib.pyplot as plt")
        
        elif "RAW_PATH =" in cell.source:
            cell.source = cell.source.replace("df_raw = pd.read_excel(RAW_PATH, header=0)", "# Leemos el Excel crudo tomando la primera fila como cabecera de las columnas\ndf_raw = pd.read_excel(RAW_PATH, header=0)")
            cell.source = cell.source.replace("cols_corrected =", "# Array manual para asegurar nombres de columnas estandarizados\ncols_corrected =")
            cell.source = cell.source.replace("df_raw.columns = cols_corrected", "# Sustituimos los nombres originales con fallos de encoding por los limpios\n    df_raw.columns = cols_corrected")
            
        elif "df_raw[\"fecha_dt\"] = pd.to_datetime" in cell.source:
            cell.source = cell.source.replace("df_raw[\"fecha_dt\"] = pd.to_datetime", "# 1. Parseamos la fecha texto a un objeto datetime de Pandas\ndf_raw[\"fecha_dt\"] = pd.to_datetime")
            cell.source = cell.source.replace("df_raw[\"Año creación\"]", "# 2. Extraemos el año, mes, día, hora y nombre del día en nuevas columnas para que el modelo pueda buscar estacionalidad\ndf_raw[\"Año creación\"]")
            cell.source = cell.source.replace("RENAME_MAP =", "# 3. Mapeo para traducir las columnas originales con espacios/tildes a 'snake_case'\nRENAME_MAP =")
            
        elif "filas_antes = len(df)" in cell.source:
            cell.source = cell.source.replace("df = df[df[\"anio_creacion\"] == 2025]", "# Nos quedamos únicamente con los leads creados en 2025\ndf = df[df[\"anio_creacion\"] == 2025]")
            cell.source = cell.source.replace("df = df[~df[\"mes_creacion\"].isin([4, 5, 6])]", "# Excluimos los meses 4, 5 y 6 (la virgulilla ~ significa 'not' / 'invertir')\ndf = df[~df[\"mes_creacion\"].isin([4, 5, 6])]")
            
        elif "is_bot =" in cell.source:
            cell.source = cell.source.replace("is_bot = df[\"plataforma\"].astype(str).str.contains(\"CHATBOT\", case=False, na=False)", "# Creamos una máscara booleana: True si en la columna plataforma dice 'CHATBOT'\nis_bot = df[\"plataforma\"].astype(str).str.contains(\"CHATBOT\", case=False, na=False)")
            cell.source = cell.source.replace("df = df[~is_bot]", "# Filtramos el dataset quedándonos solo con los que NO son chatbot (~is_bot)\ndf = df[~is_bot]")
            
        elif "sin_cualif =" in cell.source:
            cell.source = cell.source.replace("df = df.dropna(subset=[\"cualificacion\"])", "# Usamos dropna() específicamente sobre la columna cualificacion para quitar los huérfanos nulos\ndf = df.dropna(subset=[\"cualificacion\"])")
            
        elif "df[\"cualificacion\"] =" in cell.source:
            cell.source = cell.source.replace("df[\"cualificacion\"] = df[\"cualificacion\"].astype(str).str.strip()", "# Limpiamos espacios en blanco al inicio o final para evitar duplicados como 'Contacto ' y 'Contacto'\ndf[\"cualificacion\"] = df[\"cualificacion\"].astype(str).str.strip()")
            cell.source = cell.source.replace("df[\"target\"] =", "# Creamos la columna objetivo: 1 si es 'Contacto interesado', de lo contrario 0\ndf[\"target\"] =")
            
        elif "COLS_LEAKAGE =" in cell.source:
            cell.source = cell.source.replace("COLS_LEAKAGE =", "# Estas columnas son fuga de datos (ocurren después del hecho)\nCOLS_LEAKAGE =")
            cell.source = cell.source.replace("columnas_finales_requeridas =", "# Lista blanca: solo nos quedaremos con estas columnas seguras\ncolumnas_finales_requeridas =")
            cell.source = cell.source.replace("df = df.drop(columns=cols_extra)", "# Descartamos todo lo que no esté en la lista blanca\ndf = df.drop(columns=cols_extra)")
            
        elif "for col in df.columns:" in cell.source:
            cell.source = cell.source.replace("for col in df.columns:", "# Iteramos por cada columna buscando si tiene nulos\nfor col in df.columns:")
            cell.source = cell.source.replace("df[col] = df[col].fillna(\"Desconocido\")", "# Si los tiene, reemplazamos el NaN con la palabra 'Desconocido'\n        df[col] = df[col].fillna(\"Desconocido\")")
            
        elif "CLEAN_PATH =" in cell.source:
            cell.source = cell.source.replace("df.to_csv(CLEAN_PATH, index=False, encoding=\"utf-8\")", "# Guardamos el dataframe en disco sin el índice autogenerado\ndf.to_csv(CLEAN_PATH, index=False, encoding=\"utf-8\")")

# Guardar cambios
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 00 actualizado con éxito.")
