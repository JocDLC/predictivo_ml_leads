<!-- converted from Documentación Técnica y Manual del Usuario.docx -->

Documentación Técnica y Manual del Usuario

Nombre del proyecto: Predictivo ML Leads — Clasificación de Hot/Cold Leads para Renault México
Integrantes: José Manuel de la Colina
Tipo de modelo: (Ejemplo: Clasificación binaria supervisada)
Fecha: 4 mayo del 2026
Versión: v1.0
Dataset utilizado: Exportación de Salesforce CRM — Leads México (Enero 2022 – Enero 2026). 906,894 registros crudos; 8,422 registros limpios después del preprocesamiento. Pero se utilizan solo 13.500 para el primero entrenamiento.
Variable objetivo: target (1 = Hot Lead / Contacto interesado, 0 = Cold Lead / No interesado)
URL / Demo: Aplicación local con Streamlit. Ejecutar: streamlit run streamlit_app/app.py https://miapp.streamlit.app

Resumen ejecutivo
Este proyecto desarrolla un modelo de Machine Learning que predice si un lead (prospecto de cliente) generado a través de canales digitales será un Hot Lead (contacto realmente interesado en comprar un vehículo) o un Cold Lead (contacto sin intención real de compra).
Describe:
Qué hace el modelo: Clasifica automáticamente cada lead nuevo de Salesforce como Hot o Cold, asignando una probabilidad de conversión entre 0% y 100%.
Para qué sirve: Permite al equipo comercial de Renault México priorizar los leads con mayor probabilidad de conversión, enfocando el esfuerzo de los concesionarios en los prospectos más prometedores.
A quién impacta: Al equipo de marketing digital y a los concesionarios de Renault México, que reciben los leads para seguimiento comercial.
Qué problema resuelve: 
Actualmente todos los leads se tratan sin distinción. Esto genera sobrecarga de trabajo y pérdida de tiempo en leads que nunca convertirán. El modelo permite filtrar y priorizar, reduciendo esfuerzo en Cold Leads y aumentando la atención en Hot Leads. También permite que la agencia a la que le paga Renault Mexico para que haga campañas publicitarias, ejecutan y configuran las campañas solo con el feedback de cantidad de lead, y resultados de campañas.
Este predictivo permite que la agencia tenga más información (y más rapida

Problema y contexto
3.1 Problema identificado
Renault México recibe miles de leads mensuales a través de formularios web, redes sociales (Facebook, WhatsApp) y campañas digitales. Estos leads se registran en Salesforce CRM y se distribuyen a los concesionarios para seguimiento. Sin embargo, no existe un mecanismo automatizado para evaluar la calidad de cada lead. Los concesionarios invierten el mismo tiempo en leads fríos (bots, formularios incompletos, curiosos) que en leads calientes (personas genuinamente interesadas en comprar). Esto genera ineficiencia operativa y pérdida de oportunidades de venta.
3.2 Justificación del uso de IA
La clasificación de leads depende de múltiples variables interrelacionadas: el canal de origen, el horario de creación, la campaña publicitaria, el vehículo de interés, el concesionario asignado, y el formulario utilizado. Un modelo de Machine Learning puede aprender patrones complejos entre estas variables que serían imposibles de capturar con reglas manuales simples. Además, el volumen de datos (900K+ registros históricos) justifica un enfoque basado en datos para automatizar esta decisión.

Arquitectura del modelo
Flujo del sistema:
Entrada: Archivo Excel exportado desde Salesforce (28 columnas, formato estándar)
→ Preprocesamiento: Parseo de fechas, renombrado de columnas, eliminación de datos personales y columnas con data leakage
→ Limpieza: Filtrado de leads del bot, imputación de nulos, eliminación de duplicados
→ Feature Engineering: Creación de variables derivadas (franja horaria, fin de semana), agrupación de categorías de baja frecuencia, target encoding del concesionario, one-hot encoding
→ Modelo: Random Forest Classifier (200 árboles, profundidad máxima 15)
→ Salida: Clasificación Hot/Cold con probabilidad y explicación SHAP por lead
El pipeline completo se ejecuta mediante el script src/predict.py o a través de la interfaz Streamlit (streamlit_app/app.py). Ambos producen el mismo resultado: un CSV con la predicción y probabilidad para cada lead.

Datos y variables
Fuente del dataset: Exportación directa de Salesforce CRM (Renault México). Archivo Excel con 906,894 registros que cubren el período Enero 2022 – Enero 2026.
Número de registros: 906,894 crudos → 523,366 después de eliminar leads del bot → 8,422 registros limpios del período Dic 2025 – Ene 2026 usados para entrenamiento.
Variables utilizadas (12 después de limpieza):
• mes_creacion — Mes en que se creó el lead (1-12)
• dia_creacion — Día del mes (1-31)
• hora_creacion — Hora de creación (0-23)
• dia_semana_creacion — Día de la semana (lunes a domingo)
• nombre_formulario — Formulario web donde se generó el lead
• campana — Campaña de marketing digital asociada
• origen_creacion — Plataforma de origen (ONE, WhatsApp, Web)
• vehiculo_interes — Vehículo por el que preguntó (KWID, KOLEOS, DUSTER, etc.)
• concesionario — Concesionario asignado al lead
• origen — Canal de adquisición (Organic, Social paid, Referral, etc.)
• es_fin_de_semana — Variable derivada: 1 si fue sábado/domingo
• franja_horaria — Variable derivada: madrugada/mañana/tarde/noche
Variables descartadas (y por qué):
• sub_cualificacion, numero_matricula, fecha_reasignacion, primera_fecha_accion, usuario_reasignacion, propietario_candidato, ultimo_alias, tiempo_primer_contacto — Data leakage: son datos que se asignan DESPUÉS de la clasificación del lead, no estarían disponibles en producción.
• plataforma (MX_LEAD_QUALIF) — Data leakage severo: indica la cola donde se asignan leads ya clasificados como Hot. Concentraba el 41.2% de la importancia del modelo de forma artificial.
• lead_id — Identificador sin valor predictivo.
• tipo_interes — Redundante con subtipo_interes.
• cualificacion — Es la variable objetivo (target), no puede ser feature.
• anio_creacion — Solo 2 valores (2025/2026), sin variabilidad útil.
• subtipo_interes — 96.5% es 'Solicitud de compra', sin poder discriminante.
• Nombre, Apellidos, Teléfono, Correo electrónico, Descripción, Comentario, Otra información — Datos personales o texto libre sin valor predictivo.
Tipo de datos: Numéricos (mes, día, hora) y categóricos nominales (formulario, campaña, vehículo, concesionario, origen). Se aplicó one-hot encoding a las categóricas de baja cardinalidad y target encoding al concesionario (42 categorías).

Modelo de IA utilizado
Algoritmo: Random Forest Classifier (ensamble de 200 árboles de decisión)
Librerías: scikit-learn 1.6.1, pandas 2.3.1, numpy 2.3.1, shap 0.51.0, joblib 1.5.1, streamlit 1.56.0, matplotlib 3.10.3, seaborn 0.13.2
Hiperparámetros: n_estimators=200, max_depth=15, random_state=42 (demás parámetros por defecto de scikit-learn)
Tipo de aprendizaje: Supervisado — Clasificación binaria (Hot Lead = 1, Cold Lead = 0)

Entrenamiento y evaluación
División de datos: 80% entrenamiento (6,737 leads) / 20% prueba (1,685 leads) con estratificación por variable objetivo para mantener la proporción Hot/Cold (68.7%) en ambos conjuntos.
Métricas utilizadas (con umbral de decisión optimizado en 0.35):
Accuracy: 0.88 (88%)
F1-score: 0.92 (clase Hot) / 0.81 (clase Cold)
ROC-AUC: 0.9476
Resultados obtenidos:
• Precision Hot: 0.91 — De cada 100 leads predichos como Hot, 91 realmente lo son.
• Recall Hot: 0.92 — De cada 100 Hot Leads reales, el modelo captura 92.
• Precision Cold: 0.83 — De cada 100 leads predichos como Cold, 83 realmente lo son.
• Recall Cold: 0.80 — De cada 100 Cold Leads reales, el modelo identifica 80.

Se seleccionó un umbral de 0.35 (en lugar del estándar 0.50) para priorizar la captura de Hot Leads. Esto significa que el modelo es más agresivo al clasificar como Hot, reduciendo la pérdida de oportunidades de venta (False Negatives) a costa de enviar algunos Cold Leads adicionales a los concesionarios (False Positives). En el contexto de negocio, perder un Hot Lead es más costoso que atender un Cold Lead.

Interpretabilidad
Variables más importantes (por SHAP):
1. concesionario_target_enc — La tasa histórica de conversión del concesionario asignado es el factor más determinante.
2. hora_creacion — Los leads creados en la madrugada (0-6h) tienen mayor probabilidad de ser Hot.
3. mes_creacion — Hay estacionalidad: ciertos meses concentran más leads calientes.
4. campana (varias) — Las campañas de Facebook para vehículos específicos (KWID, Koleos) influyen significativamente.
5. vehiculo_interes — Vehículos como KWID y DUSTER muestran mayor conversión.
6. es_fin_de_semana — Los leads de fin de semana convierten ligeramente más.
Método utilizado: SHAP (SHapley Additive exPlanations) con TreeExplainer, optimizado para Random Forest. Permite explicar cada predicción individual mostrando qué factores empujaron hacia Hot o Cold.
Explicación del modelo:
El modelo aprende que ciertos concesionarios tienen tasas de conversión históricamente más altas, lo cual es el factor más influyente. Los leads creados durante la madrugada tienden a ser personas genuinamente interesadas (no bots). Las campañas de Facebook para vehículos populares como KWID y Koleos generan leads de mayor calidad. Los leads de fin de semana también muestran mayor interés real. Cada predicción individual puede explicarse con un gráfico SHAP que muestra exactamente qué factores influyeron y en qué dirección.

Uso del modelo
1. El usuario exporta los leads desde Salesforce en formato Excel (.xlsx) con las 28 columnas estándar.
2. El usuario sube el archivo Excel a la aplicación Streamlit o ejecuta el script src/predict.py desde la terminal.
3. El sistema procesa automáticamente cada lead, genera una predicción Hot/Cold con su probabilidad, y muestra los resultados en una grilla interactiva con filtros. Al seleccionar un lead individual, se muestra la explicación SHAP de por qué fue clasificado así.

Implementación técnica
Backend: Python 3.13 con Streamlit como framework de aplicación web
Frontend: Streamlit (interfaz nativa con componentes interactivos: tablas, filtros, gráficos)
Modelo: models/best_model.joblib (Random Forest serializado con joblib) + models/preprocessing_config.joblib (artefactos de preprocesamiento)
Base de datos: No aplica. Los datos se procesan en memoria desde archivos Excel.
Despliegue: Aplicación local ejecutada con 'streamlit run streamlit_app/app.py'. Compatible con Streamlit Cloud para despliegue en la nube.

Resultados y valor
El modelo permite identificar automáticamente el 92% de los Hot Leads con un 91% de precisión. Esto significa que los concesionarios pueden enfocar su esfuerzo en los leads con mayor probabilidad de conversión, reduciendo significativamente el tiempo invertido en prospectos fríos.

Valor cuantificable:
• Reducción estimada del 80% en leads fríos enviados a concesionarios.
• Captura del 92% de oportunidades reales de venta.
• Explicabilidad total: cada predicción se puede justificar con factores de negocio concretos.
• Procesamiento de 500K+ leads en minutos.

Limitaciones
• Datos temporales limitados: El modelo fue entrenado con datos de Diciembre 2025 – Enero 2026. Su rendimiento en datos históricos (2022-2024) baja a ~71% de accuracy, lo cual indica que los patrones de conversión cambian con el tiempo.
• Desbalance de clases: El 68.7% de los leads son Hot, lo que podría sesgar el modelo. Se mitiga parcialmente con el ajuste de umbral.
• Dependencia del formato Salesforce: El script asume las 28 columnas estándar del export de Salesforce. Cambios en el CRM requerirían actualizar el pipeline.
• Concesionarios nuevos: Si aparece un concesionario no visto en entrenamiento, se usa la media global como fallback para el target encoding, lo que puede reducir precisión.
• No captura datos de texto libre: Comentarios y descripciones del lead se descartan. NLP podría extraer señales adicionales.
• Validaciones pendientes: Existe sospecha de data leakage en campañas de Facebook con tasas de conversión del 99-100%. Se requiere validación con el equipo de negocio.

Ética y seguridad
• Datos personales: El modelo NO utiliza datos personales (nombre, apellido, teléfono, correo electrónico) para sus predicciones. Estas columnas se eliminan durante el preprocesamiento.
• Sesgo potencial: El modelo podría favorecer concesionarios con más datos históricos. Se recomienda monitorear la equidad de predicciones entre concesionarios.
• Transparencia: Cada predicción incluye una explicación SHAP que permite auditar la decisión del modelo. No es una 'caja negra'.
• Impacto en personas: Una clasificación incorrecta como Cold podría causar que un prospecto genuino no reciba atención. Por eso se optimizó el umbral (0.35) para minimizar estos casos.
• Almacenamiento: Los datos procesados se mantienen en memoria durante la sesión y no se almacenan permanentemente fuera del CSV de resultados que el usuario descarga.

Manual de uso
Paso 1: Exportar leads desde Salesforce
Desde Salesforce CRM, exportar los leads en formato Excel (.xlsx). El archivo debe contener las 28 columnas estándar, incluyendo: Fecha de creación, Nombre del formulario lead, Campaña, Origen de creación, vehículo de interés, Nombre corto de la Concesión, Cualificación, Lead ID, y Origen.
Paso 2: Ejecutar predicción
Opción A — Interfaz Streamlit:
  1. Abrir terminal en la carpeta del proyecto
  2. Ejecutar: streamlit run streamlit_app/app.py
  3. En el navegador, hacer clic en 'Upload' y seleccionar el archivo Excel
  4. Esperar a que el sistema procese (puede tardar 1-2 minutos para archivos grandes)

Opción B — Línea de comandos:
  1. Ejecutar: python src/predict.py "ruta/al/archivo.xlsx"
  2. El CSV de resultados se genera automáticamente en la misma carpeta
Paso 3: Interpretar resultado
• La grilla muestra cada lead con su predicción (Hot 🔥 / Cold ❄️) y probabilidad.
• Usar los filtros para ver solo Hot Leads, filtrar por vehículo, o ajustar rango de probabilidad.
• Seleccionar un lead individual para ver el gráfico SHAP con los factores que explican la decisión.
• Las barras rojas indican factores que empujan hacia Hot; las azules hacia Cold.
• Descargar el CSV completo con el botón '⬇️ Descargar CSV completo'.

Checklist de entrega
Este documento diligenciado (Documentación Técnica y Manual del Usuario)
Repositorio
Enlace del repositorio: https://github.com/JocDLC/predictivo_ml_leads
Estructura del proyecto:
predictivo_ml_leads/
├── data/
│   ├── processed/           # Datos limpios (leads_cleaned.csv, X_train, X_test, y_train, y_test)
│   └── test/                # Archivos de prueba (100 y 1000 leads de 2025)
├── models/
│   ├── best_model.joblib    # Modelo Random Forest entrenado
│   └── preprocessing_config.joblib  # Artefactos de preprocesamiento
├── notebooks/
│   ├── 00_data_engineering.ipynb         # Limpieza y preparación de datos
│   ├── 01_exploratory_data_analysis.ipynb  # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb      # Ingeniería de features
│   ├── 03_modeling.ipynb                 # Entrenamiento del modelo
│   └── 04_evaluation.ipynb              # Evaluación y SHAP
├── src/
│   ├── predict.py           # Script de inferencia (CLI)
│   └── save_artifacts.py    # Generador de artefactos
├── streamlit_app/
│   ├── app.py               # Aplicación Streamlit principal
│   ├── core/
│   │   ├── inference.py     # Pipeline de inferencia programático
│   │   └── shap_explainer.py  # Explicaciones SHAP
│   └── components/
│       ├── upload.py        # Widget de carga de archivo
│       ├── results_grid.py  # Grilla de resultados con filtros
│       └── lead_detail.py   # Detalle individual con SHAP
├── requirements.txt
├── PENDIENTES_VALIDACION.md
└── README.md

Nota: La estructura debe reflejar claramente separación de responsabilidades (datos, modelo, entrenamiento, inferencia, artefactos).

Video de explicación del modelo (OBLIGATORIO)
El estudiante debe entregar un video corto (2–15 minutos) donde explique:
Problema y objetivo
Qué problema resuelve y por qué es relevante.
Datos
Fuente del dataset
Variables principales
Breve mención del preprocesamiento
Modelo
Algoritmo utilizado
Por qué se eligió
Cómo funciona a alto nivel (sin entrar en teoría extensa)
Resultados
Métricas principales
Qué tan bien funciona el modelo
Interpretabilidad
Variables más importantes
Ejemplo de cómo se toma una decisión (si aplica)
Demostración
Mostrar la app, notebook o ejecución
Ejemplo de predicción en vivo
Conclusiones
Qué aprendió
Limitaciones del modelo
Posibles mejoras
Requisitos técnicos del video:
Duración: 2 a 15 minutos
Audio claro
Pantalla compartida o diapositivas
Enlace (YouTube, Drive, etc.) incluido en el documento

Notas del estudiante:
• Se identificó y corrigió un caso severo de data leakage (plataforma_MX_LEAD_QUALIF) que concentraba el 41.2% de la importancia del modelo. Tras su eliminación, el ROC-AUC solo bajó de 0.9494 a 0.9476 (caída de 0.2%), confirmando que el modelo real funciona correctamente.
• Se ajustó el umbral de decisión de 0.50 a 0.35 para priorizar la captura de Hot Leads (recall), lo cual es más valioso para el negocio que maximizar accuracy.
• Quedan pendientes validaciones de negocio documentadas en PENDIENTES_VALIDACION.md, particularmente sobre campañas de Facebook con tasas de conversión sospechosamente altas (99-100%).
• El pipeline completo es reproducible ejecutando los 5 notebooks en orden (00 a 04) y luego src/save_artifacts.py para regenerar los artefactos de inferencia.
