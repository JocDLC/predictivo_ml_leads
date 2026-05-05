# Guion — Video de Presentación del Proyecto
**Predictivo ML Leads — Lead Scoring para Renault México**
**Duración estimada: 10–12 minutos**

> **Nota para el presentador:** El video se graba con pantalla compartida mostrando la app Streamlit. Cada sección indica qué pantalla mostrar. Habla con naturalidad, no leas textual — usa esto como guía de puntos clave.

---

## INTRO (30 seg)
**Pantalla:** App abierta en la página de Predicción (pantalla inicial con logo Renault)

> "Hola, soy [TU NOMBRE]. Este es mi proyecto final de Machine Learning: un sistema de clasificación automática de leads para Renault México. Lo que ven en pantalla es la aplicación funcional que construí — les voy a explicar el problema, cómo lo resolví y les voy a hacer una demostración en vivo."

---

## 1. PROBLEMA Y OBJETIVO (1 min)
**Pantalla:** Sidebar visible con el logo y configuración del modelo

**Puntos a cubrir:**
- "Renault México recibe miles de leads al mes a través de formularios web, plataformas como Facebook, Instagram y otras fuentes digitales."
- "El equipo comercial tiene que contactar a todos estos leads, pero la realidad es que una buena parte son contactos fríos — personas que no tienen intención real de compra."
- "El problema es que no existe un mecanismo automático para separar los leads calientes de los fríos. El equipo dedica el mismo tiempo a todos, lo cual es ineficiente."
- "Mi objetivo fue construir un modelo de Machine Learning que clasifique cada lead como **Hot** o **Cold**, con una probabilidad calibrada, y que además explique *por qué* tomó esa decisión."

---

## 2. DATOS (2 min)
**Pantalla:** Tab "Reporte del Modelo" → pestaña "1. Data Engineering"

**Puntos a cubrir:**
- "Los datos vienen de una exportación del CRM Salesforce de Renault México, con leads del año 2025."
- *(Señalar el funnel de limpieza en pantalla)* "Originalmente teníamos más de 80 mil registros. Tras el pipeline de limpieza me quedé con aproximadamente 58,500 leads."
- "¿Qué se limpió? Se eliminaron leads duplicados, leads del chatbot que no representan un interés real, y leads de dos campañas específicas — Arkana y Oroch — que tenían un problema de data leakage: ya venían pre-clasificados desde la fuente."
- "También se excluyeron tres meses anómalos — abril, mayo y junio 2025 — porque tenían patrones de datos inconsistentes."

**Pantalla:** Cambiar a pestaña "2. EDA"

- "Las variables principales son: el vehículo de interés, el origen del lead, el concesionario, la campaña, el formulario, y variables temporales como la hora y el día de creación."
- *(Señalar alguna gráfica de distribución)* "En el análisis exploratorio se detectaron patrones interesantes: por ejemplo, los leads que llegan de madrugada tienen una tasa de conversión más alta que los que llegan en horario laboral, aunque son menos en volumen."

---

## 3. MODELO (2 min)
**Pantalla:** Tab "Reporte del Modelo" → pestaña "4. Modelado"

**Puntos a cubrir:**
- *(Señalar la tabla de comparación de modelos)* "Se evaluaron cuatro algoritmos: Regresión Logística como baseline, Random Forest, Gradient Boosting y LightGBM."
- "El ganador fue **Gradient Boosting Classifier**, que obtuvo el mejor balance entre ROC-AUC y F1-Score."
- "¿Cómo funciona a alto nivel? Gradient Boosting construye árboles de decisión de manera secuencial. Cada nuevo árbol se enfoca en corregir los errores del anterior. Es como si cada árbol fuera un 'especialista' que aprende de lo que los árboles previos no supieron resolver."
- *(Señalar hiperparámetros si se muestran)* "Los hiperparámetros se seleccionaron mediante una búsqueda con validación cruzada estratificada de 5 folds."

**Pantalla:** Pestaña "3. Feature Engineering"

- "Para las features, se aplicaron transformaciones como la creación de la variable `franja_horaria` a partir de la hora, la codificación One-Hot para variables categóricas de baja cardinalidad, y la codificación del concesionario con Target Encoding."
- "El modelo final trabaja con 22 features derivadas a partir de las columnas originales."

---

## 4. RESULTADOS (1.5 min)
**Pantalla:** Tab "Reporte del Modelo" → pestaña "5. Evaluación"

**Puntos a cubrir:**
- *(Señalar las métricas en pantalla y en el sidebar)* "Las métricas principales en el conjunto de test son:"
  - "**ROC-AUC de 0.78**, lo cual indica buena capacidad de separación entre leads Hot y Cold."
  - "**Recall de 0.70**: de cada 10 leads que realmente son Hot, el modelo detecta 7."
  - "**Precision de 0.56**: de los que el modelo clasifica como Hot, el 56% efectivamente lo son."
  - "**F1-Score de 0.62**: un equilibrio razonable entre no perder oportunidades y no generar falsas alarmas."
- *(Señalar la matriz de confusión)* "En la matriz de confusión se ve que de los 11,710 leads de test, el modelo clasifica correctamente a 5,192 como Cold y 2,943 como Hot. Hay 2,309 falsos positivos y 1,266 falsos negativos."
- "El umbral de clasificación se fijó en 0.325, un punto de compromiso que prioriza no perder leads Hot a costa de aceptar algunos falsos positivos."

---

## 5. INTERPRETABILIDAD (1.5 min)
**Pantalla:** Pestaña "5. Evaluación" → sección SHAP (gráficas de barras y beeswarm)

**Puntos a cubrir:**
- "Una parte fundamental del proyecto es la interpretabilidad. No basta con decir 'este lead es Hot' — el equipo comercial necesita saber *por qué*."
- "Para esto usé SHAP, que es un método que calcula cuánto contribuye cada variable a cada predicción individual."
- *(Señalar el gráfico de barras SHAP)* "A nivel global, las variables más influyentes son el vehículo de interés, la hora de creación, el concesionario y el origen del lead."
- *(Señalar el beeswarm plot)* "En este gráfico cada punto es un lead. El color indica si el valor de la variable es alto o bajo. Se puede ver, por ejemplo, que ciertos vehículos empujan fuertemente hacia Hot, mientras que la hora de creación tiene un efecto diferente según la franja del día."

---

## 6. DEMOSTRACIÓN EN VIVO (2.5 min)
**Pantalla:** Volver a "Predicción de Leads"

### 6a. Carga de archivo
- "Vamos a hacer una predicción en vivo. Voy a subir un archivo Excel con leads reales de prueba."
- *(Arrastrar o seleccionar un archivo de `data/test/`, por ejemplo `test_100_leads_2025.xlsx`)*
- *(Esperar a que procese)* "El sistema ya procesó los leads. Pueden ver el banner de resumen: X leads procesados, Y clasificados como Hot, Z como Cold."

### 6b. Grilla de resultados
- *(Mostrar la tabla de resultados)* "Aquí se ve cada lead con su predicción, probabilidad y clasificación. Puedo filtrar solo los Hot, o los Cold."

### 6c. Detalle individual con SHAP
- *(Hacer clic en un lead Hot)* "Si abro el detalle de un lead específico, puedo ver exactamente qué variables influyeron en su clasificación."
- *(Señalar las barras SHAP del lead)* "Por ejemplo, este lead fue clasificado como Hot con probabilidad X porque el vehículo de interés, la franja horaria y el concesionario empujaron la predicción hacia arriba."
- "Esto le da transparencia completa al equipo comercial — no es una caja negra."

### 6d. Historial
**Pantalla:** Ir a "Historial"

- "Por último, cada sesión de predicción queda guardada automáticamente. Puedo volver a ver sesiones anteriores, comparar la tendencia de porcentaje de leads Hot entre lotes, y exportar los resultados."

---

## 7. CONCLUSIONES (1.5 min)
**Pantalla:** Sidebar visible o pantalla principal

**Puntos a cubrir:**

### Qué aprendí
- "Este proyecto me enseñó que la parte más difícil de un proyecto de ML no es el modelo en sí, sino la limpieza y comprensión de los datos. Detectar y manejar el data leakage, entender por qué ciertos meses eran anómalos, y decidir qué variables excluir requirió mucha más iteración que el entrenamiento del modelo."
- "También aprendí la importancia de la interpretabilidad: un modelo que nadie entiende, nadie lo usa."

### Limitaciones
- "El modelo tiene limitaciones. La precisión de 56% significa que hay un porcentaje significativo de falsos positivos — leads que el modelo dice que son Hot pero no lo son."
- "El modelo fue entrenado con datos de 2025, por lo que si el comportamiento de los leads cambia con el tiempo, necesitará re-entrenamiento."
- "Actualmente se ejecuta localmente — no hay despliegue en la nube."

### Posibles mejoras
- "A futuro se podría explorar un despliegue cloud para que el equipo comercial lo use directamente."
- "Se podría incorporar más datos históricos o datos de interacción post-formulario."
- "Y se podría implementar un sistema de re-entrenamiento periódico para que el modelo se adapte a cambios en el comportamiento de los leads."

---

## CIERRE (15 seg)
- "Eso es todo. Gracias por su atención. El código completo está en GitHub y la documentación técnica detallada está incluida en la entrega."

---

## Checklist pre-grabación
- [ ] App corriendo (`streamlit run streamlit_app/app.py`)
- [ ] Archivo de prueba listo en `data/test/` (recomendado: `test_100_leads_2025.xlsx`)
- [ ] Sidebar visible con métricas
- [ ] Micrófono probado (audio claro)
- [ ] Resolución de pantalla legible (1920x1080 o similar)
- [ ] Modo claro u oscuro consistente durante todo el video
- [ ] No mostrar rutas locales con datos personales
