# Pendientes por validar con Natalia Posadas

Hallazgos del EDA (`01_exploratory_data_analysis(EDA).ipynb`, sección 5) que requieren validación de negocio antes de continuar.

---

## 1. Origen de creación: Facebook convierte al 99.3%

- Los leads con `origen_creacion = Facebook` tienen una tasa de conversión del **99.3%**, muy por encima del promedio global (68.7%).
- **Pregunta:** ¿Es real esta conversión o hay algún sesgo en cómo se registran los leads de Facebook? ¿Podría ser otro caso de data leakage similar al de `plataforma`?

## 2. Vehículos de interés con conversión ~100%

Varios modelos de vehículo tienen tasas de conversión extremadamente altas (cercanas al 100%):

- **DUSTER** — 100%
- **KOLEOS / KOLEOS 2026** — ~99.6-99.7%
- **LOGAN** — tasa muy alta
- **KWID E-TECH** — tasa muy alta
- **MASTER E-TECH** — tasa muy alta
- **ARKANA / ARKANA HYBRID E-TECH** — ~80-100%
- **KARDIAN** — tasa muy alta
- **STEPWAY** — tasa muy alta
- **KANGOO** — tasa muy alta
- **OROCH** — ~85%

- **Pregunta:** ¿Por qué casi todos los vehículos (excepto KWID estándar al 53.7%) convierten tan alto? ¿Es posible que la definición de "Hot Lead" en el CRM esté sesgada hacia ciertos modelos? ¿O es que KWID atrae un perfil de lead diferente (más exploratorio)?

## 3. Campañas con conversión ~100%

- Varias campañas activas tienen tasas de conversión extremadamente altas (92-100%), como `mx-r-l-wc-newcar-kwid-tbb` (92.2%), `mx-r-l-lg-newcar-kwid-tbb` (99.5%), `mx-r-l-lg-newcar-duster_o` (100%), `OC_RSF_FID` (100%).
- En contraste, `sin_campana` (leads orgánicos) convierte solo al **45.6%**.
- **Pregunta:** ¿Las campañas pagas de Meta/Facebook están generando leads que ya vienen pre-clasificados como Hot por algún proceso automático del CRM? ¿O realmente la segmentación de las campañas es tan efectiva?

## 4. Nombre de formulario: mismo patrón

- Los formularios específicos de vehículo (`MX_Renault_2025_Kwid_Lead`, `MX_Renault_2026_Oroch_Lead`, etc.) convierten al **99-100%**, mientras que el formulario genérico `ONE-PR` convierte solo al **52%**.
- **Pregunta:** ¿Los formularios específicos están vinculados a campañas pagas (Facebook/Meta)? Si es así, la alta conversión podría estar explicada por el canal, no por el formulario en sí. Validar si formulario, campaña y origen son variables redundantes.
