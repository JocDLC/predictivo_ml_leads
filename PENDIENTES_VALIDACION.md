# Pendientes por validar con el equipo comercial

Hallazgos del EDA que requieren validación de negocio antes de continuar.

---

## ✅ RESUELTO — Leads pre-clasificados como Hot (Arkana y Oroch)

> **Fecha de resolución:** Abril 2026  
> **Validado con:** Equipo comercial Renault México

### Hallazgo
Los formularios `MX_Renault_0125_Arkana_Lead` (67.4% Hot, n=2,981) y `MX_Renault_0125_Oroch_Lead` (87.1% Hot, n=954) tenían tasas de conversión anormalmente altas comparadas con el mismo vehículo vía `ONE-PR` (+36.6pp y +35.2pp respectivamente).

### Confirmación del equipo
**Sí, durante 2025 las campañas de Facebook para Arkana y Oroch se enviaban directamente como Hot Leads al concesionario**, sin pasar por cualificación humana. Esto constituye data leakage: el `target=1` no refleja intención de compra real.

### Acción tomada
- **Eliminados del dataset de entrenamiento** en el notebook `00_data_engineering_v2_2025.ipynb` (paso 8).
- Los formularios de Koleos (36.8%), Megane (18.2%), Kardian (67.0%) y Kangoo (65.0%) se conservan porque **sí pasaban por cualificación estándar**.
- Se documentó en los notebooks de Data Engineering, EDA y Feature Engineering.

### Hallazgo adicional: `nombre_formulario` y `campana` son redundantes
Cada formulario `MX_Renault_*` tiene **exactamente 1 campana** (relación 1:1). Son la misma variable con diferente nombre. Se recomienda evaluar si eliminar una de las dos para reducir multicolinealidad.

---

## Pendientes abiertos (V1 — contexto histórico)

Los siguientes puntos fueron identificados en la V1 del EDA. Algunos ya están resueltos por los cambios de la V2 (eliminación del chatbot, filtrado por 2025), pero se conservan como referencia.

### 1. Origen de creación: Facebook convierte al 99.3% (V1)

- En la V1, los leads con `origen_creacion = Facebook` tenían una tasa de conversión del **99.3%**.
- **Estado:** Parcialmente explicado. En la V2 (datos 2025 sin chatbot) los orígenes de Facebook se normalizaron. La alta conversión de V1 estaba inflada por la mezcla con datos del chatbot.

### 2. Vehículos de interés con conversión ~100% (V1)

- En la V1, DUSTER, KOLEOS, LOGAN, etc. tenían tasas cercanas al 100%.
- **Estado:** Parcialmente resuelto. En la V2, al filtrar solo 2025 y eliminar Arkana/Oroch pre-clasificados, las tasas se normalizan. KWID sigue siendo el vehículo con tasa más baja (36.3% via ONE-PR), lo cual es un patrón real de negocio (vehículo de entrada, leads más exploratorios).

### 3. Campañas con conversión ~100% (V1)

- **Estado:** Explicado parcialmente por el proceso de pre-clasificación de Arkana/Oroch. Las campañas restantes tienen tasas consistentes con la operación real.

### 4. Nombre de formulario = campaña (redundancia)

- Confirmado: cada formulario `MX_Renault_*` tiene 1 sola campaña asociada.
- **Acción pendiente:** Evaluar si eliminar `nombre_formulario` o `campana` como feature del modelo para evitar redundancia.
