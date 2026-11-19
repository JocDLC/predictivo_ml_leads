# Series temporales (opcional)

Aplicar solo si `.harness/context.json` incluye `ml-timeseries`.
Antes de implementar forecasting, leer `docs/profiles/ml-timeseries.md` y crear
el SDD del caso concreto. Definir horizonte, fecha de disponibilidad de cada
variable, validación cronológica y comparación contra un baseline antes de
elegir modelo. No usar particiones aleatorias de una misma serie ni información
que todavía no estuviera disponible al predecir.

Este perfil no instala dependencias, no entrena y no autoriza enviar datos
de trabajo a servicios externos. Los checks del harness verifican composición;
las pruebas del producto temporal se incorporan al definir el caso.
