# Monitoreo y reporte de drift en el proyecto

## Instrucciones
En esta actividad integradora, como equipo, dotarán al proyecto de la capacidad de vigilarse a sí mismo. El objetivo es detectar cuándo los datos de producción se alejan de los de entrenamiento y decidir, con evidencia, si una nueva versión del modelo es mejor que la actual.

Realiza, junto con tu equipo, los siguientes pasos:

1. Prepara los conjuntos de datos: Definan un conjunto de referencia (el periodo de datos de MiBici con el que se entrenó el modelo) y un conjunto actual que represente el comportamiento en producción: un periodo posterior de los datos abiertos de MiBici (otro mes, temporada o año). Si su API de la semana 8 registra las peticiones que recibe, también pueden usar esos registros. Documenten qué periodos eligieron y por qué.
2. Genera el reporte de drift con Evidently: Utilicen Evidently para comparar ambos conjuntos y producir un reporte que muestre el data drift por variable y la calidad de los datos.
3. Interpreta los resultados: Analicen el reporte e identifiquen qué variables presentan drift significativo y qué implicaciones tendría para el desempeño del modelo.
4. Entrena una versión alternativa: Generen una segunda versión del modelo (por ejemplo, reentrenada incluyendo datos del periodo actual o con distintos hiperparámetros) y regístrenla en el Model Registry de MLFlow como una nueva versión del mismo modelo.
5. Realiza el A/B testing: Comparen las dos versiones del modelo sobre el mismo conjunto de evaluación del periodo actual, contrastando sus métricas, y determinen de forma justificada cuál debería quedar como versión vigente.
6. Elabora el reporte: Integren en un documento el reporte de drift (con gráficas de Evidently), el resultado del A/B testing y una recomendación sobre si convendría reentrenar el modelo (Continuous Training).

## Criterios de evaluacion
- Define con claridad y precisión los conjuntos de referencia y actual. Genera un reporte completo y estructurado con Evidently, incluyendo gráficas y métricas relevantes.
- Identifica con precisión las variables afectadas por el drift. Analiza de manera detallada las implicaciones para el desempeño del modelo, mostrando un razonamiento profundo.
- Realiza una comparación metodológicamente sólida y bien justificada entre las dos versiones del modelo, detallando métricas y resultados con claridad.
- Presenta una recomendación clara y bien fundamentada, respaldada por los datos y el análisis realizado, demostrando un razonamiento crítico.
- El reporte es claro, bien estructurado y profesional. Evidencia claramente el trabajo colaborativo en Git, con contribuciones balanceadas y organizadas.