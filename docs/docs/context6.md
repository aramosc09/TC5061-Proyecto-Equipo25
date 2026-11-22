# CI/CD y FinOps

## Objetivo

Consolidar el proyecto integrador del equipo como un sistema operable:

- Orquestar sus servicios con Docker Compose.
- Automatizar pruebas y despliegue con un pipeline de CI/CD en GitHub Actions.
- Evaluar su desempeño bajo carga con Locust.
- Realizar un análisis de FinOps que estime el costo por inferencia del sistema.

Esta actividad corresponde a una fase de entrega del proyecto del curso (Entregable 7): a partir de la API contenerizada de la Semana 8 y de las versiones del modelo registradas en MLFlow durante la Semana 9, el equipo construye la arquitectura completa de operación, la automatiza y la analiza desde la perspectiva de desempeño y costo.

## Instrucciones

Realiza, junto con tu equipo, los siguientes pasos:

1. Orquesta los servicios con Docker Compose: Creen un archivo docker-compose.yml que levante los servicios del proyecto (por ejemplo, la API del modelo y el servidor de MLFlow) y que permita ejecutarlos juntos con un solo comando. Pueden usar Docker Compose o podman-compose. Agregar Redis y un worker de RQ para procesar predicciones de forma asíncrona es opcional.
2. Configura el pipeline de CI/CD: Implementen un workflow de GitHub Actions que, al hacer push o pull request, ejecute automáticamente las pruebas del proyecto y construya la imagen del servicio.
3. Ejecuta pruebas de carga con Locust: Diseñen un escenario de prueba que simule múltiples usuarios haciendo peticiones al endpoint de predicción, y generen un reporte con gráficas de desempeño (latencia, peticiones por segundo, tasa de error).
4. Monitorea el uso de recursos: Con herramientas como docker stats o podman stats, observen el consumo de CPU y memoria de sus contenedores durante la prueba de carga.
5. Realiza el análisis FinOps: A partir del consumo de recursos y del desempeño observado, estimen el costo por inferencia del sistema (por ejemplo, con base en un precio de referencia de una instancia en la nube) y discutan cómo optimizarlo.
6. Documenta y reporta: Integren en un reporte los resultados de la prueba de carga (con gráficas) y el análisis FinOps, y actualicen el README con las instrucciones para levantar el sistema y ejecutar el pipeline.