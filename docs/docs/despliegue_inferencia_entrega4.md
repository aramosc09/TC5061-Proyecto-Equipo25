
# Entrega 4 — Estrategia de despliegue e inferencia

## 1. Objetivo

El proyecto MiBici Equipo 25 implementa una API REST para servir el modelo P1 de predicción de demanda de bicicletas por estación y hora.

El objetivo es permitir que un cliente envíe las variables históricas necesarias y reciba una estimación de salidas, sin ejecutar nuevamente el entrenamiento del modelo.

## 2. Inferencia en tiempo real frente a procesamiento por lotes

La inferencia en tiempo real (*online inference*) permite realizar una solicitud a un modelo desplegado y obtener una respuesta durante la misma interacción. Es apropiada cuando una aplicación necesita consultar predicciones de manera individual y oportuna.

Por otro lado, la inferencia por lotes (*batch inference*) procesa conjuntos de registros de forma programada o asíncrona. Resulta conveniente cuando se requieren predicciones masivas y no es necesario recibir cada resultado inmediatamente (Google Cloud, s. f.).

| Criterio | Tiempo real | Por lotes |
|---|---|---|
| Procesamiento | Por solicitud | Conjunto de registros |
| Respuesta | Durante la petición | Al finalizar el trabajo |
| Caso de uso MiBici | Consultar una estación y hora | Generar pronósticos para múltiples estaciones |
| Ventaja principal | Consulta inmediata | Procesamiento masivo |
| Consideración | Requiere un servicio disponible | Requiere planificar y almacenar resultados |

## 3. Estrategia seleccionada para MiBici

Para la Entrega 4 se eligió una API REST con FastAPI y un modelo P1 exportado desde MLflow Model Registry.

La elección permite que un cliente solicite una predicción para una estación y hora específicas sin ejecutar todo el pipeline de experimentación.

El flujo previsto es:

1. El cliente selecciona un registro del conjunto de datos MiBici.
2. Se preparan las 24 variables históricas requeridas.
3. Pydantic valida la estructura de entrada.
4. El cliente envía una petición HTTP al endpoint `/predict`.
5. La API utiliza el modelo P1 para producir una estimación.
6. El cliente presenta la predicción y la compara con el valor observado.

FastAPI facilita la definición de endpoints y la documentación del contrato de la API mediante OpenAPI (FastAPI, s. f.).

Se utiliza el modelo registrado como `mibici-equipo25-p1-demanda`, identificado mediante el alias `champion`. MLflow permite administrar versiones, alias y metadatos de los modelos, lo que contribuye a la trazabilidad del despliegue (MLflow, s. f.).

La copia exportada del modelo se incorpora a la imagen del contenedor. De esta forma, la inferencia no necesita consultar continuamente el servidor de MLflow, aunque sí requiere exportar el modelo antes de construir la imagen.

## 4. Ejemplo reproducible del cliente

El módulo `proyecto_mibici/api/cliente.py` utiliza por defecto los siguientes datos:

- Estación: 519.
- Fecha y hora: `2026-09-10T22:00:00`.
- Conjunto: 2.
- Salidas observadas: 1.
- Variables históricas: 24.

Para validar la petición localmente:

```bash
uv run --frozen python -m proyecto_mibici.api.cliente --solo-validar
```

Para solicitar la predicción cuando la API esté ejecutándose:

```bash
uv run --frozen python -m proyecto_mibici.api.cliente
```

El cliente calcula el error absoluto como:

`error_absoluto = abs(salidas_esperadas - salidas_observadas)`

La validación local fue ejecutada satisfactoriamente. La prueba HTTP real requiere disponer del modelo exportado y de la API en funcionamiento.

## 5. Limitaciones y trabajo futuro

La implementación actual utiliza variables históricas ya preparadas. Por lo tanto, disponer de una API de inferencia en tiempo real no implica que los datos de entrada se actualicen automáticamente en tiempo real.

Para una operación productiva sería necesario implementar un mecanismo de actualización de variables, monitoreo de disponibilidad y latencia, y seguimiento de la calidad predictiva.

También podría implementarse un proceso por lotes para generar periódicamente pronósticos de demanda de todas las estaciones y apoyar la planeación operativa.

## 6. Referencias

FastAPI. (s. f.). *First steps*. https://fastapi.tiangolo.com/tutorial/first-steps/

Google Cloud. (s. f.). *Best practices for implementing machine learning on Google Cloud*. Google Cloud Documentation. https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices

MLflow. (s. f.). *ML Model Registry*. https://mlflow.org/docs/latest/model-registry
