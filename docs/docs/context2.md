## Objetivo
Esta actividad impacta con el siguiente objetivo específico de aprendizaje: 

- Implementar prácticas y técnicas que aseguren la reproducibilidad en el entrenamiento de modelos.
Adicional, esta actividad busca:
- Empaquetar el código del proyecto como un MLflow Project que cualquier integrante pueda ejecutar con un solo comando.
- Comparar sistemáticamente múltiples experimentos para P1, P2 y P3 contra la línea base registrada en la entrega anterior.
-Registrar en cada corrida los artefactos que respaldan su resultado, incluida la versión de los datos.
- Llevar el mejor modelo de cada pregunta al Model Registry, gestionando su ciclo de vida hasta el estado de producción.

## Parte A. Proyecto reproducible
1. Configurar un servidor de tracking local con base de datos: El Model Registry requiere un almacenamiento respaldado por base de datos. Levanten un servidor local, por ejemplo mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000, apunten MLFLOW_TRACKING_URI a él y documenten el comando en el README. Agreguen mlflow.db al .gitignore.
2. Crear el archivo MLproject: En la raíz del repositorio, definan un MLflow Project con:
- Un entorno declarado (por ejemplo, python_env.yaml) con las mismas versiones fijas del archivo de dependencias.
- Al menos dos puntos de entrada: ingest, que ejecuta dataset.py, y train, que recibe como parámetros la pregunta (p1, p2 o p3), el modelo y sus hiperparámetros.
- Valores por defecto tomados de config.py, sin repetir rutas, fechas ni semillas.
3. Ejecutar todo con mlflow run: Todas las corridas de esta entrega se lanzan desde el MLflow Project (por ejemplo, mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300), no desde un notebook. Documenten en el README los comandos para reproducir la ingesta y el modelo elegido de cada pregunta desde un clon limpio.

## Parte B. Experimentación y artefactos
4. Ejecutar y comparar experimentos: En el mismo experimento mibici-equipoNN, entrenen para cada pregunta al menos tres versiones adicionales al baseline, variando algoritmos, hiperparámetros o features:
- Usen al menos una búsqueda de hiperparámetros organizada como nested runs: una corrida padre por búsqueda y una corrida hija por combinación.
- Mantengan la misma partición temporal y las mismas métricas de la entrega anterior, para que la comparación contra el baseline sea válida. Si cambian alguna, justifíquenlo en el reporte.
- Agreguen etiquetas que permitan filtrar y comparar en la interfaz, por ejemplo question, model_family y stage_of_work (baseline, search, candidate), además de las de equipo y commit ya obligatorias.
5. Registrar artefactos: En cada corrida candidata, guarden con MLflow los artefactos que respaldan el resultado:
- El modelo con log_model, incluyendo su firma (signature) y un ejemplo de entrada.
- La versión de los datos: el manifiesto generado por la ingesta como artefacto y, de forma recomendada, el conjunto de entrenamiento con mlflow.log_input.
- Gráficas de desempeño acordes a la tarea: por ejemplo, real contra predicho o residuales (P1), matriz de confusión y curva precisión-recall (P2), y mapa o perfiles de los grupos de estaciones (P3).
- Los archivos de configuración con que se ejecutó la corrida.
Organicen los artefactos en subcarpetas (por ejemplo, plots/, data/, config/) y no registren datos personales de los usuarios de MiBici.
6. Seleccionar el mejor modelo: Con base en la comparación de métricas en la interfaz de MLflow, elijan de forma justificada el modelo de cada pregunta que pasará a producción. Consideren el desempeño respecto al baseline, la métrica ligada a la decisión de negocio de cada pregunta, la estabilidad de los resultados y el costo de entrenamiento o inferencia.

## Parte C. Model Registry y reporte
7. Registrar el modelo en el Model Registry: Registren el modelo elegido de cada pregunta con un nombre claro y consistente, por ejemplo mibici-equipoNN-p1-demanda, mibici-equipoNN-p2-riesgo y mibici-equipoNN-p3-grupos. Cada versión debe conservar su linaje: experimento, corrida de origen, commit y versión de datos. Agreguen una descripción a cada modelo registrado.
8. Gestionar su ciclo de vida: Lleven el modelo de cada pregunta al estado de producción:
- Pueden hacerlo mediante el esquema clásico de etapas (transición a Production) o, preferentemente, mediante el esquema vigente de aliases (por ejemplo, el alias champion para la versión en producción y, de forma opcional, challenger para la siguiente mejor).
- Verifiquen que el modelo en producción se puede cargar por su nombre y alias o etapa (por ejemplo, models:/mibici-equipoNN-p1-demanda@champion).
- Documenten en el reporte y en el README qué esquema utilizaron y por qué.