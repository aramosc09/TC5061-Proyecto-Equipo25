# Entrega 2 — Selección y Model Registry del Equipo 25

**Responsabilidad propuesta:** registro, comprobación y documentación del ciclo de vida de P1, P2 y P3. Este archivo describe el procedimiento; **no constituye evidencia de que los modelos ya estén registrados**.

## 1. Qué aporta este desarrollo

- Cada corrida candidata real guarda un modelo de scikit-learn mediante `mlflow.sklearn.log_model`, con `signature` y `input_example` **sin datos personales** (solo características agregadas por estación).
- Se adjuntan `data/data_manifest.json` en `data/` y un resumen de configuración en `config/`, además de los artefactos ya registrados por el proyecto.
- `proyecto_mibici/registry.py` inspecciona las corridas candidatas de `mibici-equipo25`, las ordena **solo por validación**, confirma su procedencia y puede registrar una versión con alias `champion`.
- El registro **requiere aprobación explícita** (`--approve`), no selecciona automáticamente el ganador y verifica que la versión registrada pueda cargarse y que tenga firma antes de cambiar el alias.
- El modelo registrado queda identificado por pregunta, fuente (`run_id`), commit, versión de datos y métrica de validación.

## 2. Antes de comenzar

Se necesita MLflow Tracking local con SQLite y los datos de MiBici procesados, conforme al [README principal](../../README.md). En dos terminales:

**Terminal A:**

```bash
uv sync --frozen
uv run --frozen mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination "file://$PWD/mlartifacts" \
  --host 127.0.0.1 --port 5000
```

**Terminal B:**

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --frozen python -m proyecto_mibici.dataset procesar-datos
```

No se deben publicar datos crudos, Parquet, `mlflow.db`, `mlruns/`, `mlartifacts/` ni modelos serializados en GitHub.

## 3. Ejecutar corridas comparables mediante MLflow Projects

Como evidencia de `nested runs`, es posible ejecutar búsquedas desde el mismo MLflow Project:

```bash
uv run --frozen mlflow run . -e search --env-manager uv --experiment-name mibici-equipo25 -P question=p1 -P model=hgb
uv run --frozen mlflow run . -e search --env-manager uv --experiment-name mibici-equipo25 -P question=p2 -P model=hgb
uv run --frozen mlflow run . -e search --env-manager uv --experiment-name mibici-equipo25 -P question=p3 -P model=kmeans
```
Cada búsqueda genera un padre y trials hijos. El mejor trial debe ejecutarse después como **candidato final** usando `-e train` y sus parámetros ganadores; los trials no se promueven directamente.

Por otro lado, para ejecutar modelos candidatos particulares por pregunta, se puede realizar lo siguiente, donde los parámetros por default se definen en `config/`. También es posible definir los hiperparámetros específicos con los que se desea realizar cada run, según cada modelo:

```bash
# P1: tres candidatos de regresión.
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p1 -P model=poisson
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p1 -P model=hgb
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p1 -P model=rf

# P2: clasificación de presión de flujo (proxy, no eventos reales de desabasto).
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p2 -P model=logistic
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p2 -P model=hgb
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p2 -P model=rf

# P3: perfiles semanales de las estaciones.
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p3 -P model=kmeans
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p3 -P model=agglomerative
uv run --frozen mlflow run . -e train --env-manager uv --experiment-name mibici-equipo25 -P question=p3 -P model=gmm
```



**Importante:** `train` en el código actual también registra métricas de prueba por candidato. El equipo debe seleccionar con **validación**, mantener prueba solo como reporte de desempeño final y explicar cualquier diferencia metodológica con la entrega anterior.

## 4. Inspeccionar candidatos sin cambiar el Registry

```bash
uv run --frozen python -m proyecto_mibici.registry candidates --question p1
uv run --frozen python -m proyecto_mibici.registry candidates --question p2
uv run --frozen python -m proyecto_mibici.registry candidates --question p3
```

Se listan `run_id`, familia, `validation_rmse` (P1), `validation_f1_macro` (P2) o `validation_silhouette` (P3), `model_uri`, commit y versión de datos. Si `model_uri` aparece `null`, el run es anterior a la incorporación de `log_model` y **debe repetirse** con el código nuevo.

Registrar una justificación razonada: mejora frente al baseline, estabilidad temporal, costo computacional, posibles errores de interpretación y limitaciones de P2. No escoger simplemente por el mejor valor de prueba.

## 5. Promoción controlada a `champion`

Primero mostrar una vista previa (no registra ni cambia aliases):

```bash
uv run --frozen python -m proyecto_mibici.registry promote \
  --question p1 --run-id ID_P1 \
  --reason "Menor RMSE de validación; estabilidad y costo aceptables"
```

Una vez que el equipo **apruebe** cuál modelo es el mejor, agregar `--approve` al mismo comando:

```bash
uv run --frozen python -m proyecto_mibici.registry promote \
  --question p1 --run-id ID_P1 \
  --reason "Menor RMSE de validación; estabilidad y costo aceptables" --approve

uv run --frozen python -m proyecto_mibici.registry promote \
  --question p2 --run-id ID_P2 \
  --reason "Mayor F1 macro de validación; revisar recall de presión alta" --approve

uv run --frozen python -m proyecto_mibici.registry promote \
  --question p3 --run-id ID_P3 \
  --reason "Mejor silhouette de validación con grupos interpretables" --approve
```

Sustituir `ID_P1`, `ID_P2` e `ID_P3` por corridas reales. El comando valida pregunta y experimento, finalización del run, etiquetas, métrica de validación, modelo con firma y posibilidad de carga **antes** de asignar el alias `champion`.

Nombres de registro:

- `mibici-equipo25-p1-demanda`
- `mibici-equipo25-p2-presion` (presión **proxy**, no desabasto real)
- `mibici-equipo25-p3-grupos`

## 6. Verificación de las tres versiones activas

```bash
uv run --frozen python -m proyecto_mibici.registry status
```

Cada entrada con estado exitoso debe informar versión, `source_run_id`, `git_commit`, `data_version`, URI y firma presente. La carga por alias usa la forma recomendada por MLflow 3:

```python
import mlflow.pyfunc
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo25-p1-demanda@champion")
```

El alias `champion` representa el **modelo seleccionado para el prototipo académico**, no un sistema de inferencia desplegado en producción real. Si hay versiones nuevas, puede etiquetarse una candidata como `challenger` antes de sustituir `champion`.

## 7. Capturas y evidencias para el reporte

1. Vista del experimento `mibici-equipo25` con baseline y al menos tres candidatos por pregunta.
2. Comparación de métricas con un ejemplo de búsqueda padre/hijos (`nested runs`).
3. Detalle de un candidato con artefactos: modelo con firma/ejemplo, manifiesto, gráficos y configuración.
4. Pantalla de los tres Registered Models y sus versiones `champion`, descripciones y linaje.
5. Salida de `registry status` con las tres URIs `models:/...@champion` cargables.
6. Tabla de resultados y justificación razonada; capturas reales tomadas de la UI, nunca fabricadas.

Las corridas, métricas e imágenes no se pueden comprobar con el ZIP de código: deben ejecutarse en el servidor real del equipo.

## 8. Evidencia personal del integrante

Esta contribución añade `proyecto_mibici/registry.py`, las pruebas `tests/test_model_registry.py`, el guardado de los modelos candidatos con firma y ejemplo en `tracking.py` y la documentación del flujo de promoción. Debe presentarse como Pull Request **de la segunda entrega** y ser revisado por otro integrante; quien lo abre también debe revisar un PR de un compañero.
