# MiBici ML · Equipo 25

Sistema integrador de Machine Learning y MLOps sobre datos abiertos del sistema de bicicleta pública **MiBici** (Área Metropolitana de Guadalajara). El proyecto abarca de extremo a extremo la ingesta reproducible, ingeniería de variables sin fuga temporal, experimentación con MLflow Projects, versionado en Model Registry, despliegue contenerizado con FastAPI y Docker Compose, CI/CD con GitHub Actions, pruebas de carga con Locust, análisis FinOps, monitoreo de drift con Evidently, evaluación A/B y documentación de gobernanza.

**Curso:** TC5061 — Operaciones de Aprendizaje Automático (MLOps) · **Equipo 25**  
**Documentación clave:** [Model Card (P1)](docs/docs/model_card.md) · [Arquitectura](references/arquitectura_propuesta_mibici.mmd) · [ML Canvas](references/ml_canvas_mibici.pdf) · [Operación y FinOps](docs/docs/operacion_entrega6.md)

---

## 1. Propósito y preguntas operativas

El usuario principal es el equipo de rebalanceo y logística de MiBici. Sobre 36 meses de viajes abiertos (octubre de 2023 a septiembre de 2026) se abordan tres preguntas operativas:

| Pregunta | Tarea y unidad | Objetivo implementado | Horizonte / uso esperado |
| :--- | :--- | :--- | :--- |
| **P1 — Demanda** *(modelo servido)* | Regresión por estación y hora | Predecir viajes iniciados (salidas); baseline de persistencia semanal (rezago de 168 h). | Siguiente hora. Apoya la planeación de flota antes de la demanda. |
| **P2 — Presión de flujo** | Clasificación por estación y hora | Clasificar dirección e intensidad del flujo (7 clases: sin presión neta o extracción/acumulación baja, moderada o alta). Baseline de clase mayoritaria. | Siguiente hora. Orienta verificaciones operativas, no órdenes automáticas de rebalanceo. |
| **P3 — Perfiles de estaciones** | Agrupamiento por estación | Agrupar estaciones por perfiles semanales normalizados de salidas y llegadas (coordenadas reservadas solo para visualización). | Descripción histórica de patrones de uso para planeación de cobertura. |

> **Limitación de P2:** Los datos abiertos contienen viajes y capacidad de catálogo, pero no inventario por hora ni eventos de desabasto/saturación. La etiqueta es un *proxy* de presión de flujo basada en la nomenclatura de septiembre de 2026.

---

## 2. Arquitectura y estructura del repositorio

El proyecto sigue una estructura **Cookiecutter Data Science** con entorno bloqueado mediante `uv` (**Python 3.13.2** y **MLflow 3.17.2**). Los datos crudos, archivos Parquet, bases SQLite de MLflow y modelos serializados permanecen locales y fuera de Git; solo se versionan código, configuración, pruebas, manifiesto de datos y reportes agregados.

```text
├── .github/workflows/ci.yml   # Pipeline CI/CD (calidad, seguridad, PyTest y humo en Compose)
├── data/
│   ├── raw/                   # CSV de origen y catálogo (locales, no versionados)
│   ├── interim/               # Parquet unificados (locales, no versionados)
│   ├── processed/             # Tablas de features (locales, no versionados)
│   └── data_manifest.json     # Manifiesto versionado con hashes SHA-256 y data_version
├── docs/docs/                 # Documentación técnica, evidencias y Model Card
├── models/                    # Artefactos exportados localmente (no versionados)
├── notebooks/                 # Exploración y auditoría reproducible
├── proyecto_mibici/
│   ├── config.py              # Configuración centralizada (rutas, fechas, espacios de búsqueda)
│   ├── dataset.py             # Ingesta reproducible, limpieza y DataManifest
│   ├── features.py            # RepositorioParquet y variables causales sin fuga temporal
│   ├── experiments.py         # Orquestación de experimentos y CLI de MLflow Projects
│   ├── tracking.py            # Registro de parámetros, métricas, etiquetas y código
│   ├── registry.py            # Selección, validación y promoción controlada con aliases
│   ├── data_quality.py        # Validación estadística y drift train/test con Deepchecks
│   ├── modeling/              # Contratos, baselines y modelos candidatos (P1, P2, P3)
│   ├── api/                   # FastAPI (/health, /model-info, /predict), cliente y exportación
│   └── monitoreo/             # Periodos, drift (Evidently), reentrenamiento y prueba A/B
├── pruebas_carga/             # Generación de payloads reales y ejecución escalonada con Locust
├── references/                # ML Canvas, diagrama de arquitectura y tarifas de nube
├── reports/                   # Gráficas versionadas, reportes de monitoreo, carga y FinOps
├── tests/                     # Suite de 195 pruebas automatizadas en PyTest
├── Containerfile              # Imagen única para API, servidor MLflow y registro del modelo
├── docker-compose.yml         # Orquestación completa: mlflow + registrar-modelo + api
├── locustfile.py              # Escenario de pruebas de carga contra /predict
├── MLproject                  # Entry points: ingest, features, train, search, main
├── python_env.yaml            # Entorno virtual declarado para MLflow Projects
├── pyproject.toml / uv.lock   # Dependencias directas y transitivas bloqueadas
└── requirements.txt           # Dependencias runtime exportadas desde uv.lock
```

---

## 3. Instalación e inicio rápido

Requisitos: **Git**, **Python 3.13**, **[uv](https://docs.astral.sh/uv/)** y **Docker** o **Podman** (para el despliegue contenerizado).

```bash
# 1. Clonar el repositorio y restaurar el entorno bloqueado (Python 3.13.2 + MLflow 3.17.2)
git clone [https://github.com/aramosc09/TC5061-Proyecto-Equipo25.git](https://github.com/aramosc09/TC5061-Proyecto-Equipo25.git)
cd TC5061-Proyecto-Equipo25
uv sync --frozen
uv run --frozen mlflow --version

# 2. En una primera terminal, iniciar el servidor local de MLflow (Tracking + Model Registry)
uv run --frozen mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination "file://$PWD/mlartifacts" \
  --host 127.0.0.1 --port 5000

# 3. En una segunda terminal, apuntar al servidor local
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)

# 4. Descargar los CSV de MiBici y construir la base limpia de 36 meses
uv run --frozen python -m proyecto_mibici.dataset procesar-datos

# 5. Ejecutar el proyecto MLflow (P1, P2 y P3) y verificar artefactos
uv run --frozen python -m proyecto_mibici.experiments ejecutar-proyecto
uv run --frozen python -m proyecto_mibici.experiments verificar-experimento
```

La interfaz de MLflow queda disponible en `http://127.0.0.1:5000` bajo el experimento **`mibici-equipo25`**.

> **Extras opcionales:** `uv sync --frozen --extra notebooks` (exploración), `--extra monitoreo` (Evidently) y `--extra carga` (Locust).

---

## 4. Datos, ingesta y validación

* **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/) (`datos_abiertos_AAAA_MM.csv` y `nomenclatura_2026_09.csv`).
* **Ventana y volumen:** octubre de 2023 a septiembre de 2026 (36 meses). De 13,621,885 registros crudos se obtienen **13,621,127 viajes limpios** y 473 estaciones observadas (471 presentes en el catálogo de 484 filas).
* **Trazabilidad:** `data/data_manifest.json` guarda hashes SHA-256, conteos de descartes, esquemas y la versión reproducible **`data_version=09a6a584dd5d`**.
* **Partición temporal estricta (sin fuga de datos):**
  * Entrenamiento: octubre de 2023 – junio de 2025
  * Validación: julio de 2025 – septiembre de 2025
  * Prueba: octubre de 2025 – septiembre de 2026
* **Privacidad:** La ingesta elimina identificadores de usuario y variables demográficas (sexo, edad). MLflow nunca recibe filas individuales.

Comandos de ingesta, features y verificación:

```bash
uv run --frozen python -m proyecto_mibici.dataset procesar-datos
uv run --frozen python -m proyecto_mibici.dataset verificar-manifiesto
uv run --frozen python -m proyecto_mibici.dataset verificar-parquet
uv run --frozen python -m proyecto_mibici.experiments features
```

**Validación de calidad con Deepchecks** (en entorno aislado de Python 3.12 por compatibilidad de Deepchecks 0.19.1):

```bash
for pregunta in p1 p2 p3; do
  uv run --no-project --python 3.12 \
    --with-requirements requirements-quality.txt \
    python -m proyecto_mibici.data_quality --question "$pregunta"
done
```

---

## 5. Experimentación con MLflow Projects

`MLproject` expone los *entry points* `ingest`, `features`, `train`, `search` y `main`, ejecutados con `--env-manager uv` y los parámetros centralizados en `proyecto_mibici/config.py`.

```bash
# Ingesta mediante MLflow Project
uv run --frozen mlflow run . -e ingest --env-manager uv --experiment-name mibici-equipo25

# Entrenar baseline o candidato con hiperparámetros individuales
uv run --frozen mlflow run . -e train --env-manager uv \
  --experiment-name mibici-equipo25 -P question=p1 -P model=baseline

uv run --frozen mlflow run . -e train --env-manager uv \
  --experiment-name mibici-equipo25 -P question=p1 -P model=rf -P n_estimators=300

# Búsqueda de hiperparámetros como nested runs (rejillas definidas en SEARCH_SPACES)
uv run --frozen mlflow run . -e search --env-manager uv \
  --experiment-name mibici-equipo25 -P question=p1 -P model=hgb
```

### Modelos disponibles por pregunta

| Pregunta | Modelos (`model`) | Hiperparámetros principales |
| :--- | :--- | :--- |
| **P1** | `baseline`, `poisson`, `hgb`, `rf` | `alpha`, `max_iter` · `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` · `n_estimators`, `max_depth`, `max_features`, `max_samples` |
| **P2** | `baseline`, `logistic`, `hgb`, `rf` | `c`, `class_weight`, `max_iter` · parámetros de `hgb` y `rf` más `class_weight` (`balanced` / `none`) |
| **P3** | `baseline`, `kmeans`, `agglomerative`, `gmm` | `n_clusters`, `n_init` · `linkage` · `covariance_type`, `pca_components` (`0` = sin PCA) |

### Resultados de experimentación

Todos los candidatos ganadores superan a su *baseline* en la misma partición temporal. La selección se realiza exclusivamente sobre el conjunto de validación; las corridas registran métricas prefijadas (`train_`, `validation_`, `test_`) y etiquetas de linaje (`team`, `question`, `git_commit`, `data_version`, `model_family`, `stage_of_work`).

| Pregunta | Métrica | Baseline | Modelo ganador | Validación | Mejora vs. baseline |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **P1** | RMSE *(menor es mejor)* | Persistencia semanal · `1.933` | HistGradientBoosting (pérdida Poisson) | `1.418` | **-26.6 %** *(prueba: `1.411`, -28.3 %)* |
| **P2** | F1 macro *(mayor es mejor)* | Clase mayoritaria · `0.097` | HistGradientBoosting (clases balanceadas) | `0.364` | **+0.267** |
| **P3** | Silhouette *(mayor es mejor)* | Mediana de fin de semana · `0.037` | GaussianMixture (2 grupos, PCA 5) | `0.189` | **+0.152** |

---

## 6. Model Registry y linaje

El registro de modelos está gobernado por `proyecto_mibici.registry`, que exige firma (*signature*), ejemplo de entrada, capacidad de carga, árbol de Git limpio, exclusión de *baselines* y justificación explícita. Sin la bandera `--approve`, el comando solo muestra una vista previa sin escribir en el Registry.

```bash
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)

# Inspeccionar candidatos ordenados por métrica de validación
uv run --frozen python -m proyecto_mibici.registry candidates --question p1

# Promover un run aprobado a @champion (o @challenger)
uv run --frozen python -m proyecto_mibici.registry promote \
  --question p1 --run-id <RUN_ID> --alias champion \
  --reason "Mejor RMSE en validación frente a baseline y candidatos" --approve

# Verificar estado, versiones y linaje
uv run --frozen python -m proyecto_mibici.registry status
```

Para el servicio de demanda P1 (`mibici-equipo25-p1-demanda`), el alias **`@champion`** apunta a la **v1** en producción y **`@challenger`** a la **v2** reentrenada en evaluación. El detalle de evidencias está en [registro_modelos_entrega2.md](docs/docs/registro_modelos_entrega2.md).

### Exportación del modelo para el contenedor

```bash
uv run --frozen python -m proyecto_mibici.api.modelo exportar \
  --model-uri models:/mibici-equipo25-p1-demanda@champion \
  --destino models/api/p1
uv run --frozen python -m proyecto_mibici.api.modelo verificar
```

La exportación crea `models/api/p1/metadata_registro.json` con la versión, alias, `run_id`, `git_commit`, `data_version` y métrica de selección.

---

## 7. Despliegue y operación

### Sistema completo con Docker Compose (recomendado)

`docker-compose.yml` construye una única imagen (`Containerfile`) y orquesta tres servicios en una red privada con puertos expuestos solo en `127.0.0.1` y límites de CPU/memoria:
1. **`mlflow`**: servidor de Tracking y Model Registry con volumen persistente.
2. **`registrar-modelo`**: registra el modelo `@champion` exportado en `models/api/p1/` con su linaje y finaliza con código `0`.
3. **`api`**: servicio FastAPI que carga `@champion` directamente desde el Registry (`"source": "registry"`).

```bash
# Levantar el sistema completo (usar docker compose o podman compose)
docker compose up -d --build
docker compose ps -a

# Verificar salud, metadatos de linaje y predicción real
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
curl [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H 'Content-Type: application/json' \
  -d @peticion.json
```

La documentación interactiva OpenAPI queda en `http://127.0.0.1:8000/docs` y la UI de MLflow en `http://127.0.0.1:5000`. Las variables de entorno (`MLFLOW_PORT`, `MODEL_SOURCE`, `API_WORKERS`, `API_CPUS`, `API_MEMORY`) se documentan en `.env.example`.

Para detener los servicios: `docker compose down` (conserva el Registry) o `docker compose down -v` (limpia volúmenes).

### Ejecución aislada del contenedor de la API

Si se requiere correr solo la API en modo `local` (sin servidor MLflow):

```bash
docker build -f Containerfile -t mibici-api:p1 .
docker run --rm --name mibici-api -p 8000:8000 mibici-api:p1
```

*(En Podman, usar `podman build --format docker -f Containerfile -t mibici-api:p1 .` para conservar el `HEALTHCHECK`.)*

### Cliente HTTP de prueba

`proyecto_mibici/api/cliente.py` prepara un registro real desde `data/processed/tabla_supervisada.parquet` (fecha_hora más las 24 variables históricas de rezagos, medias y balances, a partir de las cuales se completan las 31 variables de entrada del modelo), lo valida con Pydantic y consulta `/predict`:

```bash
# Validación local sin levantar la API
uv run --frozen python -m proyecto_mibici.api.cliente --solo-validar

# Predicción real contra la API activa (muestra predicción, valor observado y error absoluto)
uv run --frozen python -m proyecto_mibici.api.cliente
```

La justificación arquitectónica de inferencia en tiempo real frente a procesamiento por lotes está documentada en [despliegue_inferencia_entrega4.md](docs/docs/despliegue_inferencia_entrega4.md).

---

## 8. CI/CD, pruebas y seguridad

### Pipeline de GitHub Actions

`.github/workflows/ci.yml` se ejecuta en cada `push` y `pull_request` hacia `main` (y manualmente vía `workflow_dispatch`) con tres trabajos:
1. **`calidad`**: ejecuta `ruff check`, `bandit -r proyecto_mibici` y `pip-audit -r requirements.txt` sin hallazgos.
2. **`pruebas`**: ejecuta la suite completa de **195 pruebas en PyTest** sobre fixtures sintéticos con caché de `uv`.
3. **`imagen`**: genera un modelo sintético de humo (`proyecto_mibici.api.modelo_humo`), levanta el stack con `docker compose up -d --build`, verifica `/health`, `/model-info` (`"source": "registry"`) y `/predict`, y limpia los volúmenes al terminar.

### Ejecución local de pruebas y auditoría

```bash
# Suite completa de pruebas (no requiere CSV históricos ni servidor MLflow activo)
uv run --frozen python -m pytest tests -q

# Estilo y seguridad (o `make security` en macOS/Linux)
uv run --frozen ruff check proyecto_mibici tests
uv run --frozen bandit -r proyecto_mibici
uv run --frozen pip-audit -r requirements.txt
```

### Controles de gobernanza verificados por pruebas

| Riesgo | Control implementado | Prueba de respaldo (`pytest -k`) |
| :--- | :--- | :--- |
| Exposición de datos personales | Eliminación de IDs de usuario y demografía en ingesta | `test_cleaning_across_chunks_privacy_and_repeatability` |
| Descargas no confiables | Solo URLs HTTPS del dominio oficial de MiBici | `test_requests_outside_official_https_domain_are_rejected` |
| Archivos corruptos o alterados | Escritura atómica y verificación SHA-256 en manifiesto | `test_manifest_detects_modified_parquet` |
| Fuga de información del futuro | Features causales y umbrales congelados en `train` | `test_features_do_not_see_the_current_or_future_hour` |
| Selección sesgada por `test` | Búsqueda y listado bloquean métricas de prueba | `test_search_refuses_to_select_with_test_metrics` |
| Promoción indebida a `champion` | Firma obligatoria, no baseline, Git limpio y `--approve` | `test_champion_requires_loadable_model_with_signature` |
| No reproducibilidad | Semilla fija verificada en entrenamiento y muestreo | `test_random_forest_training_is_reproducible_with_same_seed` |

---

## 9. Pruebas de carga y análisis FinOps

### Pruebas de carga con Locust

`locustfile.py` y `pruebas_carga/` evalúan `/predict` con peticiones reales válidas, midiendo latencia, tasa de errores y consumo de CPU/memoria del contenedor (`make monitor-resources`):

```bash
uv sync --frozen --extra monitoreo --extra carga

# Prueba estándar (20 usuarios, 60 s, lotes de 10)
make load-test

# Escenario escalonado con modelo champion real (10, 25, 50 y 100 usuarios, 120 s por escalón)
uv run --frozen python -m pruebas_carga.generar_peticiones --fecha 2026-09-10 --estacion 2 --hora 22
MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=2 API_CPUS=2.0 API_MEMORY=1g \
  docker compose up -d --no-deps --force-recreate api
MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=2 API_CPUS=2.0 API_MEMORY=1g \
  uv run --frozen --extra monitoreo python -m pruebas_carga.ejecutar \
  --runtime docker --prefijo champion_2workers_2cpu --tipo mixto \
  --escalones 10:120,25:120,50:120,100:120
```

**Resultados principales (modelo `@champion` real P1 v1, 2 workers / 2 vCPU):**
* **0 errores HTTP** entre 10 y 100 usuarios concurrentes.
* **Individual (1 registro):** 18.01 pred/s (p95 = 610 ms).
* **Lote de sistema completo (466 estaciones):** 6,401.79 pred/s (p95 = 2,200 ms), **355× más rendimiento**.
* Ninguna modalidad mantiene p95 ≤ 500 ms bajo concurrencia alta; el uso recomendado es inferencia por lotes para pronóstico horario masivo y peticiones individuales para consultas puntuales.

### Análisis FinOps

* **Unidad de costo:** una predicción estación-hora.
* **Demanda base modelada:** 450 estaciones × 24 h × 30 días = **324,000 predicciones/mes**.
* **Fuentes:** `reports/finops/champion_una/`, `reports/finops/champion_batch466/` y `references/precios_nube_2vcpu.csv` (tarifas de referencia al 22-nov-2026, excluyendo red, discos e impuestos).

| Escenario de demanda | EC2 `t3.small` (siempre encendida) | Cloud Run (2 vCPU + 1 GiB, 1 registro) | Cloud Run (2 vCPU + 1 GiB, lote de 466) |
| :--- | :---: | :---: | :---: |
| **Base** (324 mil pred./mes) | $15.26 USD | $2.48 USD | **$0.02 USD** |
| **10×** (3.24 M pred./mes) | $15.26 USD | $24.79 USD | **$0.20 USD** |
| **100×** (32.4 M pred./mes) | $15.26 USD | $247.91 USD | **$2.00 USD** |

*(Son proyecciones de costo, no facturación real ni garantía de SLO con una sola instancia.)* Detalle completo en [operacion_entrega6.md](docs/docs/operacion_entrega6.md).

---

## 10. Monitoreo, A/B testing y gobernanza

### Monitoreo de drift con Evidently

```bash
uv sync --frozen --extra monitoreo
uv run --frozen --extra monitoreo python -m proyecto_mibici.monitoreo.drift --comparacion principal --sin-mlflow
uv run --frozen --extra monitoreo python -m proyecto_mibici.monitoreo.drift --comparacion estacional --sin-mlflow
```

* **Hallazgo:** Drift detectado en **3 de 31 columnas** (`mes`, `balance_acumulado_24`, `salidas_media_168`) en la comparación principal, pero **0 de 31** en el control estacional (mismos meses), confirmando un patrón estacional y no degradación estructural.
* **Disparadores definidos para Continuous Training:** ≥ 2 variables con drift durante dos meses consecutivos, > 10 % de filas provenientes de estaciones nuevas o incremento de RMSE > 5 % sobre la referencia.

### Reentrenamiento y evaluación A/B fuera de línea

El modelo P1 se reentrenó sobre la ventana ampliada (octubre de 2023 – junio de 2026) como **`@challenger` (v2)** sin desplazar a `@champion` (v1):

```bash
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)
uv run --frozen python -m proyecto_mibici.monitoreo.reentrenar
uv run --frozen python -m proyecto_mibici.monitoreo.ab
```

Sobre **955,951 filas** de evaluación fuera de línea (julio–septiembre de 2026):
* **Baseline semanal:** RMSE `1.9478` · MAE `0.9662` · R² `0.4139`
* **`@champion` (v1):** RMSE `1.4398` · MAE `0.7745` · R² `0.6797`
* **`@challenger` (v2):** RMSE `1.4078` (**-2.2 %**) · MAE `0.7715` · R² `0.6938`

La promoción a `@champion` exige intervalo de confianza del 95 % de $\Delta\text{RMSE}$ completamente negativo sin empeorar más de 1 % en estaciones conocidas; el *rollback* consiste en reasignar el alias en el Registry. Resultados en `reports/monitoreo/ab/` y [ab_metricas_modelos.png](reports/figures/entrega5/ab_metricas_modelos.png).

### Model Card

La documentación responsable del modelo de demanda P1 (uso previsto, datos sin información personal, métricas, limitaciones, seguridad, monitoreo y responsables por componente) se encuentra en **[docs/docs/model_card.md](docs/docs/model_card.md)** y es coherente con los metadatos devueltos por `/model-info`.

---

## 11. Equipo, flujo de trabajo y referencias

### Integrantes — Equipo 25

| Integrante | Matrícula | Contribución principal |
| :--- | :--- | :--- |
| Alejandro Ramos Contreras | A01747461 | P3; integración del experimento y seguimiento de MLflow. |
| Ángel Alexis Anaya Alamea | A01741351 | P2: clasificación de presión de flujo; CI/CD. |
| Erick Fernando Carrillo Pineda | A01840029 | P1: predicción de demanda de viajes. |
| Jose Rigoberto Cibrian Jacobo | A01740550 | Diagrama de arquitectura; módulo y documentación de Model Registry. |

### Flujo de trabajo con Git

`main` es la rama de integración protegida por Pull Requests y revisiones entre pares. Se utilizan ramas descriptivas (`feature/...`, `fix/...`, `docs/...`) y mensajes bajo [Conventional Commits](https://www.conventionalcommits.org/).

### Índice de documentación

* **Gobernanza y modelo:** [Model Card (P1)](docs/docs/model_card.md) · [Registro de modelos](docs/docs/registro_modelos_entrega2.md)
* **Diseño y datos:** [Contexto y requisitos](docs/docs/context.md) · [Propuesta de solución](docs/docs/propuestaSolucion.md) · [ML Canvas](references/ml_canvas_mibici.pdf) · [Arquitectura (Mermaid)](references/arquitectura_propuesta_mibici.mmd)
* **Calidad y seguridad:** [Evidencias de pruebas](docs/docs/evidencias_entrega3.md) · [Auditoría de seguridad](docs/docs/seguridad_entrega3.md)
* **Despliegue, carga y FinOps:** [Estrategia de despliegue e inferencia](docs/docs/despliegue_inferencia_entrega4.md) · [Operación, Compose, Locust y FinOps](docs/docs/operacion_entrega6.md)
* **Monitoreo:** [Análisis de drift y Continuous Training](docs/docs/drift_entrega5.md) · [Reentrenamiento challenger](docs/docs/entrenamiento_challenger.md)
* **Notebooks de soporte:** [Auditoría de datos](notebooks/1.3-ARC-verificacion-datos-punto2.ipynb) · [Validación de ingesta](notebooks/1.4-ARC-validacion-ingesta-reproducible.ipynb) · [Exploración y feature engineering](notebooks/1.5-eq25-exploracion-feature-engineering.ipynb)