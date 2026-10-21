# MiBici ML · Equipo 25

Proyecto académico de Machine Learning y MLOps para analizar datos abiertos del sistema de bicicleta pública MiBici del Área Metropolitana de Guadalajara. El repositorio contiene la ingesta y validación de datos, la construcción de variables, experimentos reproducibles para tres preguntas operativas y el registro de resultados con MLflow.

## Inicio rápido: cómo ejecutar el proyecto

Desde la raíz del repositorio, con `uv` instalado. El entorno actual usa **MLflow 3.17.2**, la versión estable más reciente publicada en PyPI al **21 de octubre de 2026**.

```bash
# 1. Crear/restaurar el entorno bloqueado y comprobar MLflow 3.17.2.
uv sync --frozen
uv run --frozen mlflow --version

# 2. Descargar los CSV faltantes y construir la base limpia de 36 meses.
uv run --frozen python -m proyecto_mibici.dataset procesar-datos

# 3. Ejecutar P1, P2 y P3 mediante MLflow Projects y registrar los resultados.
uv run --frozen python -m proyecto_mibici.experiments ejecutar-proyecto

# 4. Comprobar que los tres runs y sus artefactos estén completos.
uv run --frozen python -m proyecto_mibici.experiments verificar-experimento

# 5. Abrir la interfaz; mantener esta terminal abierta.
uv run --frozen mlflow ui --backend-store-uri sqlite:///mlruns/mibici-equipo25/tracking.db --host 127.0.0.1 --port 5000
```

La interfaz estará en <http://127.0.0.1:5000>, experimento **mibici-equipo25**. No hay que ejecutar notebooks, activar entornos manualmente ni lanzar un servidor antes del experimento. El comando de experimento exige la base de ingesta; no descarga datos por sí mismo. Repetirlo añade runs, no borra los existentes.

**Qué ejecuta cada pieza:** `uv` restaura y usa las dependencias; [dataset.py](proyecto_mibici/dataset.py) obtiene y limpia datos; [experiments.py](proyecto_mibici/experiments.py) inicia el proyecto declarado en [MLproject](MLproject), prepara las horas y coordina los baselines; [tracking.py](proyecto_mibici/tracking.py) registra parámetros, métricas y artefactos. La UI solo permite consultar/comparar lo ya registrado: no ejecuta el pipeline. Los resultados agregados quedan en `reports/figures/baselines/` y los artefactos de MLflow en `mlartifacts/mibici-equipo25/`.

## Propósito y preguntas

| Pregunta | Tarea y unidad | Objetivo implementado | Horizonte / uso esperado |
|---|---|---|---|
| **P1 — Demanda** | Regresión por estación y hora | Predecir viajes iniciados; baseline de la misma hora de la semana anterior. | Siguiente hora. Apoya la planificación de flota. |
| **P2 — Presión de flujo** | Clasificación por estación y hora | Clasificar dirección e intensidad del flujo: sin presión neta o extracción/acumulación baja, moderada o alta. Baseline de clase mayoritaria de entrenamiento. | Siguiente hora. Orienta verificaciones, no órdenes automáticas de rebalanceo. |
| **P3 — Perfiles de estaciones** | Agrupamiento por estación | Agrupar estaciones por perfiles semanales normalizados de salidas y llegadas. Las coordenadas se reservan para visualización, no son variables del agrupamiento. | Descripción histórica de patrones de uso que puede apoyar análisis y planeación de cobertura. |

**Limitación importante de P2:** los datos abiertos contienen viajes y capacidad de catálogo, pero no inventario por hora ni eventos de desabasto o saturación. Por tanto, la etiqueta es un proxy de presión de flujo y no una observación del riesgo real de quedarse sin bicicletas o anclajes. La capacidad usada proviene de la nomenclatura de septiembre de 2026.

## Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Se usan archivos mensuales `datos_abiertos_AAAA_MM.csv` y el catálogo de estaciones `nomenclatura_2026_09.csv`.
- **Ventana documentada:** octubre de 2023 a septiembre de 2026 (36 meses).
- **Volumen obtenido:** 13,621,885 registros de viajes y 484 filas de nomenclatura. La ingesta completa produce **13,621,127 viajes limpios**; los descartes se detallan en el manifiesto.
- **Estaciones:** 473 IDs observados en las fuentes; 471 coinciden con el catálogo. La cobertura efectiva se vuelve a contabilizar por pregunta y periodo después de limpiar.
- **Trazabilidad:** [data_manifest.json](data/data_manifest.json) contiene hashes, conteos crudos/limpios y `data_version=09a6a584dd5d` para la base ejecutada. La versión incluye las reglas de normalización.
- **Partición común P1/P2/P3:** entrenamiento octubre de 2023–junio de 2025; validación julio–septiembre de 2025; prueba octubre de 2025–septiembre de 2026. No hay división aleatoria.
- **Cobertura analítica:** se requieren archivos mensuales completos y actividad observada en las 168 horas anteriores al corte. El arranque sin ventana completa y las ausencias prolongadas quedan fuera. Un cero es una hipótesis analítica tras actividad reciente, no prueba de disponibilidad o actividad operacional continua. P1 exige también cobertura histórica de la hora de referencia semanal.
- **Datos personales:** la base común excluye usuario y demografía. Los CSV/Parquet permanecen locales; MLflow recibe metadatos y artefactos agregados, nunca filas individuales.
- **Uso:** consultar las condiciones vigentes en el [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/) y citar la fuente. Este repositorio académico no redistribuye los archivos de datos.

## Estructura del repositorio

```text
data/
├── raw/                   # CSV de origen y catálogo (locales, no versionados)
├── interim/               # Parquet de trabajo (local, no versionado)
├── processed/             # Datos derivados (locales, no versionados)
└── data_manifest.json     # Manifiesto de versión y hashes, versionado
docs/docs/                 # Contexto, propuesta y documentación del proyecto
notebooks/                 # Exploración y auditoría; no ejecutan el experimento base
references/                # ML Canvas y diagrama de arquitectura
reports/figures/baselines/  # Resultados agregados por run padre (locales)
tests/                     # Pruebas de ingesta, causalidad, baselines y tracking
MLproject                  # Entry points main, ingesta y features
pyproject.toml             # Dependencias directas fijadas
uv.lock                    # Entorno común y dependencias transitivas bloqueadas
proyecto_mibici/
├── config.py              # ProjectConfig: rutas, fechas, ventanas y semilla
├── dataset.py             # Ingesta reproducible y DataManifest
├── features.py            # RepositorioParquet y features causales
├── experiments.py         # IngestorMiBici y ProyectoMiBici; CLI/MLflow Projects
├── tracking.py            # RegistroExperimentos y copia exacta de código
└── modeling/
    ├── base.py             # Contratos PipelinePregunta y BaselineResult
    ├── supervised_baselines.py # PipelineP1 y PipelineP2
    └── p3_baseline.py      # PipelineP3
```

Los CSV, Parquet, bases de tracking, artefactos de MLflow y modelos serializados son locales y están excluidos de Git. Solo se versiona el manifiesto, que no incluye las filas de viajes.

## Instalación

Se necesita Git, Python 3.13 y [`uv`](https://docs.astral.sh/uv/). Desde la raíz del repositorio:

```bash
git clone https://github.com/aramosc09/TC5061-Proyecto-Equipo25.git
cd TC5061-Proyecto-Equipo25
uv sync --frozen
```

El entorno único usa Python 3.13.2 y **MLflow 3.17.2**, con versiones directas fijadas y dependencias transitivas en el lockfile. No se necesita activar un entorno ni instalar paquetes globalmente.

**Decisión de versión:** se eligió [MLflow 3.17.2](https://pypi.org/project/mlflow/3.17.2/), la última versión estable publicada al 21 de octubre de 2026. La actualización se validó con **35 pruebas aprobadas**, ejecución de MLflow Projects y los tres baselines sobre la base completa. Sus métricas coinciden con las de la ejecución anterior: cambió el entorno de tracking, no los modelos ni los datos.

**Migración de una instalación anterior:** en esta copia local ya se respaldó la base SQLite y se ejecutó la migración oficial con `mlflow db upgrade`. Se conservaron los runs existentes y se añadió una nueva ejecución con 3.17.2. No es necesario repetir la migración para usar esta copia. En otras instalaciones con una base antigua, detener primero la UI y cualquier escritor, realizar un respaldo mediante la API de backup de SQLite y ejecutar `uv run --frozen mlflow db upgrade sqlite:///mlruns/mibici-equipo25/tracking.db` antes de usarla. No abrir la base migrada con MLflow 2.x; una reversión requiere restaurar el respaldo. Desde un clon nuevo, los comandos de inicio rápido crean el almacenamiento sin necesitar una base previa.

```bash
uv run --frozen python --version
uv run --frozen python -m proyecto_mibici.experiments --help
```

Para exploración interactiva se puede instalar el extra opcional con `uv sync --frozen --extra notebooks`.

## Ingesta y validación

La ingesta del punto 10 descarga automáticamente los CSV faltantes de octubre de 2023 a septiembre de 2026 y la nomenclatura de septiembre de 2026, resolviendo los enlaces publicados por MiBici. No requiere descargar ni consolidar archivos manualmente. Las rutas, fechas y tamaño de lote se centralizan en [config.py](proyecto_mibici/config.py).

```bash
uv run --frozen python -m proyecto_mibici.dataset procesar-datos
uv run --frozen python -m proyecto_mibici.dataset verificar-manifiesto
uv run --frozen python -m proyecto_mibici.dataset verificar-parquet
```

La ejecución normaliza codificación, nombres, IDs y fechas explícitas; descarta registros inválidos e inicios fuera del mes fuente, deduplica viajes por ID y procesa por lotes sin cargar los 36 meses completos en memoria. No recorta llegadas posteriores al fin del rango ni infiere inventario o actividad histórica desde el catálogo. Produce:

- `data/interim/viajes_unificados.parquet`: `viaje_id`, `inicio_viaje`, `fin_viaje`, `origen_id`, `destino_id`, `fecha_archivo`.
- `data/interim/estaciones.parquet`: catálogo normalizado; capacidad no utilizable y coordenadas inválidas quedan nulas y se contabilizan.
- `data/data_manifest.json`: hashes de fuentes/salidas, rango, filas crudas/limpias, descartes, esquemas y `data_version`, recuperable con `obtener_version_datos()` para MLflow.

Los Parquet comunes **siempre excluyen identificadores de usuario y demografía**; aun así no se versionan ni se publican viajes individuales. Los CSV originales permanecen locales y excluidos de Git. El manifiesto, sin filas individuales, debe revisarse y versionarse mediante PR al cambiar la base; los runs anteriores conservan su versión histórica. La versión incluye las reglas de normalización además del contenido de las fuentes.

Por defecto se reutilizan los CSV existentes; `--actualizar` solicita descargarlos de nuevo. Para procesar exclusivamente copias locales completas se admite `--sin-descarga`; un mes faltante provoca error. `--input-path`, `--output-path` y `--manifiesto-path` permiten validar en rutas separadas sin reemplazar la base compartida. La escritura se prepara en temporales y el manifiesto se publica al final.

La [propuesta de pipelines](docs/docs/propuestaSolucion.md) y los tres baselines parten de estas mismas salidas. Las pruebas de [ingesta](tests/test_dataset_ingestion.py) cubren descarga, limpieza, privacidad y versionado. El [notebook de validación](notebooks/1.4-ARC-validacion-ingesta-reproducible.ipynb) conserva una prueba aislada de un mes; el experimento base utiliza el rango completo.

## Ejecutar los experimentos MLflow

Después de la ingesta, este comando valida la base, genera/reutiliza la tabla horaria y ejecuta los tres baselines mediante **MLflow Projects**:

```bash
uv run --frozen python -m proyecto_mibici.experiments ejecutar-proyecto
uv run --frozen python -m proyecto_mibici.experiments verificar-experimento
```

[MLproject](MLproject) declara los entry points `main`, `features` e `ingesta`. Projects usa `env_manager=local` y cada entry point ejecuta `uv run --frozen`: no depende de paquetes del Python global ni requiere Conda/Docker. El comando prepara primero el experimento `mibici-equipo25` y su almacén local, luego crea un run padre y tres hijos P1/P2/P3. Repetirlo añade otra ejecución; nunca borra runs automáticamente.

| Pregunta | Baseline implementado | Métricas |
|---|---|---|
| P1 | Persistencia semanal, rezago de 168 horas | MAE, RMSE y error medio firmado. |
| P2 | `DummyClassifier(strategy="most_frequent")`, ajustado solo en entrenamiento | F1 macro sobre siete clases fijas; precisión/recall cuando su denominador está definido, soporte, matriz de confusión y frecuencia de alertas altas. |
| P3 | Mediana de proporción de uso en fin de semana, aprendida solo en entrenamiento | Silhouette cuando es definido, ARI entre periodos sobre estaciones comunes, tamaños y cobertura. |

Los candidatos del punto 4 siguen siendo propuestas: este comando ejecuta únicamente los baselines. Los percentiles 75/90 de P2 y la mediana de P3 quedan congelados antes de validación/prueba.

Cada hijo registra configuración, rango, partición, versión/hash de datos, semilla, modelo e hiperparámetros; métricas prefijadas por conjunto; tags `team`, `question`, `git_commit`; y tablas/PNG agregados. También conserva [el manifiesto](data/data_manifest.json), un resumen JSON y una copia exacta del código y del entorno bloqueado. Si hay cambios sin commit se identifica `git_worktree_dirty=true` y `source_tree_sha256`: HEAD por sí solo no representa esos cambios y la copia adjunta permite recuperarlos. Los resultados locales se guardan por run padre en `reports/figures/baselines/`.

No se necesita un Parquet creado manualmente en notebooks. La entrada de los tres modelos es exclusivamente la base común de ingesta. Las métricas de distintas preguntas no son comparables entre sí aunque compartan fechas.

Para generar solo las horas/features, sin registrar runs:

```bash
uv run --frozen python -m proyecto_mibici.experiments features
```

Para abrir la interfaz local:

```bash
uv run --frozen mlflow ui \
  --backend-store-uri sqlite:///mlruns/mibici-equipo25/tracking.db \
  --host 127.0.0.1 --port 5000
```

Visita <http://127.0.0.1:5000>, selecciona `mibici-equipo25` y expande los hijos del run padre. Los artefactos están en `resultados/` y `reproducibilidad/`. Para comparar repeticiones, selecciona runs de la misma pregunta y usa **Compare**; no interpretes los tres baselines de tareas distintas como modelos rivales.

## Pruebas

```bash
uv run --frozen python -m pytest tests -q
uv run --frozen ruff check proyecto_mibici tests
```

Las pruebas usan fuentes sintéticas y tracking temporal; no alteran los runs reales ni requieren viajes individuales publicados.

## Equipo y roles

Equipo 25, TC5061 — Operaciones de Aprendizaje Automático (MLOps).

| Integrante | Matrícula | Rol / contribución principal |
|---|---|---|
| Alejandro Ramos Contreras | A01747461 | P3; integración del experimento y seguimiento de MLflow. |
| Ángel Alexis Anaya Alamea | A01741351 | P2: clasificación de la etiqueta proxy de presión de flujo. |
| Erick Fernando Carrillo Pineda | A01840029 | P1: predicción de demanda de viajes. |
| Jose Rigoberto Cibrian Jacobo | A01740550 | Diagrama y documentación de arquitectura. |

## Flujo de trabajo con Git y GitHub

`main` es la rama de integración. El equipo trabaja con ramas cortas, Pull Requests y revisión por otro integrante; no se deben integrar cambios directamente a `main`.

1. Actualiza la rama principal y crea una rama con nombre descriptivo:

   ```bash
   git switch main
   git pull --ff-only
   git switch -c feature/p1-descripcion
   ```

   Convenciones recomendadas: `feature/p1-...`, `feature/p2-...`, `feature/p3-...`, `docs/...` y `fix/...`.
2. Haz cambios pequeños, ejecuta las pruebas pertinentes y crea commits con [Conventional Commits](https://www.conventionalcommits.org/), por ejemplo `feat(p1): agregar baseline de demanda` o `docs(readme): actualizar instrucciones`.
3. Publica la rama y abre un Pull Request hacia `main`:

   ```bash
   git push -u origin feature/p1-descripcion
   gh pr create --base main
   ```

4. Solicita revisión a otro integrante, atiende comentarios y verifica las comprobaciones antes de integrar el Pull Request.

La configuración de protección de `main`, las revisiones requeridas y la evidencia de contribuciones individuales deben confirmarse en GitHub; este README no presupone que las reglas estén activadas.

## Referencias

- [Contexto y requisitos del proyecto](docs/docs/context.md).
- [Propuesta de solución y resultados](docs/docs/propuestaSolucion.md).
- [ML Canvas](references/ml_canvas_mibici.pdf).
- [Arquitectura propuesta — fuente Mermaid](references/arquitectura_propuesta_mibici.mmd).
- [Auditoría reproducible de tamaño de datos](notebooks/1.3-ARC-verificacion-datos-punto2.ipynb).
