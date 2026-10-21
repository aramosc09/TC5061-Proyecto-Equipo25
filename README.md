# MiBici ML · Equipo 25

Proyecto académico de Machine Learning y MLOps para analizar datos abiertos del sistema de bicicleta pública MiBici del Área Metropolitana de Guadalajara. El repositorio contiene la ingesta y validación de datos, la construcción de variables, experimentos reproducibles para tres preguntas operativas y el registro de resultados con MLflow.

## Propósito y preguntas

| Pregunta | Tarea y unidad | Objetivo implementado | Horizonte / uso esperado |
|---|---|---|---|
| **P1 — Demanda** | Regresión por estación y hora | Predecir el conteo de viajes que inician en una estación. | Conteo de la hora que comienza 24 horas después del origen de predicción. Puede apoyar la planeación de flota y rebalanceo. |
| **P2 — Presión de flujo** | Clasificación por estación y hora | Clasificar la presión futura como `balanced`, `bike_pressure` o `dock_pressure`, usando el balance de viajes normalizado por capacidad como proxy. | Ventana de dos horas. Puede orientar la priorización de estaciones para revisión o rebalanceo. |
| **P3 — Perfiles de estaciones** | Agrupamiento por estación | Agrupar estaciones por perfiles semanales normalizados de salidas y llegadas. Las coordenadas se reservan para visualización, no son variables del agrupamiento. | Descripción histórica de patrones de uso que puede apoyar análisis y planeación de cobertura. |

**Limitación importante de P2:** los datos abiertos contienen viajes y capacidad de catálogo, pero no inventario por hora ni eventos de desabasto o saturación. Por tanto, la etiqueta es un proxy de presión de flujo y no una observación del riesgo real de quedarse sin bicicletas o anclajes. La capacidad usada proviene de la nomenclatura de septiembre de 2026.

## Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Se usan archivos mensuales `datos_abiertos_AAAA_MM.csv` y el catálogo de estaciones `nomenclatura_2026_09.csv`.
- **Ventana documentada:** octubre de 2023 a septiembre de 2026 (36 meses).
- **Volumen:** 13,621,885 viajes; el manifiesto registra 484 filas de nomenclatura.
- **Estaciones:** 473 identificadores aparecen en los viajes; 471 tienen correspondencia con capacidad y coordenadas válidas en la nomenclatura. La validación de datos documenta 3,954 viajes vinculados con los IDs 405 y 407, que no tienen correspondencia en ese catálogo.
- **Trazabilidad:** `data/data_manifest.json` identifica la versión de los datos (`5e95734e844b` en la versión documentada), cuenta los registros y guarda hashes de los archivos de origen. Más detalles en [validacion_datos_mibici.md](references/validacion_datos_mibici.md).
- **Particiones supervisadas P1/P2:** entrenamiento hasta el 31 de marzo de 2026; validación abril–junio de 2026; prueba julio–septiembre de 2026. Se usa partición temporal, no aleatoria.
- **Partición P3:** entrenamiento octubre de 2023–junio de 2025; validación julio–septiembre de 2025; prueba octubre de 2025–septiembre de 2026.
- **Datos personales:** los CSV originales pueden contener identificadores y características personales. No se utilizan como predictores ni se registran en MLflow; los datos y resultados se mantienen localmente y no se suben a Git. El ejecutor de MLflow proyecta únicamente las columnas necesarias para modelar P1/P2.
- **Uso:** consultar las condiciones vigentes en el [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/) y citar la fuente. Este repositorio académico no redistribuye los archivos de datos.

## Estructura del repositorio

```text
data/
├── raw/                   # CSV de origen y catálogo (locales, no versionados)
├── interim/               # Parquet de trabajo (local, no versionado)
├── processed/             # Datos derivados (locales, no versionados)
└── data_manifest.json     # Manifiesto de versión y hashes, versionado
docs/docs/                 # Contexto, propuesta y documentación del proyecto
environments/p1/           # Entorno reproducible y lockfile de P1/MLflow
notebooks/                 # Exploración y auditoría; no es la fuente del pipeline
references/                # ML Canvas, diagramas y validación de datos
reports/figures/           # Figuras producidas localmente
tests/                     # Pruebas de datos, features y flujos
proyecto_mibici/
├── config.py              # Configuración general y rutas
├── config_p1.py           # Fechas, particiones y parámetros P1/P2
├── dataset.py             # Ingesta, Parquet y manifiesto
├── features_p1.py         # Features y objetivos horarios P1/P2; perfiles semanales
├── features_p2.py         # Features de la implementación P2 existente
├── features_p3.py         # Agregaciones y perfiles P3
└── modeling/
    ├── experiment_baselines.py # Ejecutor integrado de P1/P2/P3 y candidatos P1/P2
    ├── experiment_p3.py        # Ejecutor del baseline P3
    ├── train_p1.py             # Entrenamiento y tracking integrado P1/P2/P3
    ├── p1/, p2/, p3/           # Entrenadores y módulos de predicción
    └── ...
```

Los CSV, Parquet, bases de tracking, artefactos de MLflow y modelos serializados son locales y están excluidos de Git. Solo se versiona el manifiesto, que no incluye las filas de viajes.

## Instalación

Se necesita Git, Python 3.13 y [`uv`](https://docs.astral.sh/uv/). Desde la raíz del repositorio:

```bash
git clone https://github.com/aramosc09/TC5061-Proyecto-Equipo25.git
cd TC5061-Proyecto-Equipo25
uv sync --project environments/p1 --frozen
```

El entorno bloqueado de `environments/p1/` fija MLflow 2.21.3, compatible con la base SQLite de tracking descrita abajo. Para comprobar la instalación:

```bash
PYTHONPATH=. uv run --project environments/p1 python --version
PYTHONPATH=. uv run --project environments/p1 python -m proyecto_mibici.dataset --help
```

## Ingesta y validación

La ingesta del punto 10 descarga automáticamente los CSV faltantes de octubre de 2023 a septiembre de 2026 y la nomenclatura de septiembre de 2026, resolviendo los enlaces publicados por MiBici. No requiere descargar ni consolidar archivos manualmente. Las rutas, fechas y tamaño de lote se centralizan en [config.py](proyecto_mibici/config.py).

```bash
PYTHONPATH=. uv run --project environments/p1 --frozen python -m proyecto_mibici.dataset procesar-datos
PYTHONPATH=. uv run --project environments/p1 --frozen python -m proyecto_mibici.dataset verificar-manifiesto
PYTHONPATH=. uv run --project environments/p1 --frozen python -m proyecto_mibici.dataset verificar-parquet
```

La ejecución normaliza codificación, nombres, IDs y fechas explícitas; descarta registros inválidos e inicios fuera del mes fuente, deduplica viajes por ID y procesa por lotes sin cargar los 36 meses completos en memoria. No recorta llegadas posteriores al fin del rango ni infiere inventario o actividad histórica desde el catálogo. Produce:

- `data/interim/viajes_unificados.parquet`: `viaje_id`, `inicio_viaje`, `fin_viaje`, `origen_id`, `destino_id`, `fecha_archivo`.
- `data/interim/estaciones.parquet`: catálogo normalizado; capacidad no utilizable y coordenadas inválidas quedan nulas y se contabilizan.
- `data/data_manifest.json`: hashes de fuentes/salidas, rango, filas crudas/limpias, descartes, esquemas y `data_version`, recuperable con `obtener_version_datos()` para MLflow.

Los Parquet comunes **siempre excluyen identificadores de usuario y demografía**; aun así no se versionan ni se publican viajes individuales. Los CSV originales permanecen locales y excluidos de Git. El manifiesto, sin filas individuales, debe revisarse y versionarse mediante PR al cambiar la base; los runs anteriores conservan su versión histórica. La versión incluye las reglas de normalización además del contenido de las fuentes.

Por defecto se reutilizan los CSV existentes; `--actualizar` solicita descargarlos de nuevo. Para procesar exclusivamente copias locales completas se admite `--sin-descarga`; un mes faltante provoca error. `--input-path`, `--output-path` y `--manifiesto-path` permiten validar en rutas separadas sin reemplazar la base compartida. La escritura se prepara en temporales y el manifiesto se publica al final.

La [propuesta de pipelines](docs/docs/propuestaSolucion.md) parte de estas salidas comunes. En esta etapa no se han adaptado los ejecutores históricos de modelos a ese diseño. La [validación de ingesta](notebooks/1.4-ARC-validacion-ingesta-reproducible.ipynb) conserva una ejecución real de un mes en directorios temporales, y las pruebas de [ingesta](tests/test_dataset_ingestion.py) cubren descarga, limpieza, privacidad y versionado.

## Ejecutar los experimentos MLflow

El ejecutor integrado usa los baselines de P1/P2 y el baseline de P3, y registra parámetros, métricas, tags y artefactos en un único experimento `mibici-equipo25`.

**Pre-requisitos locales:** además de los CSV, el runner requiere `data/processed/raw_data.parquet` con las columnas originales de MiBici (`Viaje_Id`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`, `mes_archivo`) y `data/raw/nomenclatura_2026_09.csv`. `dataset.py procesar-datos` genera el Parquet normalizado `data/interim/viajes_unificados.parquet`, no ese Parquet fuente. El Parquet fuente no se versiona; el runner selecciona las columnas estrictamente necesarias y excluye los campos personales antes de producir la tabla P1/P2.

Desde la raíz, ejecuta una vez los tres baselines:

```bash
PYTHONPATH=. uv run --project environments/p1 python -m proyecto_mibici.modeling.experiment_baselines
```

Para registrar solo los modelos candidatos de P1 y P2 y compararlos con sus baselines existentes, ejecuta:

```bash
PYTHONPATH=. uv run --project environments/p1 python -m proyecto_mibici.modeling.experiment_baselines --candidates-only
```

El segundo comando no vuelve a ejecutar P3. MLflow conserva los runs previos y añade los nuevos, por lo que al repetirlo se acumulan runs en el experimento. La base y artefactos locales se guardan en `mlruns/mibici-equipo25/tracking.db` y `mlartifacts/mibici-equipo25/`.

Abre la interfaz con:

```bash
PYTHONPATH=. uv run --project environments/p1 mlflow ui \
  --backend-store-uri sqlite:///mlruns/mibici-equipo25/tracking.db
```

Visita <http://127.0.0.1:5000>, abre el experimento `mibici-equipo25` y selecciona dos o más runs para usar **Compare**. Para una comparación justa, compara runs de la misma pregunta y revisa métricas del mismo conjunto (validación o prueba). En cada run verifica `data_version`, parámetros, métricas, tags (`team`, `question`, `git_commit`) y artefactos. P3 tiene particiones distintas a P1/P2, por lo que sus métricas no deben compararse directamente con las de las preguntas supervisadas.

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
- [Arquitectura](references/arquitectura_mibici.md) y [diagrama PDF](references/arquitectura_mibici.pdf).
- [Validación del conjunto de datos](references/validacion_datos_mibici.md).
