# MiBici ML · Equipo 25

Propuesta de solución de Machine Learning para el sistema de bicicleta pública MiBici (Área Metropolitana de Guadalajara): demanda de viajes, riesgo de desabasto o saturación de estaciones y perfiles de uso. Primera entrega del proyecto integrador.

## 1. Propósito

Resolver con ML tres preguntas sobre la operación de MiBici y dejar montada la base técnica del proyecto: ingesta reproducible y versionada, módulo de feature engineering reutilizable y una línea base registrada en MLflow Tracking.

## 2. Preguntas del proyecto

Unidad de análisis común: **estación × hora**, zona horaria `America/Mexico_City`. Horizontes y umbrales se parametrizan en `config.py`.

| # | Pregunta | Tarea | Qué se predice o agrupa | Horizonte | Usuario y decisión |
|---|---|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación? | Regresión | Viajes iniciados por estación y hora | 24 h | Operador: dimensionar flota y rebalanceo del día siguiente |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación (normal / sin bicicletas / sin anclajes) | Clase de riesgo en la ventana [t, t+2 h) | 2 h | Rebalanceo: a qué estaciones enviar camión |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso? | Clustering | Estaciones según su perfil de uso (una fila por estación) | Histórico (train) | Planeación urbana: dónde ampliar o ubicar estaciones |

**Aproximación del riesgo en P2** (los datos no incluyen inventario): balance normalizado `b = (llegadas − salidas) / capacidad` en la ventana; riesgo de vacío si `b ≤ −τ`, de saturación si `b ≥ +τ`. El umbral τ se calcula **solo con entrenamiento** y se congela. Limitaciones: el rebalanceo real no aparece en los datos, la demanda se censura cuando la estación está vacía y la capacidad proviene de una sola nomenclatura (sep 2026).

En P3 las coordenadas **no** entran al clustering; solo se usan para mapear los grupos.

El ML Canvas y el diagrama de arquitectura están en `references/`.

## 3. Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/): `datos_abiertos_AAAA_MM.csv` (viajes) y `nomenclatura_2026_09.csv` (estaciones).
- **Rango:** octubre 2023 a septiembre 2026 (36 meses). Excluye la pandemia y refleja la red actual.
- **Tamaño:** 13,621,885 viajes, 484 estaciones (salida real de la ingesta; ver `references/data_versions/`).
- **Partición temporal (sin mezcla aleatoria):** entrenamiento oct 2023 – mar 2026, validación abr – jun 2026, prueba jul – sep 2026. Las variables de una predicción en *t* solo usan información hasta *t*; τ, escaladores y clustering se ajustan solo con entrenamiento.
- **Condiciones de uso:** pendiente confirmar la licencia exacta del portal (no se abrieron las URLs de origen por decisión del equipo); uso académico, con cita de la fuente.
- **Datos personales:** el Parquet intermedio conserva `usuario_id`, `genero` y `anio_nacimiento` **solo en local**. No se usan como variables de ningún modelo, no se suben a Git ni a MLflow y los resultados se reportan agregados. Con `--excluir-datos-personales` se genera una versión sin ellas, apta para compartir.

## 4. Estructura del repositorio

Generado con Cookiecutter Data Science (`ccds`, con `include_code_scaffold`).

```
├── data/
│   ├── external/
│   ├── raw/                 # CSV de MiBici (no versionados)
│   ├── interim/             # Parquet intermedio (no versionado)
│   ├── processed/           # Tablas finales (no versionadas)
│   └── data_manifest.json   # Manifiesto de datos (sí versionado)
├── docs/
├── models/                  # Modelos serializados (no versionados)
├── notebooks/               # Solo exploración: 1.0-iniciales-descripcion.ipynb
├── references/              # ML Canvas y diagrama de arquitectura
├── reports/figures/         # Gráficas y mapas generados por código
├── tests/
├── proyecto_mibici/
│   ├── config.py            # Rutas, fechas, semillas y parámetros (única fuente)
│   ├── dataset.py           # Ingesta, Parquet y manifiesto de datos
│   ├── features.py          # Feature engineering reutilizable (P1/P2) y perfiles (P3)
│   ├── plots.py
│   └── modeling/
│       ├── train.py         # Baselines y candidatos P1/P2/P3 con registro en MLflow
│       └── predict.py
├── Makefile                 # requirements, data, features, train, ui, test, lint
├── pyproject.toml           # Dependencias con versión fija
└── README.md
```

## 5. Instalación

El proyecto usa la plantilla Cookiecutter Data Science con **virtualenv** y las dependencias se declaran en **pyproject.toml**, con versiones fijas.

```bash
git clone [URL_DEL_REPOSITORIO]
cd [CARPETA]

make create_environment

make requirements
```

Verificar la instalación con `make help` (lista los comandos disponibles) y con `python -c "import proyecto_mibici"`.

Al agregar una dependencia, se instala **con versión exacta** y se actualiza el archivo de dependencias en el mismo Pull Request.

## 6. Ingesta y versionado de datos

La ingesta normaliza codificación (UTF-8 / cp1252), nombres de columnas, tipos y fechas, y guarda los datos en Parquet. Genera `data/data_manifest.json` con el conteo de registros y el hash SHA-256 de cada archivo, más una `data_version` global que se registra en MLflow.

**Descarga:** Colocar los CSV y la nomenclatura en `data/raw/`.

```bash
# Procesa los CSV, guarda el Parquet y genera el manifiesto
python -m proyecto_mibici.dataset procesar-datos

# Comprueba que los CSV y el Parquet locales coinciden con el manifiesto
python -m proyecto_mibici.dataset verificar-manifiesto

# Valida solo el Parquet (sin necesitar los CSV)
python -m proyecto_mibici.dataset verificar-parquet
```

### Cómo validar que todos usamos los mismos datos

1. `git pull` para obtener el manifiesto oficial.
2. Colocar los datos en `data/raw/` (o el Parquet compartido en `data/interim/`).
3. Si se procesa localmente, usar un manifiesto propio para no sobrescribir el oficial:
   `procesar-datos --manifiesto-path data/interim/manifest_local.json`
4. Ejecutar `verificar-manifiesto` (con CSV) o `verificar-parquet` (solo Parquet). Sin errores, los datos coinciden.

Solo una persona genera y sube el manifiesto oficial (por Pull Request). **Nunca** se suben datos al repositorio.

## 7. Features y experimento base en MLflow

Tres pasos, cada uno ejecutable desde la terminal:

```bash
make data                        # CSV crudos -> Parquet unificado + manifiesto
make features                    # Features estación-hora (P1/P2) y perfiles (P3)
make train ARGS="--candidates"   # Baselines + candidatos de P1/P2/P3 en MLflow
make ui                          # Interfaz de MLflow (http://127.0.0.1:5000)
```

`features` escribe dos tablas en `data/processed/` y el umbral de P2 aprendido
**solo con entrenamiento** en `features_meta.json`.

Cada corrida registra en MLflow (experimento `mibici-equipo25`): los parámetros
para reproducirla (rango de fechas, `data_version` del manifiesto, semilla,
cortes de partición, horizontes e hiperparámetros), métricas nombradas por
partición (`test_*`, `validation_*`), etiquetas (`team`, `question`,
`git_commit`) y artefactos (modelo, figura y tablas de la corrida).

### Modelos y métricas por pregunta

| Pregunta | Baseline | Candidato | Métricas |
|---|---|---|---|
| P1 | `DummyRegressor` (media) | `HistGradientBoostingRegressor` | MAE, RMSE, WAPE |
| P2 | `DummyClassifier` (clase mayoritaria) | `HistGradientBoostingClassifier` | F1 macro, exactitud balanceada, recall por clase |
| P3 | `KMeans` | `GaussianMixture` | Silueta, tamaño de los grupos |

## 8. Flujo de trabajo con Git/GitHub

Rama principal protegida; todo cambio entra por Pull Request siguiendo GitHub Flow.
Convención de ramas: `feat/<tema>`, `fix/<tema>`, `docs/<tema>`, `test/<tema>`.
Convención de commits: Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`).
Cada integrante debe tener commits propios, al menos un Pull Request abierto y al
menos una revisión a otro integrante.

**Nunca** se sube a Git: datos de MiBici, la carpeta `mlruns/`/`mlartifacts/`,
modelos serializados ni credenciales (ver `.gitignore`).
