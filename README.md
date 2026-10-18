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
- **Tamaño:** [COMPLETAR] viajes, [COMPLETAR] estaciones (salida de la ingesta; ver `data/data_manifest.json`).
- **Partición temporal (sin mezcla aleatoria):** entrenamiento oct 2023 – mar 2026, validación abr – jun 2026, prueba jul – sep 2026. Las variables de una predicción en *t* solo usan información hasta *t*; τ, escaladores y clustering se ajustan solo con entrenamiento.
- **Condiciones de uso:** [COMPLETAR con la licencia y fecha de consulta del portal]. Uso académico, con cita de la fuente.
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
│   ├── features.py          # Feature engineering reutilizable [PENDIENTE]
│   ├── plots.py             # Gráficas y mapas [PENDIENTE]
│   ├── tracking.py          # Registro de runs en MLflow [PENDIENTE]
│   └── modeling/
│       ├── train.py         # Baselines y modelos candidatos [PENDIENTE]
│       └── predict.py
├── Makefile                 # Comandos de la plantilla (ver `make help`)
├── pyproject.toml           # Dependencias con versión fija [AJUSTAR si usaron otro archivo]
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
