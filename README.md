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
- **Tamaño del conjunto seleccionado:** 13,621,885 registros de viajes correspondientes a 36 meses.
- **Estaciones identificadas:** 473 estaciones distintas en los viajes.
- **Nomenclatura:** 484 estaciones registradas, de las cuales 471 coinciden con los viajes y tienen capacidad y coordenadas válidas.
- **Calidad de datos:** Se identificaron las estaciones 405 y 407 sin correspondencia en el catálogo, relacionadas con 3,954 viajes (0.0290%).
- **Trazabilidad:** Los resultados de las verificaciones se documentan en `references/validacion_datos_mibici.md` y el manifiesto se encuentra en `data/data_manifest.json`.
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
│   ├── ml_canvas_mibici.pdf
│   ├── arquitectura_mibici.md
│   ├── arquitectura_mibici.pdf
│   └── validacion_datos_mibici.md
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
git clone git@github.com:aramosc09/TC5061-Proyecto-Equipo25.git
cd TC5061-Proyecto-Equipo25

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


## 7. Organización del equipo y flujo de colaboración

### 7.1. Integrantes

El proyecto MiBici es desarrollado por el Equipo 25 de la materia TC5061 - Operaciones de Aprendizaje Automático (MLOps).

| Integrante | Matrícula |
|---|---|
| Alejandro Ramos Contreras | A01747461 |
| Ángel Alexis Anaya Alamea | A01741351 |
| Erick Fernando Carrillo Pineda | A01840029 |
| Jose Rigoberto Cibrian Jacobo | A01740550 |

Las responsabilidades de implementación de P1, P2 y P3 deberán confirmarse entre los integrantes y documentarse conforme se integren sus contribuciones.

### 7.2. Estrategia de control de versiones

El equipo utiliza Git y GitHub para gestionar el código, la documentación y la colaboración durante el desarrollo.

La estrategia de trabajo contempla:

- Utilizar `main` como rama principal del proyecto.
- Desarrollar nuevas funcionalidades mediante ramas independientes.
- Realizar commits descriptivos.
- Abrir Pull Requests para proponer cambios.
- Solicitar revisiones de otros integrantes antes de integrar modificaciones.
- Evitar commits y pushes directos a `main`.
- Configurar y verificar la protección de la rama principal.

### 7.3. Ramas de trabajo

Durante la revisión del repositorio se identificaron las siguientes ramas:

| Rama | Descripción |
|---|---|
| `main` | Rama principal del proyecto |
| `desarrolloP3` | Desarrollo de componentes asociados a P3 |
| `testing_branch` | Rama de pruebas y desarrollo inicial |
| `feature/documentacion-arquitectura` | Documentación de arquitectura, validación de datos y mejoras del README |

La existencia de estas ramas no implica que todos sus cambios estén integrados en `main`.

### 7.4. Flujo de colaboración mediante Pull Requests

Para desarrollar una nueva funcionalidad o modificar documentación, se seguirá el siguiente procedimiento:

1. Actualizar la información del repositorio remoto.
2. Crear una rama de trabajo a partir de `main`.
3. Realizar los cambios únicamente en la rama correspondiente.
4. Verificar los archivos modificados y ejecutar las pruebas aplicables.
5. Registrar los cambios mediante commits descriptivos.
6. Publicar la rama en GitHub.
7. Crear un Pull Request dirigido a `main`.
8. Solicitar la revisión de otro integrante.
9. Atender las observaciones recibidas.
10. Integrar los cambios únicamente después de las revisiones y verificaciones correspondientes.

### 7.5. Evidencia de colaboración

Para cumplir con los requisitos de la actividad, se deberá verificar y documentar lo siguiente:

- [ ] Rama `main` protegida mediante reglas de GitHub.
- [ ] Commits atribuibles a los cuatro integrantes.
- [ ] Al menos un Pull Request abierto por integrante.
- [ ] Al menos una revisión de Pull Request realizada por cada integrante.
- [ ] Evidencia de los cambios y revisiones en el historial del repositorio.

Estas actividades se consideran pendientes de verificación hasta confirmar su cumplimiento directamente en GitHub.

### 7.6. Buenas prácticas de colaboración

Para mantener la reproducibilidad y evitar conflictos entre integrantes:

- No subir archivos de datos originales ni procesados de MiBici.
- No publicar carpetas locales de MLflow ni modelos serializados.
- Mantener actualizado `.gitignore`.
- Registrar modificaciones a dependencias y configuraciones.
- Documentar los comandos utilizados para ejecutar los experimentos.
- Vincular los resultados de MLflow con la versión de datos y el commit correspondiente.
- Evitar modificar archivos desarrollados por otros integrantes sin coordinación previa.