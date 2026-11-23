# Operación del sistema MiBici (Entrega 6)

Microservicios, CI/CD y FinOps sobre el modelo de demanda P1 (`mibici-equipo25-p1-demanda`).
Este documento reúne las secciones técnicas de la entrega. Cada PR agrega la suya: §1 orquestación
(PR 1), §2 pipeline de CI/CD (PR 2), §3 pruebas de carga y recursos (PR 3) y §4 análisis FinOps (PR 4).

## 1. Orquestación con Docker Compose

### 1.1 Servicios

`docker-compose.yml` levanta el sistema con un solo comando (`podman compose up -d --build` o
`docker compose up -d --build`). Los tres servicios usan **la misma imagen** (`Containerfile` de la
Entrega 4, que ya incluye MLflow 3.17.2), construida una sola vez:

| Servicio | Qué hace | Arranca cuando | Puerto (host) |
| --- | --- | --- | --- |
| `mlflow` | Servidor de tracking y Model Registry. SQLite y artefactos en el volumen `mibici-mlflow-data` | De inmediato; sano cuando responde `/health` | `127.0.0.1:5000` |
| `registrar-modelo` | De un solo uso: sube la copia exportada del `champion` al Registry de este servidor, con su linaje, y termina | `mlflow` sano | — |
| `api` | FastAPI de la Entrega 4. Carga `models:/mibici-equipo25-p1-demanda@champion` desde el Registry; sana cuando responde `/health` | `mlflow` sano y `registrar-modelo` terminado con código 0 | `127.0.0.1:8000` |

```text
                    red "mibici" (bridge)
 ┌──────────────────────────────────────────────────────────────┐
 │  registrar-modelo ──(1) registra v1 @champion──▶ mlflow:5000  │
 │                                                    │  ▲       │
 │                                volumen mibici-mlflow-data     │
 │                                                       │       │
 │  api:8000 ──(2) carga models:/…@champion ─────────────┘       │
 └──────┬───────────────────────────────────────────────┬───────┘
  127.0.0.1:8000 (/health, /model-info, /predict)   127.0.0.1:5000 (UI de MLflow)
```

### 1.2 Decisiones

- **MLflow con su propio almacenamiento.** El `mlflow.db` de cada integrante guarda rutas absolutas
  de Windows y versiones distintas, así que no puede montarse en un contenedor Linux. El servidor de
  Compose empieza vacío y `registrar-modelo` lo llena con la copia que `modelo exportar` dejó en
  `models/api/p1/`: sube **los mismos archivos**, sin reentrenar ni volver a serializar. La versión
  queda con los tags del linaje original (`source_run_id`, `git_commit`, `data_version`,
  `model_family`, `selection_metric`, `selection_value`) más `version_origen` y `registrado_desde`.
- **Idempotente.** Si el alias `champion` ya existe, `registrar-modelo` no crea otra versión
  (`"estado": "ya_registrado"`), así que `compose up` puede repetirse. `compose down -v` borra el
  volumen y el siguiente `up` vuelve a registrar desde cero.
- **La API carga desde el Registry** (`MODEL_SOURCE=registry`), lo que prueba dentro de un
  contenedor el modo que en la Entrega 4 solo se verificó en el host. Con `MODEL_SOURCE=local` usa
  la copia empaquetada en la imagen y deja de depender del servidor; ambas dan la misma predicción.
- **Una sola imagen** para los tres servicios: un solo build, una sola superficie de dependencias
  para pip-audit y menos espacio en disco y en el registro de imágenes.
- **Ejecutor de trabajos de MLflow desactivado** (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`).
  MLflow 3.17 arranca por defecto 7 procesos `huey` para evaluación de GenAI que este proyecto no
  usa. Con ellos, el servidor en reposo ocupaba **1.97 GB y ~113 % de CPU**; sin ellos y con un
  worker, **~370 MB**.
- **Límites de recursos de la API** (`API_CPUS=1.0`, `API_MEMORY=1g` por defecto). La prueba de
  carga y el costo por inferencia se miden contra estos límites, que se pueden variar sin tocar el
  YAML (por ejemplo `API_WORKERS=2 API_CPUS=2.0`).
- **Seguridad.** Los puertos se publican solo en `127.0.0.1`; los contenedores corren como
  `appuser` (UID 10001), no root; el servidor MLflow solo acepta los encabezados `Host` de la red
  interna y de localhost (`--allowed-hosts`).

### 1.3 Variables de entorno

Todas tienen un valor por defecto en el YAML. Para cambiarlas se descomentan en `.env` (a partir
de `.env.example`) o se pasan en la línea de comandos.

| Variable | Defecto | Uso |
| --- | --- | --- |
| `MIBICI_TAG` | `latest` | Etiqueta de la imagen `localhost/mibici-api` |
| `API_PORT` / `MLFLOW_PORT` | `8000` / `5000` | Puertos en el host (cambiar `MLFLOW_PORT` si el servidor local ya usa el 5000) |
| `MODEL_SOURCE` | `registry` | `registry` o `local` |
| `MODEL_URI` | `models:/mibici-equipo25-p1-demanda@champion` | Alias o versión que carga la API |
| `API_WORKERS` | `1` | Procesos de uvicorn |
| `API_CPUS` / `API_MEMORY` | `1.0` / `1g` | Límites del contenedor de la API |

### 1.4 Verificación (18-nov-2026, Windows 11 + Podman 5.8.3 con `docker-compose` 5.5.1)

| Prueba | Resultado |
| --- | --- |
| `podman compose up -d --build` (primera vez, con build) | 3 min 9 s; `mlflow` sano, `registrar-modelo` sale con 0, `api` arriba |
| `podman compose down -v` y `up -d` (sin build, volumen vacío) | 30 s hasta que `/health` responde |
| `registrar-modelo` | Registra la v1 `@champion` con `source_run_id=77ccfab…`, `git_commit=3f3e233…`, `data_version=09a6a584dd5d` |
| `GET /model-info` | `"source": "registry"`, versión 1, alias `champion`, linaje completo |
| `POST /predict` con `peticion.json` | `salidas_esperadas = 5.0772`, igual que con `MODEL_SOURCE=local` |
| `down` y `up` conservando el volumen | `"estado": "ya_registrado"`, no se duplica la versión |
| `API_WORKERS=2 API_CPUS=2.0 API_MEMORY=1536m podman compose up -d` | 2 procesos de uvicorn; límites de 2 CPU y 1.5 GB aplicados; al volver sin variables regresa a 1 CPU y 1 GB |
| Lotes en `/predict` | 1,000 registros → 1,000 predicciones (HTTP 200); 1,001 → HTTP 422 |
| `podman stats` en reposo | `api` ~205 MB; `mlflow` ~370 MB (1.97 GB antes de desactivar el ejecutor de trabajos) |
| Pruebas | `tests/test_api_registrar.py`: registro con linaje, carga desde el Registry, idempotencia, error sin copia exportada y rechazo de un modelo que no es P1 |


## 2. Pipeline de CI/CD con GitHub Actions

El flujo de integración continua (`.github/workflows/ci.yml`) automatiza la verificación de calidad, las pruebas automatizadas y la validación del despliegue contenerizado en cada `push` y `pull_request` hacia `main`, además de permitir su ejecución manual mediante `workflow_dispatch`.

### Arquitectura de los trabajos (Jobs)

| Trabajo (`job`) | Depende de | Qué ejecuta | Tiempo en Actions |
| :--- | :--- | :--- | :--- |
| `calidad` | — (en paralelo) | `ruff check`, `bandit -r proyecto_mibici` y `pip-audit -r requirements.txt` | *12s* |
| `pruebas` | — (en paralelo) | `uv sync --frozen --extra monitoreo` y `pytest tests -q` con caché de `uv` | *1m 1s* |
| `imagen` | `calidad`, `pruebas` | Genera modelo sintético (`modelo_humo.py`), levanta `docker compose up -d --build`, valida `/health`, `/model-info` (`registry`) y `/predict`, y limpia con `down -v` | *1m 42s* |

### Modelo sintético para CI (`modelo_humo.py`)

Dado que `models/api/p1/` está excluido del control de versiones (`.gitignore`), el job `imagen` ejecuta `python -m proyecto_mibici.api.modelo_humo` para entrenar un `HistGradientBoostingRegressor(loss="poisson", max_iter=5)` sobre datos aleatorios de las 30 columnas de `COLUMNAS_MODELO` en `float32`. El artefacto se guarda con `version="0"` y `model_family="humo"`, y el comando bloquea cualquier sobrescritura sobre un modelo real existente salvo que se indique explícitamente `--forzar`.

![Pipeline de CI/CD en verde en GitHub Actions](../../reports/figures/entrega6/ci_workflow_verde.png)

## 3. Pruebas de carga con Locust

### 3.1 Diseño del escenario

La carga se ejecuta contra `/predict` sobre el sistema levantado con Compose.
[`locustfile.py`](../../locustfile.py) genera entradas válidas para el esquema
Pydantic de P1 y usa una semilla configurable para reproducir la misma distribución.
Cada petición contiene lotes de registros, de modo que se mide también el costo de
procesamiento por lote permitido por la API.

| Parámetro | Valor usado |
| --- | --- |
| Usuarios concurrentes | 20 |
| Crecimiento | 5 usuarios por segundo |
| Duración | 60 segundos |
| Registros por petición | 10 |
| Semilla | 42 |
| Endpoint | `POST /predict` |
| Recursos de la API | 1 CPU y 1 GB por defecto |

Los archivos y gráficas se generan con:

```bash
uv sync --frozen --extra carga
make load-test
uv run --frozen --extra carga python -m proyecto_mibici.carga_reporte
```

El HTML generado por Locust contiene las gráficas de latencia y throughput, y el
historial CSV permite reproducirlas:

- [`Reporte interactivo de Locust`](../../reports/figures/entrega6/locust_report.html)
- [`Gráfica resumen de rendimiento`](../../reports/figures/entrega6/locust_desempeno.png)
- [`Resumen estructurado`](../../reports/monitoreo/carga/locust_2026-11-22_resumen.json)
- [`Estadísticas finales`](../../reports/monitoreo/carga/locust_2026-11-22_stats.csv)

### 3.2 Resultados

La prueba se realizó el 22 de noviembre de 2026 contra la API de Compose cargada
desde el Registry. La API respondió correctamente sin fallos:

| Métrica | Resultado |
| --- | --- |
| Solicitudes | 4,372 |
| Fallos | 0 (0 %) |
| Throughput promedio | 73.97 solicitudes/s |
| Latencia mediana | 8 ms |
| Latencia promedio | 10.32 ms |
| Latencia p95 | 26 ms |
| Latencia máxima | 103.34 ms |

El throughput mantuvo aproximadamente 74–78 solicitudes por segundo una vez
alcanzados los 20 usuarios. La latencia p95 permaneció mayormente entre 26 y
40 ms durante el periodo estable, sin respuestas con error.

![Desempeño de /predict bajo carga](../../reports/figures/entrega6/locust_desempeno.png)


### 3.3 Pruebas ampliadas con el champion real P1 v1 (PR 3)

La corrida histórica de 20 usuarios descrita en §§3.1–3.2 se conserva como
referencia anterior. Adicionalmente, el **22 de noviembre de 2026 (hora de
México)** se ejecutaron pruebas locales en **macOS Intel con Docker Desktop**,
utilizando `MODEL_SOURCE=local` y el modelo real
`mibici-equipo25-p1-demanda`, versión `1`, alias `champion`, familia
`gradient_boosting` (`data_version=09a6a584dd5d`). La identidad se confirmó
con `GET /model-info`. Los resultados corresponden a esta máquina y no
implican equivalencia de rendimiento con una instancia de nube.

**Datos y procedimiento.** Se generaron tres JSON a partir de las filas
históricas de `data/processed/tabla_supervisada.parquet`: `una` (1),
`estacion_dia` (24) y `sistema_hora` (466 predicciones por solicitud).
Las pruebas mixtas siguieron los escalones **10, 25, 50 y 100 usuarios**, con
120 segundos programados por nivel. Para comparar escalamiento, se mantuvieron
los JSON y el modelo, variando de **1 worker / 1 vCPU / 1 GiB RAM** a
**2 workers / 2 vCPU / 1 GiB RAM**. Antes de cada corrida se recreó el
contenedor `api` y se comprobó el límite efectivo con `docker inspect`;
para 2 workers se verificaron ambos procesos con `docker top`.

| Usuarios | 1 worker: RPS | 1 worker: p95 acumulado | 2 workers: RPS | 2 workers: p95 acumulado |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 7.70 | 480 ms | 7.50 | 55 ms |
| 25 | 18.00 | 220 ms | 18.50 | 470 ms |
| 50 | 26.15 | 4,500 ms | 24.70 | 1,700 ms |
| 100 | 25.20 | 4,400 ms | 22.80 | 4,700 ms |

Valores de `resumen_escalones.csv` generado desde los historiales de Locust;
las tasas de errores aproximadas fueron 0 % en los cuatro escalones.
El **p95 del historial es acumulado**, no independiente por escalón. Por tanto,
50 usuarios es el primer nivel con incumplimiento detectado en ambos resúmenes,
**no un punto exacto de saturación confirmado**; el generador conserva
`punto_saturacion_confirmado: null`. El p95 global final reportado en consola
fue 3,800 ms (1 worker) y 4,200 ms (2 workers), respectivamente, ambos
superiores al criterio p95 ≤ 500 ms. El throughput no mejoró sostenidamente
al duplicar workers y CPU.

| Configuración | CPU media API | CPU máxima | RAM media API | RAM máxima |
| --- | ---: | ---: | ---: | ---: |
| 1 worker / 1 vCPU | 71.02 % | 118.37 % | 227.51 MB | 278.80 MB |
| 2 workers / 2 vCPU | 141.65 % | 249.09 % | 433.99 MB | 505.70 MB |

Estos promedios provienen de los CSV completos del monitor y **pueden incluir
muestras anteriores al inicio efectivo de Locust**. Los porcentajes de CPU de
Docker se expresan respecto a un núcleo y los máximos son lecturas puntuales,
no utilización sostenida ni evidencia de superar la cuota configurada.

**Evidencias de escalones:**

- [1 worker: resumen de escalones](../../reports/carga/champion_1worker_1cpu_analisis/resumen_escalones.csv), [latencia](../../reports/carga/champion_1worker_1cpu_analisis/01_latencia.png), [throughput](../../reports/carga/champion_1worker_1cpu_analisis/02_throughput.png), [errores](../../reports/carga/champion_1worker_1cpu_analisis/03_errores.png) y [recursos](../../reports/carga/champion_1worker_1cpu_analisis/04_recursos.png).
- [2 workers: resumen de escalones](../../reports/carga/champion_2workers_2cpu_analisis/resumen_escalones.csv), [latencia](../../reports/carga/champion_2workers_2cpu_analisis/01_latencia.png), [throughput](../../reports/carga/champion_2workers_2cpu_analisis/02_throughput.png), [errores](../../reports/carga/champion_2workers_2cpu_analisis/03_errores.png) y [recursos](../../reports/carga/champion_2workers_2cpu_analisis/04_recursos.png).
- [Historial original, 1 worker](../../reports/carga/champion_1worker_1cpu_stats_history.csv) y [recursos](../../reports/carga/champion_1worker_1cpu_recursos.csv); [historial original, 2 workers](../../reports/carga/champion_2workers_2cpu_stats_history.csv) y [recursos](../../reports/carga/champion_2workers_2cpu_recursos.csv).

#### Comparación individual frente a lote de 466

Se efectuaron dos corridas separadas con **25 usuarios, 2 workers, 2 vCPU,
1 GiB**, 120 segundos programados y la misma versión del champion:

| Indicador | Una predicción por HTTP | 466 predicciones por HTTP |
| --- | ---: | ---: |
| Solicitudes completadas (CSV final) | 2,176 | 1,658 |
| Errores | 0 % | 0 % |
| Solicitudes/s | 18.011 | 13.738 |
| Predicciones/s | 18.011 | 6,401.792 |
| Latencia p50 | 51 ms | 250 ms |
| Latencia p95 | 610 ms | 2,200 ms |
| CPU promedio API | 117.35 % | 144.26 % |
| RAM promedio API | 489.49 MB | 504.10 MB |

El batching logró aproximadamente **355 veces más predicciones/s**, pero el
p95 de ambas modalidades **incumplió el objetivo de 500 ms**. El incremento
de capacidad observada no equivale a capacidad que cumple dicho objetivo.
Los totales del CSV final pueden diferir ligeramente de los impresos durante
el cierre de Locust.

Evidencia: [individual, estadísticas](../../reports/carga/champion_2workers_una_stats.csv),
[batch 466, estadísticas](../../reports/carga/champion_2workers_batch466_stats.csv),
[individual, recursos](../../reports/carga/champion_2workers_una_recursos.csv)
y [batch 466, recursos](../../reports/carga/champion_2workers_batch466_recursos.csv).

**Reproducción con Docker Desktop:** el ZIP del modelo real se extrae en
`models/api/p1/` y se valida con
`uv run --frozen python -m proyecto_mibici.api.modelo verificar`;
`models/` permanece excluido de Git. En macOS se usa `MLFLOW_PORT=5001` si
el 5000 está ocupado. Tras construir Compose con `MODEL_SOURCE=local`, para
cada configuración se recrea la API y luego se ejecuta la carga:

```bash
MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=1 API_CPUS=1.0 API_MEMORY=1g \
  docker compose up -d --no-deps --force-recreate api
MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=1 API_CPUS=1.0 API_MEMORY=1g \
  uv run --frozen --extra monitoreo python -m pruebas_carga.ejecutar \
  --runtime docker --prefijo champion_1worker_1cpu --tipo mixto \
  --escalones 10:120,25:120,50:120,100:120

MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=2 API_CPUS=2.0 API_MEMORY=1g \
  docker compose up -d --no-deps --force-recreate api
MLFLOW_PORT=5001 MODEL_SOURCE=local API_WORKERS=2 API_CPUS=2.0 API_MEMORY=1g \
  uv run --frozen --extra monitoreo python -m pruebas_carga.ejecutar \
  --runtime docker --prefijo champion_2workers_2cpu --tipo mixto \
  --escalones 10:120,25:120,50:120,100:120
```

Para las pruebas aisladas se mantiene la configuración de 2 workers,
se cambia `--tipo` a `una` o `sistema_hora`, y se usa `--escalones 25:120`
con un prefijo distinto por corrida. `pruebas_carga/graficas.py` genera
CSV y figuras a partir de los historiales, manteniendo la advertencia de
percentiles acumulados. La regla establecida fue p95 > 500 ms o errores > 1 %.

### 3.4 Monitoreo de uso de recursos

#### 3.4.1 Método

El uso de CPU y memoria se muestreó cada segundo con `podman stats`
simultáneamente a Locust. El módulo
[`monitoreo_recursos.py`](../../proyecto_mibici/monitoreo_recursos.py) conserva
la evidencia en CSV y genera una gráfica para API y MLflow. También admite
Docker mediante `--runtime docker`.

```bash
podman compose up -d --build
make load-test
make monitor-resources
```

Para que ambos procesos cubran exactamente la misma ventana, deben iniciarse
simultáneamente. `make monitor-resources` usa una duración de 60 segundos y un
intervalo de un segundo por defecto.

#### 3.4.2 Evidencia

La prueba final tuvo 20 usuarios concurrentes, lotes de 10 registros y 60
segundos de duración:

| Contenedor | CPU promedio | CPU máximo | Memoria promedio | Memoria máxima | Límite |
| --- | ---:| ---:| ---:| ---:| ---:|
| `mibici-api-1` | 18.88 % | 20.99 % | 205.71 MB | 212.50 MB | 1 GB |
| `mibici-mlflow-1` | 4.11 % | 4.78 % | 378.61 MB | 391.30 MB | 2 GB |

En `mibici-api-1`, el límite efectivo fue alrededor de 1.075 GB y la memoria
máxima observada representó aproximadamente 19.9 % de ese límite. En
`mibici-mlflow-1`, la memoria máxima representó 18.7 % de su límite observado
de aproximadamente 2.034 GB.

- [Evidencia CSV de recursos](../../reports/monitoreo/recursos/recursos_carga.csv)
- [Gráfica de CPU y memoria](../../reports/figures/entrega6/recursos_carga.png)

![Uso de CPU y memoria durante la carga](../../reports/figures/entrega6/recursos_carga.png)
## 4. Análisis FinOps: costo por inferencia (PR 4)

### 4.1 Unidad, mediciones y alcance

La unidad económica es **una inferencia para una estación-hora**, no una
solicitud HTTP. Se consideran **450 estaciones como hipótesis ilustrativa**,
24 horas/día y 30 días/mes: **324,000 inferencias mensuales** (base),
**3,240,000** (10×) y **32,400,000** (100×). Se asumen 730 horas de
facturación mensual de una VM y una utilización objetivo del **60 %**.

La corrida antigua de 20 usuarios y lotes de 10 (739.665 predicciones/s)
se conserva como antecedente, pero los escenarios principales de este PR
usan las mediciones realizadas el 22-nov-2026 con el **champion real P1 v1**,
2 workers, 2 vCPU y 1 GiB RAM:

| Modalidad | Solicitudes/s | Predicciones/s observadas | Respuesta media | p95 | Fallos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Una predicción | 18.011 | 18.011 | 143.60 ms | 610 ms | 0 % |
| Lote de 466 | 13.738 | 6,401.792 | 560.45 ms | 2,200 ms | 0 % |

**Ninguna de las dos modalidades cumple p95 ≤ 500 ms** en estas pruebas.
Las tasas son capacidades *observadas* bajo esa carga y hardware local, no
capacidades máximas ni tasas validadas para cumplimiento del SLO en la nube.
Las medidas de CPU y RAM, con su procedencia, se conservan en §3.3 y en los
archivos `reports/finops/champion_*/resumen_finops.json`.

### 4.2 Precios de referencia y fórmulas

Los precios consultados el **22-nov-2026**, con región y fuente, se guardan en
`references/precios_nube.csv`; la comparación principal utiliza el subconjunto
reproducible `references/precios_nube_2vcpu.csv`. Las opciones son:

- AWS EC2 `t3.small`: **USD 0.0209/h** (2 vCPU, 2 GiB, `us-east-1`).
- AWS EC2 `t3.medium`: **USD 0.0418/h** (2 vCPU, 4 GiB, `us-east-1`).
- Google Cloud Run: **2 vCPU, 1 GiB**, `us-central1`, con
  **USD 0.000024/vCPU-s**, **USD 0.0000025/GiB-s** y **USD 0.40 por millón
  de peticiones**, suponiendo escala a cero y sin nivel gratuito.

Fuentes: [AWS EC2 T3](https://aws.amazon.com/ec2/instance-types/t3/) y
[Cloud Run](https://cloud.google.com/run/pricing). Igual número de vCPU no
garantiza rendimiento equivalente entre Docker Desktop, EC2 y Cloud Run;
`T3` también está sujeta a créditos de CPU.

Definiciones: `N` = predicciones mensuales; `b` = registros por petición;
`P = ceil(N/b)` = peticiones; `C` = predicciones/s observadas; `u = 0.60`;
`t` = duración media observada por petición, con un mínimo modelado de 0.1 s.

- **VM siempre activa:** `costo_mes = precio_USD_h × 730`;
  `costo/inferencia = costo_mes/N` y `costo/HTTP = costo_mes/P`.
  A modo de techo teórico de productividad también se reporta
  `precio_USD_h/(C×3600)` al 100 % y `precio_USD_h/(C×3600×u)` al 60 %;
  **no son precios efectivos de inferencias que cumplan el SLO**.
- **Cloud Run, aproximación:** `segundos_facturados = max(N/C, P×max(t,0.1))`
  con concurrencia facturable **1**;
  `costo_mes = segundos_facturados × (vCPU×tarifa_vCPU_s + GiB×tarifa_GiB_s)
  + P×tarifa_por_peticion`.
  Se utilizan `t=0.143596 s` para individual y `t=0.560453 s` para batch.
  Se excluyen arranques en frío, facturación durante arranque/inactividad,
  escalamiento, red, almacenamiento, MLflow externo, impuestos y créditos
  gratuitos. Las duraciones son locales, no mediciones directas en Cloud Run.

La configuración de 1 vCPU permanece en el catálogo general, pero no se
usa para atribuirle tasas medidas con 2 vCPU. **Los costos AWS de una sola
instancia se mantienen fijos entre escenarios, sin demostrar que esa
instancia pueda atender la carga ni cumplir el p95 objetivo.**

### 4.3 Escenarios de demanda y costo mensual estimado (USD)

| Demanda | Predicciones/mes | EC2 t3.small (24/7) | EC2 t3.medium (24/7) | Cloud Run individual | Cloud Run batch 466 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 324,000 | 15.257 | 30.514 | 2.479 | 0.020 |
| 10× | 3,240,000 | 15.257 | 30.514 | 24.791 | 0.200 |
| 100× | 32,400,000 | 15.257 | 30.514 | 247.913 | 1.996 |

Las dos columnas de Cloud Run son **estimaciones separadas del mismo volumen
de inferencias**, una por modalidad; no deben sumarse. El costo por HTTP y el
costo por inferencia para cada escenario se encuentran en los CSV enlazados.

- [Escenarios FinOps — individual](../../reports/finops/champion_una/escenarios.csv), [resumen con supuestos](../../reports/finops/champion_una/resumen_finops.json) y [gráfica](../../reports/figures/entrega6/finops_champion_una.png).
- [Escenarios FinOps — batch 466](../../reports/finops/champion_batch466/escenarios.csv), [resumen con supuestos](../../reports/finops/champion_batch466/resumen_finops.json) y [gráfica](../../reports/figures/entrega6/finops_champion_batch466.png).

El escenario 100× individual implica **12.5 inferencias/s** de demanda media,
que exceden el **60 % de 18.011 ≈ 10.81 inferencias/s** observadas. Por tanto,
una sola instancia a esa utilización objetivo no puede asumirse suficiente.
Además, como p95 ya supera 500 ms bajo la carga medida, el dimensionamiento
final debe validarse con pruebas adicionales de SLO y ráfagas reales.

### 4.4 Sensibilidades, optimizaciones y límites

1. **Batching:** 466 inferencias/HTTP logra aproximadamente 355× más
   inferencias/s que 1 por HTTP en la carga observada, pero aumenta p95 de
   610 a 2,200 ms; usarlo para trabajos asíncronos o programados donde
   ese tiempo sea aceptable, no como recomendación universal para tiempo real.
2. **Dimensionamiento:** 2 workers / 2 vCPU incrementaron RAM media API
   de 227.51 a 433.99 MB sin mejorar sostenidamente el throughput mixto.
   Antes de pagar más CPU, medir por nivel de carga y repetir corridas.
3. **MLflow:** `MODEL_SOURCE=local` evita depender del Registry durante
   inferencia, sin eliminar por sí solo el costo de la infraestructura;
   el costo de un servidor MLflow independiente no está incluido.
4. **Infraestructura:** evaluar escala a cero y/o lotes programados;
   verificar costos reales de Cloud Run en despliegue (concurrencia,
   mínimo facturable, arranque en frío, autoscaling) y créditos CPU de EC2 T3.
5. **Metodología:** los p95 de historiales por escalones son acumulados,
   por lo que el punto exacto de saturación sigue **sin confirmarse**;
   medir por escalones independientes si se requiere un umbral preciso.
   Las estadísticas de CPU/RAM abarcan algunas muestras de preparación
   previas a Locust y se deben filtrar para un promedio estricto bajo carga.

Las estimaciones no son facturas ni una validación de la capacidad efectiva
de los proveedores. Tampoco se hizo aquí una comparación controlada de
**1 GiB frente a 1.5 GiB de RAM**, que permanece como trabajo futuro.

### 4.5 Cómo reproducir la estimación

Asegurar que existan los CSV históricos de Locust y recursos, además de los
resúmenes `reports/carga/champion_2workers_una_resumen_finops.json` y
`reports/carga/champion_2workers_batch466_resumen_finops.json`, generados
con datos reales. Ejecutar por separado las dos estimaciones:

```bash
uv run --frozen python -m proyecto_mibici.finops \
  --resumen reports/carga/champion_2workers_una_resumen_finops.json \
  --precios references/precios_nube_2vcpu.csv \
  --recursos reports/carga/champion_2workers_una_recursos.csv \
  --salida reports/finops/champion_una

uv run --frozen python -m proyecto_mibici.finops \
  --resumen reports/carga/champion_2workers_batch466_resumen_finops.json \
  --precios references/precios_nube_2vcpu.csv \
  --recursos reports/carga/champion_2workers_batch466_recursos.csv \
  --salida reports/finops/champion_batch466
```

El módulo genera una figura genérica
`reports/figures/entrega6/finops_escenarios.png` en cada corrida; guardar
una copia con el sufijo `champion_una` o `champion_batch466` inmediatamente
después de cada una para no sobrescribir el gráfico anterior. Los precios
y el archivo de modelo real no se generan automáticamente: `models/api/p1/`
se distribuye al equipo por un canal separado y **no se agrega a Git**.
