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
