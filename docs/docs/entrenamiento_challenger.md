# Reentrenamiento y versión challenger de P1

Describe cómo reentrenar el modelo de demanda (P1, `HistGradientBoostingRegressor`) con la ventana de datos ampliada, registrarlo como versión *challenger* y mantener intacto el alias `champion`.

## Flujo

1. **Reentrenar:** crea un run candidato en MLflow con los hiperparámetros del champion.
2. **Promover con `--alias challenger`:** crea una versión nueva en el Registry y le asigna ese alias. `champion` no cambia.
3. **Evaluar y comparar** (pendiente): puntuar champion y challenger sobre las mismas filas de `evaluacion_ab`.

## Requisitos

- Datos procesados: `uv run --frozen python -m proyecto_mibici.dataset procesar-datos`.
- Servidor de MLflow abierto en otra terminal:
  ```powershell
  uv run --frozen mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
  ```
- Variables de entorno en la terminal de trabajo (PowerShell):
  ```powershell
  $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
  $env:PYTHONIOENCODING = "utf8"
  ```
- Un modelo con alias `champion` en el Registry: el script lee de él los hiperparámetros.
- Cambios confirmados en Git antes de entrenar. `registry promote` rechaza runs con `git_worktree_dirty = true`.

## 1. Reentrenar

```powershell
uv run --frozen python -m proyecto_mibici.<modulo_reentrenamiento>
```

Por defecto hereda los hiperparámetros del champion (parámetros `resolved_*` de su run; si faltan usa 0.05, 500 iteraciones, 31 hojas, `min_samples_leaf` 200 y l2 1.0). Se pueden sobrescribir:

```powershell
uv run --frozen python -m proyecto_mibici.<modulo_reentrenamiento> --learning-rate 0.03 --max-iter 800
```

Opciones: `--learning-rate`, `--max-iter`, `--max-leaf-nodes`, `--min-samples-leaf` y `--l2-regularization`. Solo admite `--question p1`.

- Entrena con la ventana `entrenamiento_v2` (octubre de 2023 a junio de 2026), con tope de 2,000,000 de filas.
- Evalúa en `evaluacion_ab`. No calcula métricas de prueba (`evaluate_test=False`).
- Imprime una bitácora con rangos, hiperparámetros y su origen (heredado o sobrescrito) y, al final, el `RUN_ID`.

### Qué queda registrado en MLflow

| Elemento | Contenido |
|---|---|
| Nombre del run | `p1-hgb-challenger-<mes inicial>-<mes final>` |
| Parámetros | `entrenamiento_inicio`, `entrenamiento_fin`, `evaluacion_inicio`, `evaluacion_fin`, y `train_end_exclusive` y `validation_end_exclusive` con las ventanas realmente usadas |
| Etiquetas | `version_role = challenger`, `source_champion_run_id` |
| Modelo | Artefacto `model` con firma y ejemplo de entrada |

## 2. Promover como challenger

Primero la vista previa, que no modifica el Registry:

```powershell
uv run --frozen python -m proyecto_mibici.registry promote --question p1 --run-id <RUN_ID> --alias challenger --reason "v2: misma configuración que la v1 con datos hasta jun-2026"
```

Debe mostrar `"alias": "challenger"` y `"operation": "preview"`. Para registrar, repite el comando agregando `--approve`.

- `--alias` vale `champion` por defecto. Con otro alias solo se asigna ese alias.
- El alias se valida antes de registrar (se rechazan `latest` y nombres tipo `v1`), para no dejar versiones huérfanas.
- Con un alias distinto de `champion` se comprueba al final que `champion` sigue en la misma versión. El resultado incluye `champion_version` como evidencia.

## 3. Verificar

```powershell
uv run --frozen python -m proyecto_mibici.registry status
```

`champion` debe seguir en la misma versión que antes. El alias `challenger` se ve en la pestaña **Models** de MLflow, sobre la versión nueva.

Para servir el challenger con la API en modo `registry`:

```powershell
$env:MODEL_SOURCE = "registry"
$env:MODEL_URI = "models:/mibici-equipo25-p1-demanda@challenger"
```

## Cómo interpretar las métricas

`validation_rmse` se llama igual en el champion y en el challenger, pero **no mide las mismas filas**: el champion se validó sobre su ventana original y el challenger sobre `evaluacion_ab`. No se deben comparar directamente. La comparación válida del A/B es evaluar ambos modelos sobre las mismas filas de `evaluacion_ab`.

## Cambios en el código

- `proyecto_mibici/registry.py`: opción `--alias` en `promote`, con validación y protección de `champion`.
- `proyecto_mibici/tracking.py`: `candidate` acepta `run_name` y `extra_tags` opcionales; sin ellos se comporta como antes.
- Módulo de reentrenamiento: adaptador `DatosPorPeriodo` que carga los periodos y entrena con `CandidatosP1`.
- `tests/test_model_registry.py`: pruebas de `--alias`. Ejecutar con `uv run --frozen python -m pytest tests/test_model_registry.py -q`.

## Errores comunes

| Mensaje | Causa y solución |
|---|---|
| `Got unexpected extra argument (challenger)` | Falta un espacio entre `--run-id <ID>` y `--alias`. |
| `El run se entrenó con cambios sin commit` | Hacer commit y reentrenar. |
| `Configura MLFLOW_TRACKING_URI...` | Definir la variable y abrir el servidor de MLflow. |
| `REENTRENAMIENTO NO DISPONIBLE PARA 'P2'` | El flujo solo admite P1. |
| Error de codificación al imprimir | Definir `$env:PYTHONIOENCODING = "utf8"`. |