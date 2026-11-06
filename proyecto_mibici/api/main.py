"""API REST de demanda de MiBici (modelo P1, HistGradientBoostingRegressor).

Ejecutar localmente:
    uv run --frozen uvicorn proyecto_mibici.api.main:app --port 8000
Documentación interactiva en http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from proyecto_mibici.api.esquemas import (
    PeticionPrediccion,
    Prediccion,
    RespuestaPrediccion,
)
from proyecto_mibici.api.modelo import cargar_modelo

logger = logging.getLogger(__name__)

MENSAJE_SIN_MODELO = (
    "Modelo no disponible. Si MODEL_SOURCE=local, ejecuta "
    "'python -m proyecto_mibici.api.modelo exportar' antes de iniciar; "
    "si es 'registry', revisa MLFLOW_TRACKING_URI y MODEL_URI. "
    "Los detalles están en los registros del servidor."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al iniciar.

    Si la carga falla, la app arranca igual; /health y /predict responden 503.
    """
    app.state.modelo = None
    try:
        app.state.modelo = cargar_modelo()
        logger.info("Modelo cargado: %s", app.state.modelo.metadatos)
    except Exception:
        logger.exception("No se pudo cargar el modelo")
    yield


app = FastAPI(
    title="API de demanda MiBici (Equipo 25)",
    description=(
        "Predice las salidas esperadas de una estación de MiBici en la hora "
        "indicada, para decidir el rebalanceo de bicicletas de la siguiente hora."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def obtener_modelo(request: Request) -> Any:
    """Dependencia: el modelo servido, o 503 si no se pudo cargar."""
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(status_code=503, detail=MENSAJE_SIN_MODELO)
    return modelo


@app.get("/health", summary="Estado del servicio")
def health(_modelo: Any = Depends(obtener_modelo)) -> dict[str, str]:
    """200 si el modelo está cargado; 503 si no."""
    return {"status": "ok"}


@app.get("/model-info", summary="Linaje del modelo servido")
def model_info(modelo: Any = Depends(obtener_modelo)) -> dict[str, Any]:
    """Nombre, versión, alias, run de origen, commit, data_version y fuente."""
    return dict(modelo.metadatos)


@app.post("/predict", summary="Predicción de demanda")
def predict(
    cuerpo: PeticionPrediccion, modelo: Any = Depends(obtener_modelo)
) -> RespuestaPrediccion:
    """Demanda esperada de salidas en la hora indicada, una por registro.

    Es una estimación de la demanda, no un conteo garantizado. Acepta de 1 a
    1,000 registros, así que también sirve para lotes pequeños.
    """
    datos = pd.DataFrame([registro.a_fila() for registro in cuerpo.registros])
    try:
        valores = np.asarray(modelo.predecir(datos), dtype=float)
    except ValueError as exc:
        logger.exception("El modelo rechazó la predicción")
        raise HTTPException(
            status_code=500, detail="El modelo no pudo producir una predicción válida."
        ) from exc

    if (
        valores.shape != (len(cuerpo.registros),)
        or not np.all(np.isfinite(valores))
        or np.any(valores < 0)
    ):
        logger.error("Predicción fuera de rango: %s", valores)
        raise HTTPException(
            status_code=500, detail="El modelo devolvió una predicción fuera de rango."
        )

    predicciones = [Prediccion(salidas_esperadas=round(float(v), 4)) for v in valores]
    return RespuestaPrediccion(predicciones=predicciones, modelo=dict(modelo.metadatos))