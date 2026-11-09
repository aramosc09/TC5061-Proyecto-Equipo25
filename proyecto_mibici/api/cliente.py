"""Cliente reproducible para probar /predict con un registro real de MiBici.

Ejemplo (con API iniciada):
    uv run --frozen python -m proyecto_mibici.api.cliente

Solo validar la entrada, sin API ni modelo exportado:
    uv run --frozen python -m proyecto_mibici.api.cliente --solo-validar
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import httpx
import pyarrow.dataset as ds

from proyecto_mibici.api.esquemas import (
    CAMPOS_HISTORIAL,
    PeticionPrediccion,
    RespuestaPrediccion,
)

RUTA_DATOS = Path("data/processed/tabla_supervisada.parquet")


def buscar_registro(
    ruta: Path, estacion: int, fecha_hora: datetime, conjunto: int
) -> dict:
    """Consulta solo las columnas necesarias, filtrando sobre Parquet."""
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Prepara los datos antes de ejecutar el cliente."
        )

    columnas = ["estacion", "hora", "conjunto", "salidas", *CAMPOS_HISTORIAL]
    filtro = (
        (ds.field("estacion") == estacion)
        & (ds.field("hora") == fecha_hora)
        & (ds.field("conjunto") == conjunto)
    )
    tabla = ds.dataset(str(ruta), format="parquet").to_table(
        columns=columnas, filter=filtro
    )
    if tabla.num_rows != 1:
        raise ValueError(
            f"Se esperaban 1 registro para estación={estacion}, "
            f"fecha_hora={fecha_hora.isoformat()}, conjunto={conjunto}; "
            f"se encontraron {tabla.num_rows}."
        )
    return tabla.to_pylist()[0]


def preparar_peticion(fila: dict) -> tuple[PeticionPrediccion, float]:
    """Valida las 24 características y guarda el observado fuera del cuerpo HTTP."""
    observado = float(fila["salidas"])
    if not math.isfinite(observado) or observado < 0:
        raise ValueError("La observación 'salidas' debe ser finita y no negativa.")

    registro = {
        "fecha_hora": fila["hora"],
        **{nombre: fila[nombre] for nombre in CAMPOS_HISTORIAL},
    }
    cuerpo = PeticionPrediccion.model_validate({"registros": [registro]})
    return cuerpo, observado


def solicitar_prediccion(
    url_base: str, cuerpo: PeticionPrediccion, timeout: float = 30
) -> RespuestaPrediccion:
    """Invoca la API HTTP y valida la estructura de su respuesta."""
    url = url_base.rstrip("/") + "/predict"
    try:
        respuesta = httpx.post(
            url, json=cuerpo.model_dump(mode="json"), timeout=timeout
        )
        respuesta.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"API respondió HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"No se pudo conectar con {url}. Verifica que FastAPI esté levantada: {exc}"
        ) from exc
    return RespuestaPrediccion.model_validate(respuesta.json())



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=RUTA_DATOS)
    parser.add_argument("--estacion", type=int, default=519)
    parser.add_argument("--fecha-hora", default="2026-09-10T22:00:00")
    parser.add_argument("--conjunto", type=int, default=2)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--solo-validar", action="store_true", help="No envía peticiones HTTP."
    )
    args = parser.parse_args(argv)

    try:
        fecha_hora = datetime.fromisoformat(args.fecha_hora)
        fila = buscar_registro(args.parquet, args.estacion, fecha_hora, args.conjunto)
        cuerpo, observado = preparar_peticion(fila)
        resultado = {
            "estacion": args.estacion,
            "fecha_hora": fecha_hora.isoformat(),
            "conjunto": args.conjunto,
            "salidas_observadas": observado,
            "variables_historicas": len(CAMPOS_HISTORIAL),
        }
        if args.solo_validar:
            resultado["estado"] = "peticion_validada_sin_envio"
            resultado["peticion"] = cuerpo.model_dump(mode="json")
        else:
            respuesta = solicitar_prediccion(args.api_url, cuerpo)
            if len(respuesta.predicciones) != 1:
                raise ValueError("La API no devolvió exactamente una predicción.")
            estimado = respuesta.predicciones[0].salidas_esperadas
            resultado["salidas_esperadas"] = estimado
            resultado["error_absoluto"] = round(abs(estimado - observado), 4)
            resultado["modelo"] = respuesta.modelo
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error del cliente: {exc}\n")

    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
