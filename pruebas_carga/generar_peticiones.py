"""Genera cuerpos HTTP con registros reales de la tabla supervisada."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import pandas as pd

from proyecto_mibici.api.esquemas import CAMPOS_HISTORIAL, MAX_REGISTROS, PeticionPrediccion


TIPOS = ("una", "estacion_dia", "sistema_hora")


def construir_cuerpos(filas: pd.DataFrame, estacion: int, fecha: str, hora: int) -> dict[str, dict]:
    """Construye tres solicitudes verificables desde filas estación-hora."""
    requeridas = {"estacion", "hora", *CAMPOS_HISTORIAL}
    faltantes = sorted(requeridas.difference(filas.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")
    if not 0 <= hora <= 23:
        raise ValueError("hora debe estar entre 0 y 23")

    datos = filas.copy()
    datos["hora"] = pd.to_datetime(datos["hora"])
    datos["estacion"] = datos["estacion"].astype(int)
    datos = datos.drop_duplicates(subset=["estacion", "hora"], keep=False)
    inicio = pd.Timestamp(fecha)
    fin = inicio + timedelta(days=1)
    datos = datos.loc[datos["hora"].ge(inicio) & datos["hora"].lt(fin)]
    objetivo = inicio + timedelta(hours=hora)
    dia_estacion = datos.loc[datos["estacion"].eq(estacion)].sort_values("hora")
    hora_estacion = dia_estacion.loc[dia_estacion["hora"].eq(objetivo)]
    sistema = datos.loc[datos["hora"].eq(objetivo)].sort_values("estacion")

    if len(hora_estacion) != 1:
        raise ValueError(f"No hay un registro único para estación {estacion} a las {hora:02d}:00")
    if len(dia_estacion) != 24 or dia_estacion["hora"].nunique() != 24:
        raise ValueError(f"La estación {estacion} no tiene 24 horas completas el día {fecha}")
    if not 1 <= len(sistema) <= MAX_REGISTROS:
        raise ValueError(f"El lote horario contiene {len(sistema)} registros (límite {MAX_REGISTROS})")

    def cuerpo(subconjunto: pd.DataFrame) -> dict:
        registros = []
        for fila in subconjunto.to_dict("records"):
            registros.append(
                {"fecha_hora": pd.Timestamp(fila["hora"]).isoformat(),
                 **{campo: float(fila[campo]) for campo in CAMPOS_HISTORIAL}}
            )
        return PeticionPrediccion.model_validate({"registros": registros}).model_dump(mode="json")

    return {
        "una": cuerpo(hora_estacion),
        "estacion_dia": cuerpo(dia_estacion),
        "sistema_hora": cuerpo(sistema),
    }


def leer_parquet(ruta: Path, fecha: str, conjunto: int) -> pd.DataFrame:
    """Lee únicamente las columnas del día solicitado desde Parquet particionado o simple."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}; ejecuta antes el proceso de datos")
    import pyarrow.dataset as ds

    inicio = datetime.fromisoformat(fecha)
    fin = inicio + timedelta(days=1)
    filtro = ((ds.field("hora") >= inicio) & (ds.field("hora") < fin)
              & (ds.field("conjunto") == conjunto))
    columnas = ["estacion", "hora", *CAMPOS_HISTORIAL]
    return ds.dataset(str(ruta), format="parquet").to_table(
        columns=columnas, filter=filtro
    ).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path("data/processed/tabla_supervisada.parquet"))
    parser.add_argument("--fecha", default="2026-09-10")
    parser.add_argument("--estacion", type=int, default=2)
    parser.add_argument("--hora", type=int, default=22)
    parser.add_argument("--conjunto", type=int, default=2)
    parser.add_argument("--salida", type=Path, default=Path("pruebas_carga/peticiones"))
    args = parser.parse_args()
    cuerpos = construir_cuerpos(leer_parquet(args.parquet, args.fecha, args.conjunto),
                                args.estacion, args.fecha, args.hora)
    args.salida.mkdir(parents=True, exist_ok=True)
    for tipo, cuerpo in cuerpos.items():
        archivo = args.salida / f"{tipo}.json"
        archivo.write_text(json.dumps(cuerpo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{archivo}: {len(cuerpo['registros'])} registros reales")


if __name__ == "__main__":
    main()
