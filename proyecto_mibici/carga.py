"""Datos reproducibles para las pruebas de carga de P1."""

import json
import os
from pathlib import Path
import random

from proyecto_mibici.api.esquemas import PeticionPrediccion


def batch_size_desde_entorno() -> int:
    valor = int(os.getenv("LOCUST_BATCH_SIZE", "1"))
    if not 1 <= valor <= 1_000:
        raise ValueError("LOCUST_BATCH_SIZE debe estar entre 1 y 1000.")
    return valor


def crear_peticion(batch_size: int, rng: random.Random) -> dict:
    if not 1 <= batch_size <= 1_000:
        raise ValueError("batch_size debe estar entre 1 y 1000.")

    registros = []
    for _ in range(batch_size):
        hora = rng.randint(0, 23)
        dia = rng.randint(1, 28)
        mes = rng.randint(1, 12)
        fecha_hora = f"2026-{mes:02d}-{dia:02d}T{hora:02d}:00:00"

        salidas_lag = [rng.randint(0, 20) for _ in range(4)]
        llegadas_lag = [rng.randint(0, 20) for _ in range(4)]
        entradas = rng.randint(0, 10)
        salidas = rng.randint(0, 10)

        registro = {
            "fecha_hora": fecha_hora,
            "salidas_lag_1": salidas,
            "salidas_lag_3": max(0, salidas - rng.randint(0, 3)),
            "salidas_lag_24": rng.randint(0, 20),
            "salidas_lag_168": salidas_lag[0],
            "salidas_lag_336": salidas_lag[1],
            "salidas_lag_504": salidas_lag[2],
            "salidas_lag_672": salidas_lag[3],
            "salidas_media_3": round(rng.uniform(0, 20), 3),
            "salidas_media_24": round(rng.uniform(0, 20), 3),
            "salidas_media_168": round(rng.uniform(0, 20), 3),
            "salidas_media_4_semanas": sum(salidas_lag) / len(salidas_lag),
            "llegadas_lag_1": entradas,
            "llegadas_lag_3": max(0, entradas - rng.randint(0, 3)),
            "llegadas_lag_24": rng.randint(0, 20),
            "llegadas_lag_168": llegadas_lag[0],
            "llegadas_lag_336": llegadas_lag[1],
            "llegadas_lag_504": llegadas_lag[2],
            "llegadas_lag_672": llegadas_lag[3],
            "llegadas_media_3": round(rng.uniform(0, 20), 3),
            "llegadas_media_24": round(rng.uniform(0, 20), 3),
            "llegadas_media_168": round(rng.uniform(0, 20), 3),
            "llegadas_media_4_semanas": sum(llegadas_lag) / len(llegadas_lag),
            "balance_acumulado_3": round(rng.uniform(-20, 20), 3),
            "balance_acumulado_24": round(rng.uniform(-100, 100), 3),
        }
        registros.append(registro)

    peticion = PeticionPrediccion.model_validate({"registros": registros})
    return peticion.model_dump(mode="json")


def cargar_peticion_base() -> dict:
    ruta = Path(__file__).resolve().parents[1] / "peticion.json"
    return json.loads(ruta.read_text(encoding="utf-8"))
