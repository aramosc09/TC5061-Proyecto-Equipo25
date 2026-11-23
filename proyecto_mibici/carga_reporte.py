"""Convierte el historial de Locust en una gráfica y resumen reproducibles."""

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generar_grafica(historial: Path, destino: Path) -> pd.DataFrame:
    datos = pd.read_csv(historial)
    datos = datos.loc[datos["Name"].eq("Aggregated")].copy()
    datos["segundo"] = datos["Timestamp"] - datos["Timestamp"].min()

    figura, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    ejes[0].plot(datos["segundo"], datos["User Count"], color="#2c7fb8", marker="o", ms=2)
    ejes[0].set_ylabel("Usuarios")
    ejes[0].grid(alpha=0.25)

    eje_rps = ejes[1].twinx()
    ejes[1].plot(datos["segundo"], datos["95%"], color="#d95f02", marker="o", ms=2)
    eje_rps.plot(datos["segundo"], datos["Requests/s"], color="#31a354", marker="o", ms=2)
    ejes[1].set_ylabel("Latencia p95 (ms)", color="#d95f02")
    eje_rps.set_ylabel("Peticiones por segundo", color="#31a354")
    ejes[1].grid(alpha=0.25)
    ejes[1].set_xlabel("Tiempo transcurrido (s)")
    ejes[1].legend(["Latencia p95"], loc="upper left")
    eje_rps.legend(["RPS"], loc="upper right")

    figura.suptitle("Prueba de carga /predict")
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=140)
    plt.close(figura)
    return datos


def generar_resumen(stats: Path, destino: Path) -> dict:
    datos = pd.read_csv(stats)
    agregado = datos.loc[datos["Name"].eq("Aggregated")].iloc[-1]
    resumen = {
        "endpoint": "/predict",
        "lote_registros": int(10),
        "usuarios": int(20),
        "spawn_rate": int(5),
        "duracion_segundos": int(60),
        "semilla": int(42),
        "request_count": int(agregado["Request Count"]),
        "failure_count": int(agregado["Failure Count"]),
        "requests_per_second": float(agregado["Requests/s"]),
        "median_response_time_ms": float(agregado["Median Response Time"]),
        "average_response_time_ms": float(agregado["Average Response Time"]),
        "min_response_time_ms": float(agregado["Min Response Time"]),
        "max_response_time_ms": float(agregado["Max Response Time"]),
        "p95_response_time_ms": float(agregado["95%"]),
    }
    destino.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resumen


if __name__ == "__main__":
    directorio = Path("reports/monitoreo/carga")
    generar_grafica(
        directorio / "locust_2026-11-22_stats_history.csv",
        Path("reports/figures/entrega6/locust_desempeno.png"),
    )
    generar_resumen(
        directorio / "locust_2026-11-22_stats.csv",
        directorio / "locust_2026-11-22_resumen.json",
    )
