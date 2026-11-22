"""Muestreo de CPU y memoria de contenedores durante las pruebas de carga."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False)


def seleccionar_runtime(runtime: str) -> str:
    runtime = runtime.lower().strip()
    if runtime == "auto":
        if shutil.which("podman"):
            return "podman"
        if shutil.which("docker"):
            return "docker"
        raise RuntimeError("Instala Docker o Podman.")
    if runtime not in {"docker", "podman"}:
        raise ValueError("runtime debe ser docker, podman o auto.")
    if not shutil.which(runtime):
        raise RuntimeError(f"{runtime} no está instalado.")
    return runtime


def _porcentaje(valor: str) -> float:
    texto = str(valor).strip().rstrip("%").replace(",", ".")
    try:
        return float(texto)
    except (TypeError, ValueError):
        return 0.0


def _memoria_mb(valor: str) -> float:
    texto = str(valor).strip().split()[0].replace(",", ".")
    unidades = {"B": 1 / 1024 / 1024, "KB": 1 / 1024, "MB": 1, "GB": 1024, "KiB": 1 / 1024,
                "MiB": 1, "GiB": 1024}
    for unidad, factor in sorted(unidades.items(), key=lambda item: -len(item[0])):
        if texto.upper().endswith(unidad.upper()):
            numero = texto[:-len(unidad)]
            return float(numero) * factor
    return float(texto) / 1024 / 1024


def muestrear(runtime: str, contenedores: list[str]) -> list[dict]:
    comando = [
        runtime,
        "stats",
        "--no-stream",
        "--format",
        "json",
    ]
    salida = subprocess.run(comando, check=True, capture_output=True, text=True)
    payload = json.loads(salida.stdout)
    filas = payload if isinstance(payload, list) else [payload]
    seleccionados = set(contenedores)
    filas_normalizadas = []
    for fila in filas:
        nombre = fila.get("Name") or fila.get("name") or fila.get("Container") or ""
        if seleccionados and nombre not in seleccionados:
            continue
        cpu = fila.get("CPUPerc") or fila.get("CPU") or fila.get("cpu_percent") or "0%"
        memoria = (
            fila.get("MemUsage")
            or fila.get("Mem")
            or fila.get("memory_usage")
            or fila.get("mem_usage")
            or "0MB"
        )
        limites = (
            fila.get("MemLimit")
            or fila.get("mem_limit")
            or (memoria.split(" / ")[1] if " / " in str(memoria) else "")
        )
        filas_normalizadas.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "contenedor": nombre,
                "cpu_porcentaje": _porcentaje(cpu),
                "memoria_mb": _memoria_mb(memoria.split(" / ")[0]),
                "limite_memoria_mb": _memoria_mb(limites) if limites else None,
            }
        )
    return filas_normalizadas


def guardar_grafica(datos: pd.DataFrame, destino: Path) -> None:
    figura, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    datos["segundo"] = (
        pd.to_datetime(datos["timestamp"], utc=True) - pd.to_datetime(datos["timestamp"], utc=True).min()
    ).dt.total_seconds()

    for contenedor, grupo in datos.groupby("contenedor"):
        ejes[0].plot(grupo["segundo"], grupo["cpu_porcentaje"], marker="o", ms=2, label=contenedor)
        ejes[1].plot(grupo["segundo"], grupo["memoria_mb"], marker="o", ms=2, label=contenedor)
    ejes[0].set_ylabel("CPU (%)")
    ejes[1].set_ylabel("Memoria (MB)")
    ejes[1].set_xlabel("Tiempo desde el inicio (s)")
    ejes[0].grid(alpha=0.25)
    ejes[1].grid(alpha=0.25)
    ejes[0].legend()
    ejes[1].legend()
    figura.suptitle("Uso de recursos durante la prueba de carga")
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=140)
    plt.close(figura)


@app.command()
def main(
    duracion: int = typer.Option(60, help="Duración total del muestreo en segundos."),
    intervalo: float = typer.Option(1.0, help="Intervalo entre muestras en segundos."),
    runtime: str = typer.Option("auto", help="docker, podman o auto."),
    contenedor: list[str] = typer.Option(
        ["mibici-api-1", "mibici-mlflow-1"],
        "--contenedor",
        help="Contenedores a muestrear.",
    ),
    salida: Path = typer.Option(
        Path("reports/monitoreo/recursos/recursos_carga.csv"),
        help="Archivo CSV de salida.",
    ),
):
    runtime = seleccionar_runtime(runtime)
    inicio = time.monotonic()
    muestras = []
    while time.monotonic() - inicio < duracion:
        muestras.extend(muestrear(runtime, contenedor))
        restante = inicio + duracion - time.monotonic()
        if restante > 0:
            time.sleep(min(intervalo, restante))

    if not muestras:
        raise RuntimeError(
            "No se encontraron muestras para los contenedores indicados. "
            "Verifica que Compose esté activo y los nombres sean correctos."
        )
    datos = pd.DataFrame(muestras)
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(salida, index=False, float_format="%.3f")
    guardar_grafica(datos, Path("reports/figures/entrega6/recursos_carga.png"))
    resumen = (
        datos.groupby("contenedor", as_index=False)
        .agg(
            cpu_promedio_porcentaje=("cpu_porcentaje", "mean"),
            cpu_maximo_porcentaje=("cpu_porcentaje", "max"),
            memoria_promedio_mb=("memoria_mb", "mean"),
            memoria_maxima_mb=("memoria_mb", "max"),
        )
    )
    typer.echo(json.dumps(resumen.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
