"""Captura CPU y memoria en CSV de forma incremental durante una prueba Locust."""

import argparse
import csv
from pathlib import Path
import time

from proyecto_mibici.monitoreo_recursos import muestrear, seleccionar_runtime


CAMPOS = ("timestamp", "runtime", "contenedor", "cpu_porcentaje",
          "memoria_mb", "limite_memoria_mb")


def ejecutar(runtime: str, salida: Path, intervalo: float, duracion: float) -> int:
    """Escribe cada muestra inmediatamente para conservarlas al interrumpirse la carga."""
    if intervalo <= 0 or duracion <= 0:
        raise ValueError("Duración e intervalo deben ser positivos")
    runtime = seleccionar_runtime(runtime)
    salida.parent.mkdir(parents=True, exist_ok=True)
    contados = 0
    inicio = time.monotonic()
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        archivo.flush()
        while time.monotonic() - inicio < duracion:
            for fila in muestrear(runtime, ["mibici-api-1", "mibici-mlflow-1"]):
                escritor.writerow({columna: fila.get(columna) for columna in CAMPOS})
                contados += 1
            archivo.flush()
            transcurrido = time.monotonic() - inicio
            time.sleep(min(intervalo, max(0, duracion - transcurrido)))
    return contados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="auto", choices=("auto", "podman", "docker"))
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--intervalo", type=float, default=2)
    parser.add_argument("--duracion", type=float, default=480)
    args = parser.parse_args()
    print(f"Se guardaron {ejecutar(args.runtime, args.salida, args.intervalo, args.duracion)} muestras")


if __name__ == "__main__":
    main()
