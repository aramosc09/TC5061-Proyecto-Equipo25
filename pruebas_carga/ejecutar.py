"""Ejecuta Locust y muestrea recursos en paralelo, sin crear resultados ficticios."""

import argparse
import os
from pathlib import Path
import subprocess  # nosec B404
import sys

from pruebas_carga.escenarios import leer_escalones


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("docker", "podman"), default="podman")
    parser.add_argument("--prefijo", default="escalones")
    parser.add_argument("--tipo", choices=("mixto", "una", "estacion_dia", "sistema_hora"), default="mixto")
    parser.add_argument("--escalones", default="10:120,25:120,50:120,100:120")
    args = parser.parse_args()
    if not args.prefijo.replace("-", "").replace("_", "").isalnum():
        parser.error("prefijo solo acepta letras, números, guiones y guiones bajos")
    duracion = sum(segundos for _, segundos in leer_escalones(args.escalones))
    destino = Path("reports/carga") / f"{args.prefijo}_recursos.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CARGA_PREFIJO": args.prefijo, "LOCUST_TIPO": args.tipo,
                "LOCUST_ESCALONES": args.escalones})
    # Margen para cubrir el arranque de Docker Compose y toda la prueba.
    # El monitor se detiene automáticamente cuando finaliza Locust.
    monitor = [sys.executable, "-m", "pruebas_carga.stats", "--runtime",
               args.runtime, "--duracion", str(duracion + 300), "--intervalo", "2",
               "--salida", str(destino)]
    carga = [args.runtime, "compose", "--profile", "carga", "run", "--rm", "locust"]
    proceso_stats = subprocess.Popen(monitor, env=env)  # nosec B603
    try:
        retorno = subprocess.call(carga, env=env)  # nosec B603
        if retorno != 0:
            raise SystemExit(f"Locust terminó con código {retorno}")
    finally:
        if proceso_stats.poll() is None:
            proceso_stats.terminate()
        proceso_stats.wait(timeout=15)
    print(f"Resultados de Locust en reports/carga/{args.prefijo}*")
    print(f"Muestras de recursos: {destino}")


if __name__ == "__main__":
    main()
