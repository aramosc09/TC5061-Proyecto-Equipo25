"""Estimación FinOps reproducible a partir de CSV de carga y precios públicos."""

import argparse
import csv
import json
import math
from pathlib import Path

HORAS_MES = 730
SEGUNDOS_HORA = 3600


def capacidad_observada(resumen: dict) -> float:
    """Calcula predicciones útiles por segundo a partir de un ensayo de Locust."""
    lote = int(resumen["lote_registros"])
    solicitudes = int(resumen["request_count"])
    fallos = int(resumen["failure_count"])
    rps = float(resumen["requests_per_second"])
    if lote < 1 or solicitudes <= 0 or fallos < 0 or fallos >= solicitudes or rps <= 0:
        raise ValueError("Resumen Locust sin solicitudes válidas suficientes")
    return rps * (solicitudes - fallos) / solicitudes * lote


def escenarios(estaciones: int, dias: int = 30) -> dict[str, int]:
    """Genera demanda mensual para estaciones activas y multiplicadores."""
    if estaciones < 1 or dias < 1:
        raise ValueError("Estaciones y días deben ser positivos")
    base = estaciones * 24 * dias
    return {"base": base, "10x": 10 * base, "100x": 100 * base}


def estimar_costos(
    resumen: dict,
    precios: list[dict],
    estaciones: int = 450,
    dias: int = 30,
    utilizacion: float = 0.60,
    concurrencia_facturacion: int = 1,
) -> list[dict]:
    """Compara VM permanente y Cloud Run con una predicción como unidad."""
    if not 0 < utilizacion <= 1 or concurrencia_facturacion < 1:
        raise ValueError("Utilización y concurrencia de facturación inválidas")
    capacidad = capacidad_observada(resumen)
    lote = int(resumen["lote_registros"])
    resultados = []
    for nombre, cantidad in escenarios(estaciones, dias).items():
        peticiones = math.ceil(cantidad / lote)
        tasa_media = cantidad / (dias * 24 * 3600)
        for precio in precios:
            modelo = precio["modelo_facturacion"]
            if modelo == "siempre_encendido":
                costo = float(precio["usd_hora"]) * HORAS_MES
                costo_inferencia_capacidad = float(precio["usd_hora"]) / (
                    capacidad * SEGUNDOS_HORA * utilizacion
                )
                costo_inferencia_100pct = float(precio["usd_hora"]) / (
                    capacidad * SEGUNDOS_HORA
                )
                segundos_activos_estimados = None
            elif modelo == "escala_a_cero":
                # Modelo simplificado con una sola concurrencia facturable por defecto:
                # mínimo 100 ms por solicitud; la facturación real depende de concurrencia y duración.
                duracion_promedio_solicitud = max(
                    float(resumen.get("average_response_time_ms", 100.0)) / 1000,
                    0.1,
                )
                segundos_activos_estimados = max(
                    cantidad / capacidad,
                    peticiones * duracion_promedio_solicitud / concurrencia_facturacion,
                )
                costo = segundos_activos_estimados * (
                    float(precio["usd_vcpu_segundo"]) * float(precio["vcpus"])
                    + float(precio["usd_gib_segundo"]) * float(precio["gib_memoria"])
                ) + peticiones * float(precio["usd_millon_peticiones"]) / 1_000_000
                costo_inferencia_capacidad = None
                costo_inferencia_100pct = None
            else:
                raise ValueError(f"Modelo de facturación no soportado: {modelo}")
            resultados.append({
                "escenario": nombre, "predicciones_mes": cantidad,
                "peticiones_estimadas_mes": peticiones, "tasa_media_pred_s": tasa_media,
                "proveedor": precio["proveedor"], "opcion": precio["instancia"],
                "region": precio["region"], "facturacion": modelo,
                "costo_mensual_usd_estimado": round(costo, 6),
                "costo_por_inferencia_usd": round(costo / cantidad, 11),
                "costo_por_peticion_http_usd": round(costo / peticiones, 11),
                "costo_teorico_capacidad_100pct_usd": (
                    round(costo_inferencia_100pct, 11)
                    if costo_inferencia_100pct is not None else None
                ),
                "costo_teorico_capacidad_60pct_usd": (
                    round(costo_inferencia_capacidad, 11)
                    if costo_inferencia_capacidad is not None else None
                ),
                "segundos_facturados_estimados": (
                    round(segundos_activos_estimados, 2)
                    if segundos_activos_estimados is not None else None
                ),
                "fuente_precio": precio["fuente_url"],
            })
    return resultados


def leer_precios(ruta: Path) -> list[dict]:
    """Lee datos en CSV con fuente, región y fecha por cada proveedor."""
    with ruta.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if len(filas) < 2 or any(not fila.get("fuente_url") or not fila.get("fecha_consulta") for fila in filas):
        raise ValueError("Los precios requieren al menos dos opciones con URL y fecha")
    return filas


def resumir_recursos(ruta: Path) -> list[dict]:
    """Calcula CPU y memoria observadas por contenedor desde el CSV de Podman."""
    with ruta.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise ValueError(f"CSV de recursos vacío: {ruta}")
    grupos: dict[str, dict[str, list[float]]] = {}
    for fila in filas:
        nombre = fila["contenedor"]
        grupo = grupos.setdefault(nombre, {"cpu": [], "memoria": []})
        grupo["cpu"].append(float(fila["cpu_porcentaje"]))
        grupo["memoria"].append(float(fila["memoria_mb"]))
    return [
        {"contenedor": nombre, "cpu_promedio_pct": round(sum(valores["cpu"]) / len(valores["cpu"]), 2),
         "cpu_maximo_pct": round(max(valores["cpu"]), 2),
         "memoria_promedio_mb": round(sum(valores["memoria"]) / len(valores["memoria"]), 2),
         "memoria_maxima_mb": round(max(valores["memoria"]), 2)}
        for nombre, valores in sorted(grupos.items())
    ]


def graficar_costos(tabla: list[dict], salida: Path) -> None:
    """Representa costos mensuales por escenario, con escala logarítmica."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    escenarios_unicos = ["base", "10x", "100x"]
    opciones = list(dict.fromkeys(registro["opcion"] for registro in tabla))
    posiciones = list(range(len(escenarios_unicos)))
    ancho = 0.75 / len(opciones)
    figura, eje = plt.subplots(figsize=(8.8, 4.7))
    for i, opcion in enumerate(opciones):
        valores = [next(registro["costo_mensual_usd_estimado"] for registro in tabla
                        if registro["escenario"] == escenario and registro["opcion"] == opcion)
                   for escenario in escenarios_unicos]
        eje.bar([pos - 0.375 + (i + 0.5) * ancho for pos in posiciones],
                valores, width=ancho, label=opcion)
    eje.set_xticks(posiciones, escenarios_unicos)
    eje.set_ylabel("Costo estimado por mes (USD, escala log)")
    eje.set_yscale("log")
    eje.set_title("Escenarios de demanda: cómputo estimado sin impuestos ni red")
    eje.legend(fontsize=8)
    eje.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(salida, dpi=160)
    plt.close(figura)


def ejecutar(
    resumen_path: Path,
    precios_path: Path,
    salida: Path,
    estaciones: int,
    utilizacion: float,
    recursos_path: Path | None = None,
) -> dict:
    """Escribe tabla de escenarios y resumen del cálculo."""
    resumen = json.loads(resumen_path.read_text(encoding="utf-8"))
    precios = leer_precios(precios_path)
    registros = estimar_costos(resumen, precios, estaciones=estaciones, utilizacion=utilizacion)
    salida.mkdir(parents=True, exist_ok=True)
    with (salida / "escenarios.csv").open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(registros[0]))
        writer.writeheader()
        writer.writerows(registros)
    graficar_costos(registros, Path("reports/figures/entrega6/finops_escenarios.png"))
    resultado = {
        "fuente_recursos": str(recursos_path) if recursos_path and recursos_path.exists() else None,
        "recursos_observados": resumir_recursos(recursos_path) if recursos_path and recursos_path.exists() else [],
        "fuente_locust": str(resumen_path), "fuente_precios": str(precios_path),
        "capacidad_observada_pred_s": round(capacidad_observada(resumen), 3),
        "lote_del_ensayo": int(resumen["lote_registros"]),
        "estaciones_supuesto": estaciones, "horas_mes_vm": HORAS_MES,
        "utilizacion_objetivo_vm": utilizacion,
        "concurrencia_facturable_cloud_run": 1,
        "modelo_benchmark": resumen.get("modelo"),
        "version_modelo": resumen.get("version_modelo"),
        "workers_benchmark": resumen.get("workers"),
        "vcpus_benchmark": resumen.get("vcpus"),
        "latencia_p95_ms": resumen.get("latencia_p95_ms"),
        "cumple_objetivo_p95": resumen.get("cumple_objetivo_p95"),
        "estado": (
            "benchmark_fuera_objetivo_p95"
            if resumen.get("cumple_objetivo_p95") is False
            else "benchmark_dentro_objetivo_p95"
            if resumen.get("cumple_objetivo_p95") is True
            else "estimacion_sin_evaluacion_slo"
        ),
        "limitaciones": [
            "La capacidad observada no equivale necesariamente a la "
            "capacidad sostenida que cumple p95 <= 500 ms.",
            "Las mediciones de Docker local no garantizan el mismo "
            "rendimiento en infraestructura de nube.",
            "El benchmark de 2 vCPU no valida el rendimiento de "
            "Cloud Run configurado con 1 vCPU.",
            "Las instancias AWS EC2 T3 utilizan creditos de CPU; "
            "los cargos adicionales no estan incluidos.",
            "Cloud Run utiliza concurrencia facturable 1 y un minimo "
            "de 0.1 segundos por solicitud como hipotesis.",
            "Los costos son estimaciones, no importes facturados.",
            "No incluye transferencia de red, almacenamiento, "
            "MLflow externo ni impuestos.",
            "Se excluye el nivel gratuito de Cloud Run.",
        ],
    }
    (salida / "resumen_finops.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumen", type=Path, default=Path(
        "reports/monitoreo/carga/locust_2026-11-22_resumen.json"
    ))
    parser.add_argument("--precios", type=Path, default=Path("references/precios_nube.csv"))
    parser.add_argument("--salida", type=Path, default=Path("reports/finops"))
    parser.add_argument("--recursos", type=Path, default=Path(
        "reports/monitoreo/recursos/recursos_carga.csv"
    ))
    parser.add_argument("--estaciones", type=int, default=450)
    parser.add_argument("--utilizacion", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(ejecutar(args.resumen, args.precios, args.salida,
                             args.estaciones, args.utilizacion, args.recursos),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
