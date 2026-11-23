"""Crea gráficas y resúmenes reproducibles de Locust y de podman stats."""

import argparse
import json
from pathlib import Path
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGISTROS_EN_NOMBRE = re.compile(r"\[(\d+)\]")


def lotes_por_nombre(nombre: str) -> int:
    """Obtiene registros por POST, excluyendo el endpoint de salud."""
    resultado = REGISTROS_EN_NOMBRE.search(str(nombre))
    return int(resultado.group(1)) if resultado and str(nombre).startswith("POST") else 0


def extraer_series(historial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distingue filas agregadas y por endpoint sin contarlas dos veces."""
    historial = historial.copy()
    for columna in ("Timestamp", "User Count", "Requests/s", "Failures/s", "50%", "95%"):
        historial[columna] = pd.to_numeric(historial[columna], errors="coerce")
    historial["Name"] = historial["Name"].fillna("")
    agregado = historial.loc[historial["Name"].eq("Aggregated")].sort_values("Timestamp").copy()
    por_tipo = historial.loc[~historial["Name"].eq("Aggregated")].copy()
    if agregado.empty:
        raise ValueError("No se encontró la fila Aggregated en el CSV de Locust")
    agregado["segundo"] = agregado["Timestamp"] - agregado["Timestamp"].min()
    if por_tipo.empty:
        # Historial legado: solo un tipo registrado; inferir el lote desde ese nombre si está disponible.
        por_tipo = agregado.copy()
        por_tipo["Name"] = "POST legado [10]"
    por_tipo["lote"] = por_tipo["Name"].map(lotes_por_nombre)
    por_tipo["predicciones_s"] = por_tipo["Requests/s"].fillna(0) * por_tipo["lote"]
    return agregado, por_tipo


def resumir_escalones(agregado: pd.DataFrame, por_tipo: pd.DataFrame) -> pd.DataFrame:
    """Resume los tramos de usuarios efectivamente observados en los CSV."""
    datos = []
    for usuarios, tramo in agregado.groupby("User Count", dropna=True):
        if usuarios <= 0 or tramo["Timestamp"].max() - tramo["Timestamp"].min() < 30:
            # Las rampas cortas no son un escalón estable medido.
            continue
        fin = tramo.sort_values("Timestamp").iloc[-1]
        inicio = tramo.sort_values("Timestamp").iloc[0]
        pred = por_tipo.loc[por_tipo["Timestamp"].isin(tramo["Timestamp"])]
        tasa = float(tramo["Failures/s"].fillna(0).sum()) / max(1e-12, float(tramo["Requests/s"].fillna(0).sum()))
        datos.append({
            "usuarios": int(usuarios), "duracion_observada_s": float(fin["Timestamp"] - inicio["Timestamp"]),
            "rps_observado": float(tramo["Requests/s"].fillna(0).median()),
            "predicciones_s_observadas": float(pred.groupby("Timestamp")["predicciones_s"].sum().median()),
            "p50_acumulado_ms": float(fin["50%"]), "p95_acumulado_ms": float(fin["95%"]),
            "tasa_error_aproximada": float(tasa),
            "umbral_incumplido": bool(fin["95%"] > 500 or tasa > 0.01),
        })
    return pd.DataFrame(datos).sort_values("usuarios")


def generar(historial_csv: Path, recursos_csv: Path | None, salida: Path) -> dict:
    """Genera cuatro gráficas y resumen, sin presentar percentiles acumulados como independientes."""
    salida.mkdir(parents=True, exist_ok=True)
    historial = pd.read_csv(historial_csv)
    agregado, por_tipo = extraer_series(historial)
    tramos = resumir_escalones(agregado, por_tipo)
    tramos.to_csv(salida / "resumen_escalones.csv", index=False)
    tiempo = agregado["segundo"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for nombre, grupo in por_tipo.groupby("Name"):
        if nombre.startswith("GET"):
            continue
        grupo = grupo.sort_values("Timestamp")
        ax.plot(grupo["Timestamp"] - agregado["Timestamp"].min(), grupo["95%"], label=nombre)
    ax.axhline(500, linestyle="--", color="black", label="Umbral p95 (500 ms)")
    ax.set(xlabel="Tiempo transcurrido (s)", ylabel="p95 histórico acumulado (ms)",
           title="Latencia de /predict por tipo de petición")
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(salida / "01_latencia.png", dpi=150); plt.close(fig)

    por_segundo = por_tipo.groupby("Timestamp", as_index=False).agg(pred_s=("predicciones_s", "sum"))
    fig, ax = plt.subplots(figsize=(9, 4.4)); ax.plot(tiempo, agregado["Requests/s"], label="Peticiones/s")
    ax.plot(por_segundo["Timestamp"] - agregado["Timestamp"].min(), por_segundo["pred_s"],
            label="Predicciones/s")
    ax.set(xlabel="Tiempo transcurrido (s)", ylabel="Tasa por segundo", title="Peticiones y predicciones")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(salida / "02_throughput.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    if not tramos.empty:
        ax.bar(tramos["usuarios"].astype(str), 100 * tramos["tasa_error_aproximada"])
    ax.axhline(1, linestyle="--", color="black", label="Umbral 1%")
    ax.set(xlabel="Usuarios observados", ylabel="Errores/s ÷ peticiones/s (%)",
           title="Tasa aproximada de errores por nivel de usuarios")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(salida / "03_errores.png", dpi=150); plt.close(fig)

    recursos_disponibles = bool(recursos_csv and recursos_csv.exists())
    if recursos_disponibles:
        recursos = pd.read_csv(recursos_csv)
        recursos["momento"] = pd.to_datetime(recursos["timestamp"], utc=True)
        recursos["segundo"] = recursos["momento"].astype("int64") / 1e9 - agregado["Timestamp"].min()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for nombre, datos in recursos.groupby("contenedor"):
            if "api" not in nombre:
                continue
            ax1.plot(datos["segundo"], datos["cpu_porcentaje"], label=nombre)
            ax2.plot(datos["segundo"], datos["memoria_mb"], label=nombre)
        ax1.set_ylabel("CPU (% del runtime)"); ax2.set_ylabel("Memoria (MB)")
        ax2.set_xlabel("Tiempo relativo al inicio de Locust (s)")
        ax1.set_title("Recursos API, alineados con la prueba de carga")
        for ax in (ax1, ax2):ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(salida / "04_recursos.png", dpi=150); plt.close(fig)
    # Ningún umbral puede considerarse probado si solamente existe el ensayo de 20 usuarios.
    resultado = {"fuente": str(historial_csv), "recursos": str(recursos_csv) if recursos_disponibles else None,
                 "niveles_estables": tramos["usuarios"].tolist() if not tramos.empty else [],
                 "nota_percentiles": "Los percentiles del historial son acumulados, no independientes por escalón.",
                 "punto_saturacion_confirmado": None}
    (salida / "resultado_graficas.json").write_text(json.dumps(resultado, indent=2) + "\n", encoding="utf-8")
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historial", type=Path, required=True)
    parser.add_argument("--recursos", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generar(args.historial, args.recursos, args.salida), indent=2))


if __name__ == "__main__":
    main()
