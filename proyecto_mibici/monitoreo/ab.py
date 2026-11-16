"""Evaluación A/B reproducible de champion y challenger para P1."""

from datetime import datetime, timezone
import json

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from proyecto_mibici.api.modelo import REGISTERED_MODEL, cargar_modelo
from proyecto_mibici.config import FIGURES_DIR, MONITORING_REPORTS_DIR, ProjectConfig
from proyecto_mibici.monitoreo.periodos import cargar_periodo, definir_periodos, metadatos_tabla

matplotlib.use("Agg")

import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False)
DEFAULT_BATCH_SIZE = 100_000


def _metricas(y_real: np.ndarray, predicciones: np.ndarray) -> dict[str, float]:
    """Calcula métricas comparables sobre las mismas observaciones."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_real, predicciones))),
        "mae": float(mean_absolute_error(y_real, predicciones)),
        "r2": float(r2_score(y_real, predicciones)),
    }


def _predecir_por_lotes(modelo, features: pd.DataFrame, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo.")
    predicciones = [
        modelo.predecir(features.iloc[inicio:inicio + batch_size])
        for inicio in range(0, len(features), batch_size)
    ]
    return np.concatenate(predicciones) if predicciones else np.array([], dtype=float)


def _guardar_grafica_metricas(tabla: pd.DataFrame, destino) -> None:
    """Guarda una comparación legible de las métricas A/B."""
    figura, ejes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    colores = {"baseline": "#7f8c8d", "champion": "#2c7fb8", "challenger": "#d95f02"}
    for eje, metrica in zip(ejes, ("rmse", "mae", "r2")):
        barras = eje.bar(
            tabla["modelo"],
            tabla[metrica],
            color=[colores[modelo] for modelo in tabla["modelo"]],
        )
        eje.set_title(metrica.upper())
        eje.set_ylabel("Valor")
        eje.grid(axis="y", alpha=0.25)
        eje.set_axisbelow(True)
        eje.tick_params(axis="x", rotation=25)
        for barra in barras:
            eje.annotate(
                f"{barra.get_height():.4f}",
                (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )
    figura.suptitle("Comparación A/B de modelos P1")
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=140)
    plt.close(figura)


def evaluar(
    *,
    config: ProjectConfig | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    semilla: int = 42,
) -> dict:
    """Evalúa ambas versiones en evaluacion_ab y guarda resultados reproducibles."""
    config = config or ProjectConfig()
    mlflow_uri = mlflow.get_tracking_uri()
    periodos = definir_periodos(config)
    periodo = periodos["evaluacion_ab"]
    datos = cargar_periodo(periodo, config=config, semilla=semilla)
    features = datos.drop(columns=["estacion", "hora", "salidas"])
    y_real = datos["salidas"].to_numpy(dtype=float)

    modelos = {}
    predicciones = {}
    for rol in ("champion", "challenger"):
        uri = f"models:/{REGISTERED_MODEL}@{rol}"
        modelos[rol] = cargar_modelo("registry", model_uri=uri)
        predicciones[rol] = _predecir_por_lotes(modelos[rol], features, batch_size)

    baseline = datos["salidas_lag_168"].to_numpy(dtype=float)
    metricas = {
        rol: _metricas(y_real, predicciones[rol]) for rol in ("champion", "challenger")
    }
    metricas["baseline"] = _metricas(y_real, baseline)

    salida = config.path(MONITORING_REPORTS_DIR) / "ab"
    salida.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(
        {
            "modelo": ["baseline", "champion", "challenger"],
            "rmse": [
                metricas["baseline"]["rmse"],
                metricas["champion"]["rmse"],
                metricas["challenger"]["rmse"],
            ],
            "mae": [metricas["baseline"]["mae"], metricas["champion"]["mae"], metricas["challenger"]["mae"]],
            "r2": [metricas["baseline"]["r2"], metricas["champion"]["r2"], metricas["challenger"]["r2"]],
        }
    )
    tabla.to_csv(salida / "metricas_ab.csv", index=False, float_format="%.8f")
    grafica = config.path(FIGURES_DIR) / "entrega5" / "ab_metricas_modelos.png"
    _guardar_grafica_metricas(tabla, grafica)

    informe = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": mlflow_uri,
        "pregunta": "P1",
        "data_version": metadatos_tabla(config)["data_version"],
        "periodo": periodo.como_dict(),
        "filas": len(datos),
        "batch_size": batch_size,
        "semilla": semilla,
        "modelos": {
            rol: {
                "uri": f"models:/{modelos[rol].metadatos['registered_model']}@{rol}",
                **modelos[rol].metadatos,
            }
            for rol in ("champion", "challenger")
        },
        "metricas": metricas,
        "grafica_metricas": str(grafica.relative_to(config.root)),
        "mejor_modelo_por_rmse": min(
            ("champion", "challenger"), key=lambda rol: metricas[rol]["rmse"]
        ),
    }
    (salida / "resultado_ab.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return informe


@app.command()
def main(
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Filas por lote de predicción."),
    semilla: int = typer.Option(42, help="Semilla documentada para la carga reproducible."),
):
    """Ejecuta y guarda la comparación A/B sobre evaluacion_ab."""
    informe = evaluar(batch_size=batch_size, semilla=semilla)
    typer.echo(json.dumps(informe["metricas"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
