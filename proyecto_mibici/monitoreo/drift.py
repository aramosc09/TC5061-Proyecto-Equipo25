"""Reporte de drift y calidad de datos de P1 con Evidently, con registro opcional en MLflow."""

from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd
import typer

from proyecto_mibici.config import DEFAULT_RANDOM_STATE, MONITORING_REPORTS_DIR, ProjectConfig
from proyecto_mibici.modeling.datasets import P1_FEATURES
from proyecto_mibici.monitoreo.periodos import (
    COMPARACIONES,
    cargar_periodo,
    definir_periodos,
    metadatos_tabla,
    resumen_periodo,
)
from proyecto_mibici.tracking import RegistroExperimentos, SourceSnapshot

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

COLUMNAS_DRIFT = [*P1_FEATURES, "salidas"]
COLUMNAS_CATEGORICAS = ["fin_semana"]
MAX_FILAS = 100_000
CALIDAD = {
    "RowCount": "filas",
    "DatasetMissingValueCount": "valores_faltantes",
    "DuplicatedRowCount": "filas_duplicadas",
    "ConstantColumnsCount": "columnas_constantes",
    "AlmostConstantColumnsCount": "columnas_casi_constantes",
    "EmptyColumnsCount": "columnas_vacias",
}


def generar_reporte(referencia: pd.DataFrame, actual: pd.DataFrame):
    """Corre los presets de drift y de resumen de datos sobre las variables de P1 y el objetivo."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as error:
        raise ImportError(
            "Evidently es una dependencia opcional: uv sync --frozen --extra monitoreo"
        ) from error
    definicion = DataDefinition(
        numerical_columns=[c for c in COLUMNAS_DRIFT if c not in COLUMNAS_CATEGORICAS],
        categorical_columns=COLUMNAS_CATEGORICAS,
    )
    # Evidently no acepta flotantes como etiquetas de una variable categórica.
    enteros = dict.fromkeys(COLUMNAS_CATEGORICAS, "int64")
    datos = {
        nombre: Dataset.from_pandas(
            frame[COLUMNAS_DRIFT].astype(enteros), data_definition=definicion
        )
        for nombre, frame in (("referencia", referencia), ("actual", actual))
    }
    reporte = Report([DataDriftPreset(columns=COLUMNAS_DRIFT), DataSummaryPreset()])
    return reporte.run(current_data=datos["actual"], reference_data=datos["referencia"])


def _tipo(metrica: dict) -> str:
    return metrica["config"]["type"].rsplit(":", 1)[-1]


def resumir_drift(resultado: dict) -> tuple[pd.DataFrame, dict]:
    """Tabla de drift por variable y resumen global a partir del JSON de Evidently."""
    filas, resumen = [], {}
    for metrica in resultado["metrics"]:
        tipo, config = _tipo(metrica), metrica["config"]
        if tipo == "ValueDrift":
            valor, umbral = float(metrica["value"]), float(config["threshold"])
            # Las pruebas estadísticas reportan p-valor (drift si es bajo); las distancias, al revés.
            es_p_valor = "p_value" in config["method"]
            filas.append(
                {
                    "variable": config["column"],
                    "metodo": config["method"],
                    "valor": valor,
                    "umbral": umbral,
                    "drift": valor <= umbral if es_p_valor else valor >= umbral,
                }
            )
        elif tipo == "DriftedColumnsCount":
            resumen.update(
                columnas_con_drift=int(metrica["value"]["count"]),
                proporcion_con_drift=float(metrica["value"]["share"]),
                umbral_drift_conjunto=float(config["drift_share"]),
            )
        elif tipo in CALIDAD:
            valor = metrica["value"]
            resumen[f"actual_{CALIDAD[tipo]}"] = float(
                valor["count"] if isinstance(valor, dict) else valor
            )
    if not filas or "columnas_con_drift" not in resumen:
        raise ValueError("El resultado de Evidently no contiene métricas de drift.")
    tabla = pd.DataFrame(filas).sort_values("valor", ascending=False, ignore_index=True)
    if int(tabla.drift.sum()) != resumen["columnas_con_drift"]:
        raise ValueError("El drift por variable no coincide con el conteo de Evidently.")
    resumen["drift_del_conjunto"] = (
        resumen["proporcion_con_drift"] >= resumen["umbral_drift_conjunto"]
    )
    resumen["variables_con_drift"] = tabla.loc[tabla.drift, "variable"].tolist()
    return tabla, resumen


def ejecutar(
    comparacion: str,
    config: ProjectConfig | None = None,
    *,
    max_filas: int = MAX_FILAS,
    semilla: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Carga ambos periodos, genera el reporte y guarda HTML, JSON y la tabla por variable."""
    config = config or ProjectConfig()
    if comparacion not in COMPARACIONES:
        raise ValueError(f"comparacion debe ser una de {sorted(COMPARACIONES)}.")
    tabla_meta = metadatos_tabla(config)
    periodos = definir_periodos(config)
    nombres = dict(zip(("referencia", "actual"), COMPARACIONES[comparacion]))
    datos = {
        papel: cargar_periodo(periodos[nombre], config, max_filas=max_filas, semilla=semilla)
        for papel, nombre in nombres.items()
    }
    snapshot = generar_reporte(datos["referencia"], datos["actual"])
    resultado = snapshot.dict()
    tabla, resumen = resumir_drift(resultado)

    salida = config.path(MONITORING_REPORTS_DIR) / comparacion
    salida.mkdir(parents=True, exist_ok=True)
    archivos = {
        "html": salida / f"drift_{comparacion}.html",
        "evidently": salida / f"drift_{comparacion}.json",
        "variables": salida / f"drift_por_variable_{comparacion}.csv",
        "resumen": salida / f"resumen_{comparacion}.json",
    }
    snapshot.save_html(str(archivos["html"]))
    archivos["evidently"].write_text(
        json.dumps(resultado, indent=1, default=str) + "\n", encoding="utf-8"
    )
    tabla.to_csv(archivos["variables"], index=False, float_format="%.6f")
    informe = {
        "comparacion": comparacion,
        "pregunta": "P1",
        "data_version": tabla_meta["data_version"],
        "tabla_supervisada_sha256": tabla_meta["sha256"],
        "max_filas_por_periodo": max_filas,
        "semilla": semilla,
        "version_evidently": version("evidently"),
        "periodos": {
            papel: {"nombre": nombres[papel], **resumen_periodo(periodos[nombres[papel]], frame)}
            for papel, frame in datos.items()
        },
        **resumen,
    }
    archivos["resumen"].write_text(
        json.dumps(informe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"informe": informe, "tabla": tabla, "archivos": archivos}


def registrar_en_mlflow(resultado: dict, config: ProjectConfig | None = None) -> str:
    """Registra el reporte como run de monitoreo; no es candidato y el Registry lo ignora."""
    config = config or ProjectConfig()
    informe, tabla = resultado["informe"], resultado["tabla"]
    experiment_id = RegistroExperimentos(config).configure()
    with tempfile.TemporaryDirectory() as directorio:
        fuente, codigo = SourceSnapshot().create(Path(directorio))
        nombre = f"P1-drift-{informe['comparacion']}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=nombre) as run:
            mlflow.set_tags(
                {
                    **fuente,
                    "team": "25",
                    "question": "P1",
                    "run_kind": "drift_report",
                    "stage_of_work": "monitoring",
                    "data_version": informe["data_version"],
                    "comparacion": informe["comparacion"],
                }
            )
            parametros = {
                "comparacion": informe["comparacion"],
                "max_filas_por_periodo": informe["max_filas_por_periodo"],
                "semilla": informe["semilla"],
                "version_evidently": informe["version_evidently"],
                "tabla_supervisada_sha256": informe["tabla_supervisada_sha256"],
            }
            for papel, periodo in informe["periodos"].items():
                parametros.update(
                    {
                        f"{papel}_periodo": periodo["nombre"],
                        f"{papel}_inicio": periodo["inicio"],
                        f"{papel}_fin_exclusivo": periodo["fin_exclusivo"],
                        f"{papel}_filas": periodo["filas"],
                    }
                )
            mlflow.log_params(parametros)
            mlflow.log_metrics(
                {
                    "drift_columnas": informe["columnas_con_drift"],
                    "drift_proporcion": informe["proporcion_con_drift"],
                    "drift_conjunto": float(informe["drift_del_conjunto"]),
                    **{f"drift_{fila.variable}": fila.valor for fila in tabla.itertuples()},
                }
            )
            mlflow.log_artifact(str(codigo), artifact_path="reproducibilidad")
            for ruta in resultado["archivos"].values():
                mlflow.log_artifact(str(ruta), artifact_path="drift")
            return run.info.run_id


@app.command()
def main(
    comparacion: str = typer.Option("principal", help="principal o estacional"),
    max_filas: int = typer.Option(MAX_FILAS, help="Filas muestreadas por periodo; 0 = todas"),
    semilla: int = DEFAULT_RANDOM_STATE,
    registrar: bool = typer.Option(True, "--mlflow/--sin-mlflow", help="Registrar en MLflow"),
):
    """Genera el reporte de drift de P1 entre la referencia y el periodo actual."""
    resultado = ejecutar(comparacion, max_filas=max_filas, semilla=semilla)
    informe = resultado["informe"]
    logger.info(
        "Drift en {} de {} variables ({:.1%}); drift del conjunto: {}",
        informe["columnas_con_drift"],
        len(resultado["tabla"]),
        informe["proporcion_con_drift"],
        informe["drift_del_conjunto"],
    )
    logger.info("Variables con drift: {}", ", ".join(informe["variables_con_drift"]) or "ninguna")
    logger.success("Reporte en {}", resultado["archivos"]["html"].parent)
    if registrar:
        logger.success("Run de MLflow: {}", registrar_en_mlflow(resultado))


if __name__ == "__main__":
    app()
