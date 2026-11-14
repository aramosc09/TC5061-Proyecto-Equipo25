"""Reentrenamiento de la versión alternativa (v2 Challenger) para P1.

Lee los hiperparámetros de la versión v1 champion desde MLflow por defecto,
permite sobrescribirlos mediante CLI si se desea, reentrena con la ventana ampliada
(oct-2023 a jun-2026) e imprime una bitácora detallada del proceso.
"""

import os
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
import typer

from proyecto_mibici.config import ProjectConfig
from proyecto_mibici.registry import MODELS

app = typer.Typer(help="Reentrenamiento de la versión v2 Challenger para P1.")


def _importar_caracteristicas_p1():
    from proyecto_mibici.modeling.datasets import P1_FEATURES
    return P1_FEATURES


class DatosPorPeriodo:
    """Adaptador de datos por periodo para ser consumido por CandidatosP1."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        from proyecto_mibici.monitoreo.periodos import definir_periodos
        self.periodos = definir_periodos(self.config)

    def split(self, question: str, name: str, *, max_rows: int = 0, seed: int = 0):
        """Devuelve un Conjunto para el split solicitado según el periodo correspondiente."""
        if question.lower() != "p1":
            raise ValueError("DatosPorPeriodo solo soporta la pregunta p1.")

        from proyecto_mibici.modeling.datasets import Conjunto
        from proyecto_mibici.monitoreo.periodos import cargar_periodo

        p1_features = _importar_caracteristicas_p1()

        if name == "train":
            df = cargar_periodo(
                self.periodos["entrenamiento_v2"],
                max_filas=max_rows,
                semilla=seed,
                config=self.config,
            )
        elif name in ("validation", "test"):
            df = cargar_periodo(
                self.periodos["evaluacion_ab"],
                config=self.config,
            )
        else:
            raise ValueError(f"Split no reconocido: {name}")

        X = df[list(p1_features)]
        y = df["salidas"]
        baseline = df["salidas_lag_168"]
        estacion = df["estacion"]
        hora = df["hora"]

        return Conjunto(
            X=X,
            y=y,
            baseline=baseline,
            estacion=estacion,
            hora=hora
        )


def obtener_hiperparametros_champion(
    client: MlflowClient, model_name: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Obtiene versión, run_id e hiperparámetros resueltos de la versión champion actual."""
    raw_name = model_name or MODELS["p1"]
    nombre_modelo = raw_name[0] if isinstance(raw_name, tuple) else raw_name

    version = client.get_model_version_by_alias(nombre_modelo, "champion")
    run = client.get_run(version.run_id)
    params_raw = run.data.params

    params = {
        "learning_rate": float(params_raw.get("resolved_learning_rate", 0.05)),
        "max_iter": int(params_raw.get("resolved_max_iter", 500)),
        "max_leaf_nodes": int(params_raw.get("resolved_max_leaf_nodes", 31)),
        "min_samples_leaf": int(params_raw.get("resolved_min_samples_leaf", 200)),
        "l2_regularization": float(params_raw.get("resolved_l2_regularization", 1.0)),
    }
    return str(version.version), version.run_id, params


@app.command()
def reentrenar(
    question: str = typer.Option(
        "p1",
        "--question",
        "-q",
        help="Pregunta a reentrenar. Únicamente 'p1' está soportada en el flujo de monitoreo/A-B.",
    ),
    learning_rate: Optional[float] = typer.Option(
        None, help="Sobrescribir learning_rate del champion."
    ),
    max_iter: Optional[int] = typer.Option(
        None, help="Sobrescribir max_iter del champion."
    ),
    max_leaf_nodes: Optional[int] = typer.Option(
        None, help="Sobrescribir max_leaf_nodes del champion."
    ),
    min_samples_leaf: Optional[int] = typer.Option(
        None, help="Sobrescribir min_samples_leaf del champion."
    ),
    l2_regularization: Optional[float] = typer.Option(
        None, help="Sobrescribir l2_regularization del champion."
    ),
):
    """Ejecuta el reentrenamiento v2 imprimiendo la bitácora y registrando el candidato."""
    q = question.lower().strip()

    # 1. Validación de pregunta con mensaje técnico personalizado (sin tracebacks de Python)
    if q != "p1":
        mensaje_error = (
            f"\n⛔ REENTRENAMIENTO NO DISPONIBLE PARA '{question.upper()}'\n"
            f"{'=' * 65}\n"
            f"El script de reentrenamiento únicamente admite la pregunta 'p1'.\n\n"
            f"📋 EXPLICACIÓN TÉCNICA Y RAZONES DE ARQUITECTURA:\n\n"
            f" 1. API REST en Contenedor (Docker/Podman):\n"
            f"    La infraestructura de despliegue en 'models/api/p1' empaqueta y sirve\n"
            f"    exclusivamente el modelo P1 (predicción de demanda de viajes) para\n"
            f"    inferencias en tiempo real mediante el puerto 8000.\n\n"
            f" 2. Monitoreo de Data Drift y Prueba A/B (Entrega 5):\n"
            f"    Los conjuntos de datos ('entrenamiento_v2' y 'evaluacion_ab'), el reporte\n"
            f"    de Evidently y las comparaciones Champion/Challenger en MLflow están\n"
            f"    diseñados específicamente para evaluar la degradación de P1.\n\n"
            f" 3. Alcance de P2 y P3:\n"
            f"    Las preguntas P2 (presión de flujo) y P3 (perfiles de estaciones) son\n"
            f"    análisis secundarios y no cuentan con tubería de reentrenamiento\n"
            f"    continuo ni evaluación A/B pareada en esta etapa del proyecto.\n"
            f"{'=' * 65}\n"
        )
        typer.echo(mensaje_error, err=True)
        raise typer.Exit(code=1)

    # Carga diferida de dependencias de modelado (solo para P1)
    from proyecto_mibici.features import RepositorioParquet
    from proyecto_mibici.modeling.p1_candidates import CandidatosP1
    from proyecto_mibici.monitoreo.periodos import definir_periodos
    from proyecto_mibici.tracking import RegistroExperimentos

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    config = ProjectConfig()
    periodos = definir_periodos(config)
    p_train = periodos["entrenamiento_v2"]
    p_eval = periodos["evaluacion_ab"]

    model_name = MODELS["p1"][0] if isinstance(MODELS["p1"], tuple) else MODELS["p1"]
    champion_ver, champion_run_id, params_champion = obtener_hiperparametros_champion(
        client, model_name
    )

    # Resolver hiperparámetros finales (champion por defecto, o CLI si se especificaron)
    params = {
        "learning_rate": learning_rate if learning_rate is not None else params_champion["learning_rate"],
        "max_iter": max_iter if max_iter is not None else params_champion["max_iter"],
        "max_leaf_nodes": max_leaf_nodes if max_leaf_nodes is not None else params_champion["max_leaf_nodes"],
        "min_samples_leaf": min_samples_leaf if min_samples_leaf is not None else params_champion["min_samples_leaf"],
        "l2_regularization": l2_regularization if l2_regularization is not None else params_champion["l2_regularization"],
    }

    # Bitácora detallada en consola
    print("=" * 70)
    print("🚀 INICIANDO REENTRENAMIENTO V2 (CHALLENGER) - PREGUNTA P1")
    print("=" * 70)
    print(f"📌 Modelo base: {model_name} (v{champion_ver} @champion)")
    print(f"📌 Run ID origen: {champion_run_id}")
    print("-" * 70)
    print("📅 RANGOS TEMPORALES DEFINIDOS:")
    print(f"   • Entrenamiento (v2) : {p_train.inicio} ➔ {p_train.fin}")
    print(f"   • Evaluación A/B     : {p_eval.inicio} ➔ {p_eval.fin}")
    print("-" * 70)
    print("⚙️  HIPERPARÁMETROS UTILIZADOS:")
    for k, v in params.items():
        origen = "(sobrescrito por CLI)" if locals()[k] is not None else "(heredado de champion)"
        print(f"   • {k:20s}: {v} {origen}")
    print("=" * 70)

    manifest = RepositorioParquet(config).verify_base()
    datos = DatosPorPeriodo(config)

    def entrenar(salida: str):
        return CandidatosP1(config).run(
            datos,
            "hgb",
            params,
            salida,
            max_train_rows=2_000_000,
            evaluate_test=False,
        )

    extra_parameters = {
        "entrenamiento_inicio": str(p_train.inicio),
        "entrenamiento_fin": str(p_train.fin),
        "evaluacion_inicio": str(p_eval.inicio),
        "evaluacion_fin": str(p_eval.fin),
        # Mismos nombres que el champion, con las ventanas realmente usadas.
        # Sustituyen a los valores de ProjectConfig, que no cambian.
        "train_end_exclusive": str(p_train.fin),
        "validation_end_exclusive": str(p_eval.fin),
    }
    extra_tags = {
        "version_role": "challenger",
        "source_champion_run_id": champion_run_id,
    }
    resultado_candidato = RegistroExperimentos(config).candidate(
        manifest=manifest,
        question="p1",
        model="hgb",
        family="gradient_boosting",
        train=entrenar,
        extra_parameters=extra_parameters,
        extra_tags=extra_tags,
        run_name=f"p1-hgb-challenger-{p_train.inicio}-{p_train.fin}",
    )

    # Extraer el run_id tolerando si devuelve un diccionario o un objeto Run
    if isinstance(resultado_candidato, dict):
        run_id_v2 = resultado_candidato.get("run_id") or resultado_candidato["run"].info.run_id
    else:
        run_id_v2 = resultado_candidato.info.run_id

    print("\n" + "=" * 70)
    print("✅ REENTRENAMIENTO V2 COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 RUN_ID Generado: {run_id_v2}")
    print("\n👉 Siguiente paso: Promover la versión con alias 'challenger':")
    print(
        f"   uv run --frozen python -m proyecto_mibici.registry promote "
        f"--question p1 --run-id {run_id_v2} --alias challenger "
        f'--reason "v2: reentrenado con datos hasta jun-2026" --approve'
    )
    print("=" * 70 + "\n")

    return run_id_v2


if __name__ == "__main__":
    app()