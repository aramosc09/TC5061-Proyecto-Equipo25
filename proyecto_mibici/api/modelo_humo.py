"""Genera un modelo sintético P1 para validación de imagen y Compose en CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from proyecto_mibici.api.esquemas import COLUMNAS_MODELO


def generar_modelo_humo(destino: Path, forzar: bool = False) -> Path:
    """Entrena y guarda un modelo sintético con la firma de 30 variables de P1."""
    meta_path = destino / "metadata_registro.json"

    if destino.exists() and any(destino.iterdir()):
        es_humo = False
        if meta_path.exists():
            meta_actual = json.loads(meta_path.read_text(encoding="utf-8"))
            es_humo = meta_actual.get("model_family") == "humo"

        if not es_humo and not forzar:
            raise FileExistsError(
                f"La carpeta '{destino}' contiene un modelo real. Usa --forzar para sobrescribir."
            )
        shutil.rmtree(destino)

    destino.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_filas = 64
    datos = rng.uniform(0.0, 10.0, size=(n_filas, len(COLUMNAS_MODELO))).astype(np.float32)
    x_train = pd.DataFrame(datos, columns=list(COLUMNAS_MODELO))
    y_train = rng.poisson(lam=4.0, size=n_filas).astype(np.float32)

    modelo = HistGradientBoostingRegressor(loss="poisson", max_iter=5, random_state=42)
    modelo.fit(x_train, y_train)

    ejemplo = x_train.iloc[:5]
    predicciones = modelo.predict(ejemplo)
    firma = infer_signature(ejemplo, predicciones)

    mlflow.sklearn.save_model(
        sk_model=modelo,
        path=str(destino / "model"),
        signature=firma,
        input_example=ejemplo,
        metadata={"question": "P1"},
        skops_trusted_types=[
            "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        ],
    )

    metadata = {
        "registered_model": "mibici-equipo25-p1-demanda",
        "version": "0",
        "alias": "champion",
        "run_id": "ci-synthetic-smoke-run",
        "git_commit": "ci-smoke",
        "data_version": "synthetic",
        "model_family": "humo",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destino


def main() -> None:
    """Punto de entrada CLI para generar el modelo de humo."""
    parser = argparse.ArgumentParser(description="Genera un modelo sintético P1 para CI.")
    parser.add_argument(
        "--destino",
        type=Path,
        default=Path("models/api/p1"),
        help="Directorio de salida del modelo exportado.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Permite sobrescribir un modelo real existente.",
    )
    args = parser.parse_args()
    ruta = generar_modelo_humo(destino=args.destino, forzar=args.forzar)
    print(f"Modelo sintético generado en: {ruta}")


if __name__ == "__main__":
    main()