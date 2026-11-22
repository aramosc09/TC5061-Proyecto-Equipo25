"""Pruebas unitarias para el generador de modelo sintético de humo."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.api.esquemas import COLUMNAS_MODELO
from proyecto_mibici.api.modelo import cargar_modelo
from proyecto_mibici.api.modelo_humo import generar_modelo_humo


def test_generar_modelo_humo_carga_y_predice(tmp_path: Path) -> None:
    """El modelo sintético debe cargar en modo local y predecir valores válidos."""
    destino = tmp_path / "p1"
    generar_modelo_humo(destino=destino)

    assert (destino / "model").exists()
    meta = json.loads((destino / "metadata_registro.json").read_text(encoding="utf-8"))
    assert meta["model_family"] == "humo"
    assert meta["version"] == "0"

    artefacto = cargar_modelo("local", model_dir=destino)
    modelo = artefacto.modelo if hasattr(artefacto, "modelo") else artefacto

    muestra = pd.DataFrame(
        np.ones((3, len(COLUMNAS_MODELO)), dtype=np.float32),
        columns=list(COLUMNAS_MODELO),
    )
    preds = np.asarray(modelo.predict(muestra))
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)


def test_no_sobrescribe_modelo_real_sin_forzar(tmp_path: Path) -> None:
    """Debe lanzar FileExistsError si existe un modelo real y no se usa --forzar."""
    destino = tmp_path / "p1"
    destino.mkdir(parents=True)
    (destino / "metadata_registro.json").write_text(
        json.dumps({"registered_model": "mibici-equipo25-p1-demanda", "version": "1"}),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        generar_modelo_humo(destino=destino, forzar=False)

    # Con forzar=True sí debe sobrescribir
    generar_modelo_humo(destino=destino, forzar=True)
    meta_nueva = json.loads((destino / "metadata_registro.json").read_text(encoding="utf-8"))
    assert meta_nueva["model_family"] == "humo"