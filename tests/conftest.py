"""Repositorio sintético compartido por las pruebas de candidatos (sin viajes individuales)."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import ProjectConfig


class SyntheticRepository:
    """Seis estaciones con rutinas laborales o de fin de semana y ruido Poisson con semilla."""

    def __init__(self, config, *, change_future=False, missing_capacity_station=None):
        self.config = config
        self.change_future = change_future
        self.missing_capacity_station = missing_capacity_station
        self.timeline = pd.date_range(config.start, config.end, freq="h", inclusive="left")

    def station_hours(self):
        rng = np.random.default_rng(7)
        hour = self.timeline.hour.to_numpy()
        weekend = self.timeline.dayofweek.to_numpy() >= 5
        for station in range(1, 7):
            peak = np.where(weekend, 1 + station % 2 * 4, 4 - station % 2 * 3)
            rate = 0.5 + peak * np.exp(-((hour - 8 - station % 3 * 4) ** 2) / 8)
            hours = pd.DataFrame(
                {
                    "salidas": rng.poisson(rate),
                    "llegadas": rng.poisson(rate[::-1]),
                    "elegible": True,
                },
                index=self.timeline,
            )
            if self.change_future:
                hours.loc[hours.index >= self.config.train_end, "salidas"] += 50
            yield station, hours

    def stations(self):
        capacity = [
            np.nan if station == self.missing_capacity_station else 10.0 for station in range(1, 7)
        ]
        return pd.DataFrame({"id": range(1, 7), "dpcapacity": capacity})


@pytest.fixture
def candidate_config(tmp_path):
    return ProjectConfig(
        root=tmp_path,
        start_month="2024-01",
        end_month="2024-02",
        train_end="2024-02-05",
        validation_end="2024-02-12",
    )


@pytest.fixture
def synthetic_repository():
    return SyntheticRepository


@pytest.fixture
def tabla_monitoreo(tmp_path):
    """Tabla supervisada diaria con un solo drift sembrado: salidas_media_168 en el periodo actual."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from proyecto_mibici.modeling.datasets import P1_FEATURES

    config = ProjectConfig(root=tmp_path)
    horas = pd.date_range(config.start, config.end, freq="D", inclusive="left") + pd.Timedelta(
        hours=8
    )
    rng = np.random.default_rng(11)
    partes = []
    for estacion in (1, 2):
        frame = pd.DataFrame(
            rng.gamma(2.0, 1.5, size=(len(horas), len(P1_FEATURES))).astype("float32"),
            columns=P1_FEATURES,
        )
        frame["fin_semana"] = (horas.dayofweek >= 5).astype("float32")
        frame["mes"] = horas.month.astype("float32")
        frame.loc[horas >= pd.Timestamp(config.validation_end), "salidas_media_168"] += 5
        frame.insert(0, "hora", horas)
        frame.insert(0, "estacion", estacion)
        frame["salidas"] = rng.poisson(2.0, len(horas)).astype("float32")
        partes.append(frame)
    tabla = pd.concat(partes, ignore_index=True)
    config.supervised_table_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(tabla, preserve_index=False), config.supervised_table_path)
    metadatos = {
        "data_version": "sintetica-v1",
        "sha256": "sintetico",
        **{
            c: getattr(config, c)
            for c in ("start_month", "end_month", "train_end", "validation_end")
        },
    }
    config.supervised_table_path.with_suffix(".json").write_text(json.dumps(metadatos))
    return config, tabla
