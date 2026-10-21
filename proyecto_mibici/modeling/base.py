"""Contratos de resultados y pipelines independientes de MLflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proyecto_mibici.config import ProjectConfig


@dataclass
class BaselineResult:
    question: str
    model_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


class PipelinePregunta(ABC):
    def __init__(self, config: ProjectConfig):
        self.config = config

    @abstractmethod
    def run(self, repository, output_dir: Path) -> BaselineResult:
        """Ajusta solo con entrenamiento y evalúa los periodos posteriores.

        repository.station_hours() produce (id_entero, DataFrame) por estación.
        El índice DatetimeIndex recorre [config.start, config.end), por hora;
        columnas: salidas, llegadas (conteos), elegible (bool, solo pasado).
        repository.stations() devuelve id/name/latitude/longitude/dpcapacity.
        Los artefactos deben contener agregados, nunca viajes individuales.
        """
