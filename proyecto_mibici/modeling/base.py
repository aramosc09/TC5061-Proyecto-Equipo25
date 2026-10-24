"""Contratos de resultados y pipelines independientes de MLflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proyecto_mibici.config import CANDIDATE_MODELS, ProjectConfig


@dataclass
class BaselineResult:
    question: str
    model_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class CandidateResult(BaselineResult):
    """Resultado de un candidato; conserva el estimador ajustado para registrarlo después."""

    model_family: str = ""
    estimator: Any = None
    feature_names: list[str] = field(default_factory=list)


def resolve_hyperparameters(question: str, model: str, overrides: dict[str, Any]) -> dict:
    """Combina defaults de config.py con valores explícitos; rechaza claves ajenas."""
    try:
        defaults = CANDIDATE_MODELS[question][model]["defaults"]
    except KeyError as error:
        options = ", ".join(CANDIDATE_MODELS.get(question, {})) or "ninguno"
        raise ValueError(f"Modelo {model!r} no disponible para {question}: {options}.") from error
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"{model} no acepta los hiperparámetros: {', '.join(unknown)}.")
    resolved = dict(defaults)
    for key, value in overrides.items():
        kind = type(defaults[key])
        try:
            resolved[key] = kind(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} debe ser de tipo {kind.__name__}: {value!r}.") from error
    return resolved


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
