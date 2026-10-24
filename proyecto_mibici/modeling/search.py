"""Rejillas de búsqueda declaradas en config.py y expandidas a hiperparámetros completos."""

from itertools import product

from proyecto_mibici.config import SEARCH_CRITERIA, SEARCH_SPACES
from proyecto_mibici.modeling.base import resolve_hyperparameters


def expand_grid(question: str, model: str) -> list[dict]:
    """Devuelve un diccionario resuelto por combinación, en orden determinista."""
    try:
        space = SEARCH_SPACES[question][model]
    except KeyError as error:
        available = ", ".join(SEARCH_SPACES.get(question, {})) or "ninguno"
        raise ValueError(f"No hay búsqueda para {question}/{model}; opciones: {available}.") from error
    keys = list(space)
    return [
        resolve_hyperparameters(question, model, dict(zip(keys, values, strict=True)))
        for values in product(*(space[key] for key in keys))
    ]


def criterion(question: str) -> tuple[str, str]:
    return SEARCH_CRITERIA[question]
