"""Esquemas Pydantic de la API de demanda de MiBici (modelo P1).

La API recibe la fecha y hora a predecir y calcula aquí las 6 variables de
calendario; el cliente solo manda las 24 variables de historial de la estación.
"""

from datetime import datetime
import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Cotas de validación. En entrenamiento: salidas_lag_1 máx 207, llegadas_lag_1
# máx 211, balance de -229 a 179. Se dejan con holgura para no rechazar casos
# reales; ajústalas si el equipo prefiere otras.
MAX_REGISTROS = 1_000
MAX_CONTEO = 1_000.0
MAX_BALANCE = 1_000.0
# Tolerancia de redondeo al comparar la media de 4 semanas con sus 4 rezagos.
TOLERANCIA_MEDIA = 0.05

Conteo = Annotated[float, Field(ge=0, le=MAX_CONTEO, allow_inf_nan=False)]
Balance = Annotated[float, Field(ge=-MAX_BALANCE, le=MAX_BALANCE, allow_inf_nan=False)]

CAMPOS_CALENDARIO = (
    "hora_sin",
    "hora_cos",
    "dia_semana_sin",
    "dia_semana_cos",
    "fin_semana",
    "mes",
)


def calcular_calendario(fecha_hora: datetime) -> dict[str, float]:
    """Variables de calendario, con la misma fórmula que el entrenamiento.

    Fórmulas: sin/cos(2π·hora/24), sin/cos(2π·día/7) con lunes = 0,
    fin_semana = 1 si sábado o domingo, mes de 1 a 12.
    """
    hora = fecha_hora.hour
    dia = fecha_hora.weekday()  # lunes = 0
    return {
        "hora_sin": math.sin(2 * math.pi * hora / 24),
        "hora_cos": math.cos(2 * math.pi * hora / 24),
        "dia_semana_sin": math.sin(2 * math.pi * dia / 7),
        "dia_semana_cos": math.cos(2 * math.pi * dia / 7),
        "fin_semana": 1.0 if dia >= 5 else 0.0,
        "mes": float(fecha_hora.month),
    }


class RegistroEntrada(BaseModel):
    """Una estación y una hora a predecir, con su historial reciente."""

    model_config = ConfigDict(extra="forbid")

    fecha_hora: datetime = Field(
        description="Hora a predecir, en hora exacta (minutos y segundos en 0).",
        examples=["2026-03-10T08:00:00"],
    )

    # Historial de salidas de la estación
    salidas_lag_1: Conteo
    salidas_lag_3: Conteo
    salidas_lag_24: Conteo
    salidas_lag_168: Conteo
    salidas_lag_336: Conteo
    salidas_lag_504: Conteo
    salidas_lag_672: Conteo
    salidas_media_3: Conteo
    salidas_media_24: Conteo
    salidas_media_168: Conteo
    salidas_media_4_semanas: Conteo

    # Historial de llegadas a la estación
    llegadas_lag_1: Conteo
    llegadas_lag_3: Conteo
    llegadas_lag_24: Conteo
    llegadas_lag_168: Conteo
    llegadas_lag_336: Conteo
    llegadas_lag_504: Conteo
    llegadas_lag_672: Conteo
    llegadas_media_3: Conteo
    llegadas_media_24: Conteo
    llegadas_media_168: Conteo
    llegadas_media_4_semanas: Conteo

    # Balance (salidas - llegadas): puede ser negativo
    balance_acumulado_3: Balance
    balance_acumulado_24: Balance

    @field_validator("fecha_hora")
    @classmethod
    def _hora_exacta(cls, valor: datetime) -> datetime:
        if valor.minute or valor.second or valor.microsecond:
            raise ValueError(
                "fecha_hora debe ser una hora exacta (minutos y segundos en 0)"
            )
        return valor

    @model_validator(mode="after")
    def _medias_coherentes(self) -> "RegistroEntrada":
        """La media de 4 semanas debe ser el promedio de sus 4 rezagos."""
        for variable in ("salidas", "llegadas"):
            rezagos = [
                getattr(self, f"{variable}_lag_{n}") for n in (168, 336, 504, 672)
            ]
            esperado = sum(rezagos) / len(rezagos)
            media = getattr(self, f"{variable}_media_4_semanas")
            if not math.isclose(media, esperado, abs_tol=TOLERANCIA_MEDIA):
                raise ValueError(
                    f"{variable}_media_4_semanas ({media}) no coincide con el "
                    f"promedio de {variable}_lag_168, _336, _504 y _672 "
                    f"({esperado:.3f})"
                )
        return self

    def a_fila(self) -> dict[str, float]:
        """Las 30 variables del modelo: 6 de calendario + 24 de historial."""
        historial = self.model_dump(exclude={"fecha_hora"})
        return {**calcular_calendario(self.fecha_hora), **historial}


CAMPOS_HISTORIAL = tuple(
    nombre for nombre in RegistroEntrada.model_fields if nombre != "fecha_hora"
)
COLUMNAS_MODELO = (*CAMPOS_CALENDARIO, *CAMPOS_HISTORIAL)


class PeticionPrediccion(BaseModel):
    """Cuerpo de POST /predict: de 1 a MAX_REGISTROS registros."""

    model_config = ConfigDict(extra="forbid")

    registros: list[RegistroEntrada] = Field(min_length=1, max_length=MAX_REGISTROS)


class Prediccion(BaseModel):
    salidas_esperadas: float = Field(
        ge=0,
        allow_inf_nan=False,
        description=(
            "Demanda esperada de salidas en la hora indicada; "
            "no es un conteo garantizado."
        ),
    )


class RespuestaPrediccion(BaseModel):
    predicciones: list[Prediccion]
    modelo: dict[str, Any] = Field(
        description="Linaje del modelo que produjo las predicciones."
    )