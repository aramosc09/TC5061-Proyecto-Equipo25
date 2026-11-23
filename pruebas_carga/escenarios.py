"""Definición del perfil de carga sin dependencias del proyecto."""

import os


def leer_escalones(texto: str) -> list[tuple[int, int]]:
    """Convierte usuarios:segundos a escalones acumulativos."""
    escalones = []
    for entrada in texto.split(","):
        partes = entrada.strip().split(":")
        if len(partes) != 2:
            raise ValueError("Formato de escalón: usuarios:segundos")
        usuarios, duracion = (int(valor) for valor in partes)
        if usuarios <= 0 or duracion <= 0:
            raise ValueError("Usuarios y duración deben ser positivos")
        escalones.append((usuarios, duracion))
    if not escalones:
        raise ValueError("Se necesita al menos un escalón")
    return escalones


def estado_en_segundo(segundo: float, escalones: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Devuelve usuarios objetivo y rampa según segundos desde el inicio."""
    limite = 0
    for usuarios, duracion in escalones:
        limite += duracion
        if segundo < limite:
            return usuarios, max(1, min(usuarios, int(os.getenv("LOCUST_SPAWN_RATE", "10"))))
    return None


ESCALONES_PREDETERMINADOS = "10:120,25:120,50:120,100:120"
