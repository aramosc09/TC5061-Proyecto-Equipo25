"""Escenario de carga con solicitudes reales generadas previamente."""

import json
import os
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

from escenarios import ESCALONES_PREDETERMINADOS, estado_en_segundo, leer_escalones


DIRECTORIO = Path(__file__).resolve().parent / "peticiones"


def abrir_cuerpos() -> dict[str, dict]:
    """Carga únicamente archivos JSON, sin depender de pandas dentro del contenedor."""
    cuerpos = {}
    for tipo in ("una", "estacion_dia", "sistema_hora"):
        ruta = DIRECTORIO / f"{tipo}.json"
        cuerpos[tipo] = json.loads(ruta.read_text(encoding="utf-8"))
        if not cuerpos[tipo].get("registros"):
            raise ValueError(f"No hay registros para {tipo}")
    return cuerpos


class UsuarioMiBici(HttpUser):
    """Simula consultas puntuales, diarias, de todo el sistema y de salud."""

    host = "http://api:8000"
    wait_time = between(0.5, 2)

    def on_start(self):
        self.cuerpos = abrir_cuerpos()
        self.tipo = os.environ.get("LOCUST_TIPO", "mixto")
        if self.tipo not in {"mixto", "una", "estacion_dia", "sistema_hora"}:
            raise ValueError("LOCUST_TIPO inválido")

    def enviar(self, tipo: str) -> None:
        tipo = tipo if self.tipo == "mixto" else self.tipo
        cuerpo = self.cuerpos[tipo]
        cantidad = len(cuerpo["registros"])
        with self.client.post(
            "/predict", json=cuerpo, name=f"POST {tipo} [{cantidad}]", catch_response=True
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
                return
            try:
                datos = respuesta.json()
            except ValueError:
                respuesta.failure("Respuesta no es JSON")
                return
            if len(datos.get("predicciones", [])) != cantidad:
                respuesta.failure(f"Se esperaban {cantidad} predicciones")
                return
            respuesta.success()

    @task(6)
    def una(self):
        self.enviar("una")

    @task(3)
    def estacion_dia(self):
        self.enviar("estacion_dia")

    @task(1)
    def sistema_hora(self):
        self.enviar("sistema_hora")

    @task(1)
    def salud(self):
        if self.tipo != "mixto":
            self.enviar(self.tipo)
            return
        with self.client.get("/health", name="GET salud", catch_response=True) as respuesta:
            try:
                correcto = respuesta.status_code == 200 and respuesta.json().get("status") == "ok"
            except ValueError:
                correcto = False
            if not correcto:
                respuesta.failure("Servicio sin respuesta saludable")
            else:
                respuesta.success()


class PerfilEscalones(LoadTestShape):
    """Rampas configurables de 10, 25, 50 y 100 usuarios."""

    def tick(self):
        escalones = leer_escalones(os.getenv("LOCUST_ESCALONES", ESCALONES_PREDETERMINADOS))
        return estado_en_segundo(self.get_run_time(), escalones)
