"""Escenario reproducible de carga HTTP para el endpoint /predict."""

import os
import random

from locust import HttpUser, between, task

from proyecto_mibici.carga import batch_size_desde_entorno, crear_peticion


class MiBiciPredictUser(HttpUser):
    """Usuario que genera solicitudes variadas al endpoint de demanda P1."""

    host = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN", "0")),
        float(os.getenv("LOCUST_WAIT_MAX", "0.5")),
    )

    def on_start(self):
        self.rng = random.Random(int(os.getenv("LOCUST_SEED", "42")) + self.environment.runner.user_count)
        self.batch_size = batch_size_desde_entorno()

    @task
    def predict(self):
        cuerpo = crear_peticion(self.batch_size, self.rng)
        nombre = f"/predict [lote={self.batch_size}]"
        with self.client.post("/predict", json=cuerpo, name=nombre, catch_response=True) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}: {respuesta.text[:200]}")
                return
            datos = respuesta.json()
            predicciones = datos.get("predicciones")
            if not isinstance(predicciones, list):
                respuesta.failure("La respuesta no contiene una lista de predicciones.")
                return
            if len(predicciones) != self.batch_size:
                respuesta.failure(
                    f"Se esperaban {self.batch_size} predicciones y llegaron {len(predicciones)}."
                )
                return
            respuesta.success()
