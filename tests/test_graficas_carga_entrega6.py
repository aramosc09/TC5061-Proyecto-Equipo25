"""Pruebas con historial mínimo de Locust sin ejecutar contenedores."""

import pandas as pd

from pruebas_carga.graficas import extraer_series, generar, resumir_escalones


def test_extrae_tipo_y_predicciones_por_segundo(tmp_path):
    filas = []
    for segundo in range(40):
        base = {"Timestamp": 100 + segundo, "User Count": 20, "Requests/s": 10,
                "Failures/s": 0, "50%": 8, "95%": 26}
        filas.append({**base, "Name": "Aggregated"})
        filas.append({**base, "Name": "POST una [1]"})
    historial = pd.DataFrame(filas)
    agregado, por_tipo = extraer_series(historial)
    assert por_tipo["predicciones_s"].iloc[0] == 10
    assert resumir_escalones(agregado, por_tipo).iloc[0]["usuarios"] == 20
    entrada = tmp_path / "history.csv"
    historial.to_csv(entrada, index=False)
    salida = tmp_path / "salida"
    resultado = generar(entrada, None, salida)
    assert resultado["niveles_estables"] == [20]
    assert (salida / "01_latencia.png").exists()
    assert (salida / "02_throughput.png").exists()
    assert (salida / "03_errores.png").exists()
