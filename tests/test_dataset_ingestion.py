"""Contracts for automatic ingestion, without contacting MiBici or reading personal data."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from proyecto_mibici import config, dataset

COLUMNS = [
    "Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento", "Inicio_del_viaje",
    "Fin_del_viaje", "Origen_Id", "Destino_Id",
]


def trip(identifier, start="2023-10-01 10:00:00", end="2023-10-01 10:15:00"):
    return [identifier, "synthetic-private", "X", "1990", start, end, "1", "2"]


def write_sources(root, rows=None):
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    pd.DataFrame(rows if rows is not None else [trip(1)], columns=COLUMNS).to_csv(
        raw / "datos_abiertos_2023_10.csv", index=False, encoding="cp1252"
    )
    pd.DataFrame({
        "id": [1, 2], "name": ["Estación sintética", "Otra"],
        "latitude": [20.6, 20.7], "longitude": [-103.3, -103.4], "dpcapacity": [10, 0],
    }).to_csv(raw / "nomenclatura_2026_09.csv", index=False)
    return raw


def ingest(root, **kwargs):
    return dataset.ejecutar_ingesta(
        anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=10,
        input_path=root / "data/raw",
        parquet_path=root / "data/interim/viajes.parquet",
        estaciones_path=root / "data/interim/estaciones.parquet",
        manifiesto_path=root / "data/data_manifest.json",
        sin_descarga=True, chunk_size=2, **kwargs,
    )


def test_cleaning_across_chunks_privacy_and_repeatability(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJ_ROOT", tmp_path)
    rows = [
        trip(1), trip(2, "02/10/2023 10:00", "02/10/2023 10:15"), trip(1),
        trip(3, "invalid"), trip(4, end="2023-10-01 09:00:00"),
        trip(5, "2023-11-01 10:00:00", "2023-11-01 10:15:00"),
        trip(6, "2023-10-31 23:59:00", "2023-11-01 00:15:00"),
    ]
    raw = write_sources(tmp_path, rows)
    before = {path: dataset.calcular_sha256(path) for path in raw.iterdir()}
    manifest = ingest(tmp_path)
    result = pd.read_parquet(tmp_path / manifest["salida"]["ruta"])
    assert result.columns.tolist() == dataset.TRIP_SCHEMA.names
    assert result.viaje_id.tolist() == [1, 2, 6]
    assert result.inicio_viaje.iloc[1] == pd.Timestamp("2023-10-02 10:00")
    assert result.fin_viaje.iloc[-1] == pd.Timestamp("2023-11-01 00:15")
    assert not set(dataset.COLUMNAS_PERSONALES) & set(result.columns)
    record = manifest["viajes_mensuales"][0]
    assert record["n_filas"] == 7
    assert record["n_descartadas_calidad"] == 3
    assert record["n_duplicados"] == 1
    assert record["n_filas_salida"] == 3
    assert record["codificacion"] == "cp1252"
    assert manifest["resumen"]["n_viajes_limpios"] == 3
    assert manifest["nomenclatura"][0]["n_sin_capacidad"] == 1
    assert dataset.DataManifest(tmp_path / "data/data_manifest.json").verificar() == []
    assert ingest(tmp_path)["data_version"] == manifest["data_version"]
    assert {path: dataset.calcular_sha256(path) for path in raw.iterdir()} == before
    assert "synthetic-private" not in json.dumps(manifest)


def test_maps_headers_by_name_not_position(tmp_path):
    raw = write_sources(tmp_path)
    path = raw / "datos_abiertos_2023_10.csv"
    frame = pd.read_csv(path, encoding="cp1252")
    frame.loc[:, list(reversed(COLUMNS))].to_csv(path, index=False, encoding="utf-8-sig")
    manifest = ingest(tmp_path)
    assert manifest["resumen"]["n_viajes_limpios"] == 1
    assert pd.read_parquet(tmp_path / "data/interim/viajes.parquet").origen_id.tolist() == [1]


def test_observed_damaged_birth_year_header_is_recognized(tmp_path):
    raw = write_sources(tmp_path)
    path = raw / "datos_abiertos_2023_10.csv"
    frame = pd.read_csv(path, encoding="cp1252").rename(
        columns={"Año_de_nacimiento": "A}äe_nacimiento"}
    )
    frame.to_csv(path, encoding="cp1252", index=False)
    assert ingest(tmp_path)["resumen"]["n_viajes_limpios"] == 1


def test_missing_month_does_not_publish_partial_outputs(tmp_path):
    write_sources(tmp_path)
    with pytest.raises(FileNotFoundError, match="2023_11"):
        dataset.ejecutar_ingesta(
            anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=11,
            input_path=tmp_path / "data/raw", sin_descarga=True,
            parquet_path=tmp_path / "viajes.parquet",
            estaciones_path=tmp_path / "estaciones.parquet",
            manifiesto_path=tmp_path / "manifest.json",
        )
    assert not (tmp_path / "viajes.parquet").exists()


def test_failure_preserves_previous_outputs(tmp_path):
    raw = write_sources(tmp_path)
    ingest(tmp_path)
    output = tmp_path / "data/interim/viajes.parquet"
    before = dataset.calcular_sha256(output)
    path = raw / "datos_abiertos_2023_10.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        ingest(tmp_path)
    assert dataset.calcular_sha256(output) == before
    assert not list(output.parent.glob(".ingesta-*"))


def test_duplicate_station_id_rejected(tmp_path):
    raw = write_sources(tmp_path)
    path = raw / "nomenclatura_2026_09.csv"
    frame = pd.read_csv(path)
    frame["id"] = [1, 1]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="IDs inválidos o duplicados"):
        ingest(tmp_path)


def test_version_changes_when_source_or_rules_change(tmp_path, monkeypatch):
    raw = write_sources(tmp_path)
    first = ingest(tmp_path)["data_version"]
    monkeypatch.setattr(dataset, "NORMALIZATION_VERSION", "synthetic-new-policy")
    second = ingest(tmp_path)["data_version"]
    assert second != first
    path = raw / "datos_abiertos_2023_10.csv"
    pd.DataFrame([trip(2)], columns=COLUMNS).to_csv(path, index=False)
    assert ingest(tmp_path)["data_version"] != second


def test_manifest_detects_modified_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJ_ROOT", tmp_path)
    write_sources(tmp_path)
    ingest(tmp_path)
    path = tmp_path / "data/interim/viajes.parquet"
    frame = pd.read_parquet(path)
    frame["origen_id"] = 99
    frame.to_parquet(path, index=False)
    errors = dataset.DataManifest(tmp_path / "data/data_manifest.json").verificar()
    assert any("hash del Parquet" in error for error in errors)


class Response(io.BytesIO):
    def __init__(self, content, url, expected=None):
        super().__init__(content)
        self.url = url
        self.headers = {"Content-Length": str(expected if expected is not None else len(content))}

    def geturl(self):
        return self.url


def test_download_discovers_sources_and_reuses_cache(tmp_path, monkeypatch):
    source_root = tmp_path / "fixtures"
    raw = write_sources(source_root)
    links = {path.name: f"https://www.mibici.net/site/assets/files/1/{path.name}"
             for path in raw.iterdir()}
    page = "".join(f'<a href="{url}">archivo</a>' for url in links.values()).encode()
    calls = []

    def open_url(request, timeout):
        url = request.full_url
        assert request.get_header("User-agent") == "MiBici-Equipo25/1.0"
        calls.append(url)
        if url == config.DATA_URL_ONLINE:
            return Response(page, url)
        return Response((raw / Path(url).name).read_bytes(), url)

    monkeypatch.setattr(dataset, "urlopen", open_url)
    output_root = tmp_path / "download"
    for _ in range(2):
        manifest = dataset.ejecutar_ingesta(
            anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=10,
            input_path=output_root / "raw", parquet_path=output_root / "trips.parquet",
            estaciones_path=output_root / "stations.parquet",
            manifiesto_path=output_root / "manifest.json", chunk_size=1,
        )
    assert manifest["resumen"]["n_viajes_limpios"] == 1
    assert calls.count(config.DATA_URL_ONLINE) == 2
    assert all(calls.count(url) == 1 for url in links.values())
    assert manifest["viajes_mensuales"][0]["url"] == links["datos_abiertos_2023_10.csv"]


def test_incomplete_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "raw.csv"
    output.write_bytes(b"previous")
    url = "https://www.mibici.net/site/assets/files/1/raw.csv"
    monkeypatch.setattr(dataset, "urlopen", lambda *args, **kwargs: Response(b"x", url, 5))
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="incompleta"):
        dataset.descargar_archivo(url, output)
    assert output.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [output]


def test_cli_runs_offline_from_local_sources(tmp_path):
    raw = write_sources(tmp_path)
    result = CliRunner().invoke(dataset.app, [
        "procesar-datos", "--anio-inicio", "2023", "--mes-inicio", "10",
        "--anio-fin", "2023", "--mes-fin", "10", "--input-path", str(raw),
        "--output-path", str(tmp_path / "output"), "--manifiesto-path", str(tmp_path / "manifest.json"),
        "--sin-descarga", "--chunk-size", "1",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "output/viajes_unificados.parquet").is_file()
    assert (tmp_path / "output/estaciones.parquet").is_file()
    assert dataset.obtener_version_datos(tmp_path / "manifest.json")


def test_duplicate_across_months_is_counted(tmp_path):
    write_sources(tmp_path)
    pd.DataFrame([trip(1, "2023-11-01 10:00:00", "2023-11-01 10:15:00")], columns=COLUMNS).to_csv(
        tmp_path / "data/raw/datos_abiertos_2023_11.csv", index=False,
    )
    manifest = dataset.ejecutar_ingesta(
        anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=11,
        input_path=tmp_path / "data/raw", parquet_path=tmp_path / "trips.parquet",
        estaciones_path=tmp_path / "stations.parquet", manifiesto_path=tmp_path / "manifest.json",
        sin_descarga=True, chunk_size=1,
    )
    assert manifest["viajes_mensuales"][1]["n_duplicados"] == 1
    assert manifest["resumen"]["n_viajes_limpios"] == 1
    assert dataset.DataManifest(tmp_path / "manifest.json").verificar_parquet() == []


def test_multiple_files_for_one_month_are_rejected(tmp_path):
    raw = write_sources(tmp_path)
    source = raw / "datos_abiertos_2023_10.csv"
    (raw / "datos_abiertos_2023_10-1.csv").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="varios CSV"):
        ingest(tmp_path)


def test_source_files_cannot_be_used_as_output_paths(tmp_path):
    raw = write_sources(tmp_path)
    source = raw / "datos_abiertos_2023_10.csv"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="archivos de origen"):
        dataset.ejecutar_ingesta(
            anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=10,
            input_path=raw, parquet_path=source,
            estaciones_path=tmp_path / "stations.parquet",
            manifiesto_path=tmp_path / "manifest.json", sin_descarga=True,
        )
    assert source.read_bytes() == before


def test_version_does_not_depend_on_output_path_or_chunk_size(tmp_path):
    raw = write_sources(tmp_path, [trip(1), trip(2), trip(1)])
    first = ingest(tmp_path)
    second = dataset.ejecutar_ingesta(
        anio_inicio=2023, mes_inicio=10, anio_fin=2023, mes_fin=10,
        input_path=raw, parquet_path=tmp_path / "other.parquet",
        estaciones_path=tmp_path / "stations.parquet",
        manifiesto_path=tmp_path / "other_manifest.json", sin_descarga=True, chunk_size=1,
    )
    assert first["data_version"] == second["data_version"]
    assert first["resumen"] == second["resumen"]
