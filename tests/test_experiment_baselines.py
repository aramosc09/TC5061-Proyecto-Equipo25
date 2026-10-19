import pyarrow as pa
import pyarrow.parquet as pq

from proyecto_mibici.modeling.experiment_baselines import prepare_supervised_input


def test_prepare_supervised_input_filters_months_and_excludes_personal_columns(tmp_path):
    source = tmp_path / "raw.parquet"
    output = tmp_path / "supervised.parquet"
    pq.write_table(
        pa.table(
            {
                "Viaje_Id": ["a", "b", "c"],
                "Usuario_Id": ["private-1", "private-2", "private-3"],
                "Inicio_del_viaje": ["2023-09-30", "2023-10-01", "2023-11-01"],
                "Fin_del_viaje": ["2023-09-30", "2023-10-02", "2023-11-02"],
                "Origen_Id": ["1", "2", "3"],
                "Destino_Id": ["2", "3", "4"],
                "mes_archivo": ["2023-09", "2023-10", "2023-11"],
            }
        ),
        source,
    )

    count = prepare_supervised_input(
        source,
        output,
        start_month="2023-10",
        end_month="2023-10",
        batch_size=1,
    )

    result = pq.read_table(output)
    assert count == 1
    assert result.column_names == [
        "viaje_id",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
        "mes_archivo",
    ]
    assert result.column("viaje_id").to_pylist() == ["b"]
