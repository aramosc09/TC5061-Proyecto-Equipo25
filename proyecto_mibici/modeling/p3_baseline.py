"""P3: perfiles con exposición observada y baseline de mediana de fin de semana."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

from proyecto_mibici.modeling.base import BaselineResult, PipelinePregunta

matplotlib.use("Agg")

DAYS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
PROFILE_COLUMNS = [
    f"{direction}_{day}_h{hour:02d}"
    for direction in ("salidas", "llegadas")
    for day in DAYS
    for hour in range(24)
]
WEEKEND_MASK = np.tile(np.repeat(np.arange(7) >= 5, 24), 2)


def perfil_estacion(hours: pd.DataFrame) -> tuple[pd.Series | None, dict]:
    """Divide eventos entre horas elegibles por franja y normaliza las 336 tasas.

    Una franja sin exposición es desconocida, no cero uso: se excluye el perfil
    si falta alguna de las 168 franjas semanales o no hay actividad. Los conteos
    cero sí son válidos en franjas con exposición analítica documentada.
    """
    eligible = hours.loc[hours.elegible]
    slots = eligible.index.dayofweek * 24 + eligible.index.hour
    exposure = np.bincount(slots, minlength=168)
    audit = {
        "horas_elegibles": len(eligible),
        "franjas_cubiertas": int((exposure > 0).sum()),
        "eventos": int(eligible[["salidas", "llegadas"]].to_numpy().sum()),
    }
    if not (exposure > 0).all():
        return None, {**audit, "estado": "cobertura_semanal_incompleta"}
    if audit["eventos"] == 0:
        return None, {**audit, "estado": "sin_actividad"}
    rates = np.concatenate(
        [
            np.bincount(slots, weights=eligible[direction], minlength=168) / exposure
            for direction in ("salidas", "llegadas")
        ]
    )
    return pd.Series(rates / rates.sum(), index=PROFILE_COLUMNS), {**audit, "estado": "incluida"}


def weekend_shares(profiles: pd.DataFrame) -> pd.Series:
    """Suma la actividad relativa de sábado/domingo en ambas direcciones."""
    return profiles.loc[:, np.asarray(PROFILE_COLUMNS)[WEEKEND_MASK]].sum(axis=1)


def silhouette_euclidea(profiles: pd.DataFrame, labels: pd.Series) -> float:
    """Calcula la misma distancia euclídea sin depender del producto matricial BLAS.

    Se observaron avisos aritméticos del backend de NumPy incluso con perfiles
    sintéticos acotados y finitos. La distancia cuadrada vía SciPy evita esa ruta
    sin silenciar avisos ni cambiar la métrica de comparación.
    """
    values = profiles.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Los perfiles de P3 deben ser finitos.")
    distances = np.sqrt(pairwise_distances(values, metric="sqeuclidean"))
    np.fill_diagonal(distances, 0)
    return float(silhouette_score(distances, labels, metric="precomputed"))


class PipelineP3(PipelinePregunta):
    def run(self, repository, output_dir: Path) -> BaselineResult:
        """Fija la mediana solo con entrenamiento y evalúa sin reajustar grupos."""
        output_dir.mkdir(parents=True, exist_ok=True)
        profiles = {split: {} for split in self.config.splits}
        coverage = []
        for station, hours in repository.station_hours():
            for split, (start, end) in self.config.splits.items():
                selected = hours.loc[(hours.index >= start) & (hours.index < end)]
                profile, audit = perfil_estacion(selected)
                coverage.append({"estacion": station, "conjunto": split, **audit})
                if profile is not None:
                    profiles[split][station] = profile
        frames = {
            split: pd.DataFrame.from_dict(
                values, orient="index", columns=PROFILE_COLUMNS
            ).rename_axis("estacion")
            for split, values in profiles.items()
        }
        for split, frame in frames.items():
            if frame.empty:
                raise ValueError(f"P3 no tiene perfiles semanales completos en {split}.")
        total_stations = len({row["estacion"] for row in coverage})
        shares = {split: weekend_shares(frame) for split, frame in frames.items()}
        threshold = float(shares["train"].median())
        assignments = {
            split: values.gt(threshold).astype("int8") for split, values in shares.items()
        }
        result = BaselineResult(
            "P3",
            "weekend_share_training_median",
            parameters={
                "model": "weekend_share_training_median",
                "horizon_hours": "not_applicable_descriptive",
                "training_median": threshold,
                "threshold_fit_split": "train",
                "group_rule": "1_if_weekend_share_gt_training_median_else_0",
                "n_profile_features": len(PROFILE_COLUMNS),
                "distance": "euclidean",
                "distance_implementation": "sqrt_sqeuclidean_precomputed_without_blas_matmul",
                "profile_normalization": "per_eligible_hour_slot_then_joint_sum_one",
                "minimum_covered_weekly_slots": 168,
                "undefined_metric_policy": "omit_and_record_defined_flag",
                "coordinates_used_for_clustering": False,
            },
        )
        assignment_rows = []
        group_rows = []
        for split, frame in frames.items():
            labels = assignments[split]
            n = len(frame)
            groups = int(labels.nunique())
            defined = 2 <= groups < n
            result.metrics.update(
                {
                    f"{split}_n_stations": n,
                    f"{split}_n_groups": groups,
                    f"{split}_group_0_size": int(labels.eq(0).sum()),
                    f"{split}_group_1_size": int(labels.eq(1).sum()),
                    f"{split}_silhouette_defined": int(defined),
                    f"{split}_excluded_stations": total_stations - n,
                    f"{split}_eligible_hours": sum(
                        row["horas_elegibles"] for row in coverage if row["conjunto"] == split
                    ),
                }
            )
            if defined:
                result.metrics[f"{split}_silhouette"] = silhouette_euclidea(frame, labels)
            for station in frame.index:
                assignment_rows.append(
                    {
                        "estacion": station,
                        "conjunto": split,
                        "proporcion_fin_semana": float(shares[split].loc[station]),
                        "grupo": int(labels.loc[station]),
                    }
                )
            for group in (0, 1):
                members = frame.loc[labels.eq(group)]
                if not members.empty:
                    group_rows.append(
                        {
                            "conjunto": split,
                            "grupo": group,
                            "n_estaciones": len(members),
                            **members.mean().to_dict(),
                        }
                    )
            if split != "train":
                common = labels.index.intersection(assignments["train"].index)
                train_labels = assignments["train"].loc[common]
                other_labels = labels.loc[common]
                ari_defined = (
                    len(common) >= 2
                    and train_labels.nunique() >= 2
                    and other_labels.nunique() >= 2
                )
                result.metrics[f"{split}_common_stations_vs_train"] = len(common)
                result.metrics[f"{split}_ari_vs_train_defined"] = int(ari_defined)
                if ari_defined:
                    result.metrics[f"{split}_ari_vs_train"] = float(
                        adjusted_rand_score(train_labels, other_labels)
                    )
        assignments_table = pd.DataFrame(assignment_rows)
        catalog = repository.stations()[["id", "name", "latitude", "longitude"]]
        assignments_table = assignments_table.merge(
            catalog, left_on="estacion", right_on="id", how="left", validate="many_to_one"
        ).drop(columns="id")
        assignments_path = output_dir / "asignaciones_por_periodo.csv"
        assignments_table.to_csv(assignments_path, index=False)
        coverage_path = output_dir / "cobertura_perfiles.csv"
        pd.DataFrame(coverage).to_csv(coverage_path, index=False)
        means_path = output_dir / "perfiles_medios_grupos.csv"
        pd.DataFrame(group_rows).to_csv(means_path, index=False)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
        for axis, split in zip(axes, self.config.splits, strict=True):
            axis.hist(shares[split], bins=np.linspace(0, 1, 26), color="steelblue")
            axis.axvline(threshold, color="darkred", linestyle="--", label="Mediana de train")
            axis.set(
                title=f"{split} (n={len(shares[split])})",
                xlabel="Proporción de actividad en fin de semana",
                ylabel="Número de estaciones",
            )
            axis.legend(fontsize=8)
        figure.tight_layout()
        plot = output_dir / "p3_proporcion_fin_semana.png"
        figure.savefig(plot, dpi=140)
        plt.close(figure)
        result.artifacts = [assignments_path, coverage_path, means_path, plot]
        return result
