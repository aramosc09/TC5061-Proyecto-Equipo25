"""Baselines desde cero: persistencia semanal (P1) y clase mayoritaria (P2)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from proyecto_mibici.config import SUPERVISED_HORIZON_HOURS
from proyecto_mibici.modeling.base import BaselineResult, PipelinePregunta

PRESSURE_CLASSES = (
    "sin_presion_neta",
    "extraccion_baja",
    "acumulacion_baja",
    "extraccion_moderada",
    "acumulacion_moderada",
    "extraccion_alta",
    "acumulacion_alta",
)


def pressure_labels(pressure: np.ndarray, q75: float, q90: float) -> np.ndarray:
    """Codifica las siete clases del proxy con umbrales congelados de entrenamiento."""
    magnitude = np.abs(pressure)
    intensity = np.where(magnitude <= q75, 0, np.where(magnitude <= q90, 1, 2))
    labels = (1 + 2 * intensity + (pressure < 0)).astype("int8")
    labels[pressure == 0] = 0
    return labels


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Obtiene F1 macro de las siete clases y métricas por clase con soporte explícito.

    Una precisión/recall indefinida no se registra como cero: se informa su
    denominador. F1 macro usa etiquetas fijas y zero_division=0 para comparar
    corridas sobre el mismo universo de clases, incluidas las clases vacías.
    """
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    f1 = np.divide(
        2 * true_positive,
        support + predicted,
        out=np.zeros(7, dtype=float),
        where=(support + predicted) > 0,
    )
    total = int(matrix.sum())
    result = {
        "f1_macro": float(f1.mean()),
        "n_observations": total,
        "alert_rate": float(predicted[5:].sum() / total) if total else 0.0,
        "true_high_pressure_rate": float(support[5:].sum() / total) if total else 0.0,
    }
    for index, name in enumerate(PRESSURE_CLASSES):
        result[f"support_{name}"] = int(support[index])
        result[f"predicted_{name}"] = int(predicted[index])
        if support[index]:
            result[f"recall_{name}"] = float(true_positive[index] / support[index])
        if predicted[index]:
            result[f"precision_{name}"] = float(true_positive[index] / predicted[index])
    return result


class PipelineP1(PipelinePregunta):
    def run(self, repository, output_dir: Path) -> BaselineResult:
        """Predice salidas con la misma hora de la semana anterior, sin ajuste al futuro."""
        output_dir.mkdir(parents=True, exist_ok=True)
        totals = {split: np.zeros(4) for split in self.config.splits}
        rows = []
        hourly = {split: np.zeros((24, 3)) for split in self.config.splits}
        for station, hours in repository.station_hours():
            prediction = hours.salidas.shift(self.config.weekly_lag)
            previous_eligible = hours.elegible.shift(self.config.weekly_lag, fill_value=False)
            valid = hours.elegible & previous_eligible & prediction.notna()
            for split, (start, end) in self.config.splits.items():
                mask = valid & (hours.index >= start) & (hours.index < end)
                actual = hours.loc[mask, "salidas"].to_numpy(dtype=float)
                forecast = prediction.loc[mask].to_numpy(dtype=float)
                error = forecast - actual
                stats = np.array(
                    [len(actual), np.abs(error).sum(), np.square(error).sum(), error.sum()]
                )
                totals[split] += stats
                rows.append(
                    {
                        "estacion": station,
                        "conjunto": split,
                        "n": len(actual),
                        "n_excluidas": int(((hours.index >= start) & (hours.index < end)).sum())
                        - len(actual),
                        "mae": float(np.abs(error).mean()) if len(actual) else np.nan,
                        "rmse": float(np.sqrt(np.square(error).mean())) if len(actual) else np.nan,
                        "mean_error": float(error.mean()) if len(actual) else np.nan,
                    }
                )
                for hour in range(24):
                    selected = hours.index[mask].hour == hour
                    hourly[split][hour] += [
                        int(selected.sum()),
                        actual[selected].sum(),
                        forecast[selected].sum(),
                    ]
        result = BaselineResult(
            "P1",
            "seasonal_naive",
            parameters={
                "model": "seasonal_naive",
                "horizon_hours": SUPERVISED_HORIZON_HOURS,
                "lag_hours": self.config.weekly_lag,
                "fitted_parameters": "none",
                "forecast_rounding": "none",
                "eligibility": "current_and_lag_hour_eligible_from_past_only",
            },
        )
        for split, (n, absolute, square, signed) in totals.items():
            if n == 0:
                raise ValueError(f"P1 no tiene observaciones elegibles en {split}.")
            result.metrics.update(
                {
                    f"{split}_mae": absolute / n,
                    f"{split}_rmse": np.sqrt(square / n),
                    f"{split}_mean_error": signed / n,
                    f"{split}_n_observations": int(n),
                }
            )
        table = output_dir / "errores_por_estacion.csv"
        pd.DataFrame(rows).to_csv(table, index=False)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for axis, split in zip(axes, ("validation", "test"), strict=True):
            data = hourly[split]
            denominator = np.where(data[:, 0] > 0, data[:, 0], np.nan)
            axis.plot(range(24), data[:, 1] / denominator, label="Observado")
            axis.plot(range(24), data[:, 2] / denominator, label="Semana anterior")
            axis.set(title=split, xlabel="Hora del día", ylabel="Viajes medios / estación-hora")
            axis.legend()
        figure.tight_layout()
        plot = output_dir / "p1_perfil_observado_vs_baseline.png"
        figure.savefig(plot, dpi=140)
        plt.close(figure)
        result.artifacts = [table, plot]
        return result


class PipelineP2(PipelinePregunta):
    def run(self, repository, output_dir: Path) -> BaselineResult:
        """Ajusta percentiles y DummyClassifier solo en entrenamiento; evalúa el proxy."""
        output_dir.mkdir(parents=True, exist_ok=True)
        capacities = repository.stations().set_index("id")["dpcapacity"].to_dict()
        pressures = []
        excluded = set()
        train_start, train_end = self.config.splits["train"]
        for station, hours in repository.station_hours():
            capacity = capacities.get(station)
            if pd.isna(capacity) or capacity is None or capacity <= 0:
                excluded.add(station)
                continue
            mask = hours.elegible & (hours.index >= train_start) & (hours.index < train_end)
            pressures.append(
                ((hours.salidas - hours.llegadas).loc[mask] / capacity).to_numpy(dtype=float)
            )
        training = np.concatenate(pressures) if pressures else np.array([])
        if not training.size:
            raise ValueError("P2 no tiene horas de entrenamiento con capacidad utilizable.")
        q75, q90 = np.quantile(np.abs(training), self.config.pressure_quantiles, method="linear")
        labels = pressure_labels(training, q75, q90)
        classifier = DummyClassifier(strategy="most_frequent", random_state=self.config.seed)
        classifier.fit(np.zeros((len(labels), 1), dtype="uint8"), labels)
        majority = int(classifier.predict(np.zeros((1, 1)))[0])
        del training, labels, pressures
        matrices = {split: np.zeros((7, 7), dtype="int64") for split in self.config.splits}
        for station, hours in repository.station_hours():
            capacity = capacities.get(station)
            if pd.isna(capacity) or capacity is None or capacity <= 0:
                continue
            pressure = ((hours.salidas - hours.llegadas) / capacity).to_numpy(dtype=float)
            actual = pressure_labels(pressure, q75, q90)
            for split, (start, end) in self.config.splits.items():
                mask = hours.elegible.to_numpy() & (hours.index >= start) & (hours.index < end)
                truth = actual[mask]
                if not len(truth):
                    continue
                prediction = classifier.predict(np.zeros((len(truth), 1), dtype="uint8"))
                matrices[split] += confusion_matrix(truth, prediction, labels=range(7))
        result = BaselineResult(
            "P2",
            "DummyClassifier",
            parameters={
                "model": "DummyClassifier",
                "horizon_hours": SUPERVISED_HORIZON_HOURS,
                "strategy": "most_frequent",
                "random_state": self.config.seed,
                "pressure_q75": float(q75),
                "pressure_q90": float(q90),
                "quantile_method": "linear",
                "quantile_fit_split": "train",
                "quantile_levels": str(self.config.pressure_quantiles),
                "majority_class": PRESSURE_CLASSES[majority],
                "labels": ",".join(PRESSURE_CLASSES),
                "macro_f1_labels": "all_seven_fixed_classes",
                "f1_zero_division": 0,
                "excluded_missing_capacity_stations": len(excluded),
                "target_kind": "retrospective_net_flow_proxy_not_inventory",
            },
        )
        distribution = []
        plots = []
        for split, matrix in matrices.items():
            if not matrix.sum():
                raise ValueError(f"P2 no tiene observaciones evaluables en {split}.")
            result.metrics.update(
                {f"{split}_{key}": value for key, value in classification_metrics(matrix).items()}
            )
            for index, name in enumerate(PRESSURE_CLASSES):
                distribution.append(
                    {
                        "conjunto": split,
                        "clase": name,
                        "observados": int(matrix[index].sum()),
                        "predichos": int(matrix[:, index].sum()),
                    }
                )
            matrix_path = output_dir / f"{split}_matriz_confusion.csv"
            pd.DataFrame(matrix, index=PRESSURE_CLASSES, columns=PRESSURE_CLASSES).to_csv(
                matrix_path
            )
            result.artifacts.append(matrix_path)
            if split != "train":
                figure, axis = plt.subplots(figsize=(9, 7))
                image = axis.imshow(matrix, cmap="Blues")
                axis.set_xticks(range(7), PRESSURE_CLASSES, rotation=60, ha="right", fontsize=8)
                axis.set_yticks(range(7), PRESSURE_CLASSES, fontsize=8)
                axis.set(
                    xlabel="Clase predicha",
                    ylabel="Clase observada del proxy",
                    title=f"P2 - {split}",
                )
                for row in range(7):
                    for column in range(7):
                        axis.text(
                            column,
                            row,
                            f"{matrix[row, column]:,}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                        )
                figure.colorbar(image, ax=axis)
                figure.tight_layout()
                plot = output_dir / f"p2_{split}_matriz_confusion.png"
                figure.savefig(plot, dpi=140)
                plt.close(figure)
                plots.append(plot)
        table = output_dir / "distribucion_clases.csv"
        pd.DataFrame(distribution).to_csv(table, index=False)
        result.artifacts.extend([table, *plots])
        return result
