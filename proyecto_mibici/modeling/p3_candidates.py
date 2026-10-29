"""P3: algoritmos de agrupamiento candidatos sobre perfiles semanales de estación."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz: seguro en scripts, servidores y CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline

from proyecto_mibici.config import CANDIDATE_MODELS, ProjectConfig
from proyecto_mibici.features import PROFILE_COLUMNS, proporcion_fin_semana
from proyecto_mibici.modeling.base import CandidateResult
from proyecto_mibici.modeling.datasets import SPLITS, PerfilesP3
from proyecto_mibici.modeling.p3_baseline import silhouette_euclidea


class AglomerativoConCentroides(ClusterMixin, BaseEstimator):
    """Ward con asignación de periodos posteriores al centroide de train más cercano."""

    def __init__(self, n_clusters: int = 4, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        labels = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage
        ).fit_predict(values)
        self.cluster_centers_ = np.vstack(
            [values[labels == group].mean(axis=0) for group in range(self.n_clusters)]
        )
        self.labels_ = labels
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        distances = ((values[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)


def build_estimator(model: str, params: dict, seed: int):
    if model == "kmeans":
        return KMeans(random_state=seed, **params)
    if model == "agglomerative":
        return AglomerativoConCentroides(**params)
    if model == "gmm":
        mixture = GaussianMixture(
            n_components=params["n_clusters"],
            covariance_type=params["covariance_type"],
            random_state=seed,
        )
        components = params["pca_components"]
        return make_pipeline(PCA(components, random_state=seed), mixture) if components else mixture
    raise ValueError(f"Modelo P3 desconocido: {model}.")


def _check_saved(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"No se pudo guardar la gráfica en {path}.")
    return path


def save_pca_scatter_plot(
    frames: dict[str, pd.DataFrame],
    assignments: dict[str, pd.Series],
    n_groups: int,
    model: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """Proyección PCA 2D: se ajusta con train y se proyectan todos los splits evaluados."""
    path = output_dir / "pca_grupos_por_split.png"
    splits = list(assignments)
    pca = PCA(n_components=2, random_state=seed).fit(np.asarray(frames["train"], dtype=float))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        1, len(splits), figsize=(6 * len(splits), 5.5), sharex=True, sharey=True, squeeze=False
    )
    try:
        for ax, split in zip(axes[0], splits):
            points = pca.transform(np.asarray(frames[split], dtype=float))
            labels = assignments[split].to_numpy()
            for group in range(n_groups):
                mask = labels == group
                if mask.any():
                    ax.scatter(
                        points[mask, 0], points[mask, 1], s=22, alpha=0.7,
                        color=cmap(group % 10), edgecolors="none",
                        label=f"Grupo {group} (n={int(mask.sum())})",
                    )
            ax.set_title(f"{split} (n={len(labels)})")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes[0][0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
        fig.suptitle(f"P3 {model.upper()} - Perfiles de estación en PCA (ajustado con train)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)  # siempre libera la figura, incluso si algo falla
    return _check_saved(path)


def save_group_profiles_plot(
    frame: pd.DataFrame, labels: pd.Series, n_groups: int, model: str, split: str, output_dir: Path
) -> Path:
    """Perfil medio (± 1 desv. estándar) de cada grupo, en el orden de PROFILE_COLUMNS."""
    path = output_dir / f"perfiles_medios_por_grupo_{split}.png"
    values = frame.to_numpy(dtype=float)
    x = np.arange(values.shape[1])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    try:
        for group in range(n_groups):
            mask = (labels == group).to_numpy()
            if not mask.any():
                continue  # grupo vacío en este split
            mean = values[mask].mean(axis=0)
            std = values[mask].std(axis=0)
            ax.plot(x, mean, color=cmap(group % 10), lw=1.6,
                    label=f"Grupo {group} (n={int(mask.sum())})")
            ax.fill_between(x, mean - std, mean + std, color=cmap(group % 10), alpha=0.12)
        # Si las columnas son franjas horarias, marca cada bloque de 24 (un día).
        if values.shape[1] % 24 == 0:
            for boundary in range(24, values.shape[1], 24):
                ax.axvline(boundary - 0.5, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("Franja del perfil semanal (orden de PROFILE_COLUMNS)")
        ax.set_ylabel("Valor normalizado del perfil")
        ax.set_title(f"P3 {model.upper()} - Perfil medio por grupo ({split})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return _check_saved(path)


class CandidatosP3:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def run(
        self,
        data: PerfilesP3,
        model: str,
        params: dict,
        output_dir: Path,
        *,
        max_train_rows: int,
        evaluate_test: bool,
    ) -> CandidateResult:
        """Agrupa perfiles de train y asigna los demás periodos sin reajustar."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = data.frames
        estimator = build_estimator(model, params, self.config.seed).fit(frames["train"])
        threshold = float(proporcion_fin_semana(frames["train"]).median())
        total_stations = len({row["estacion"] for row in data.coverage})
        result = CandidateResult(
            "P3",
            model,
            parameters={
                "model": model,
                **params,
                "horizon_hours": "not_applicable_descriptive",
                "fit_split": "train",
                "assignment_rule": "groups_frozen_in_train_then_predicted",
                "n_profile_features": len(PROFILE_COLUMNS),
                "distance": "euclidean",
                "profile_normalization": "per_eligible_hour_slot_then_joint_sum_one",
                "coordinates_used_for_clustering": False,
                "undefined_metric_policy": "omit_and_record_defined_flag",
                "baseline_on_same_rows": "weekend_share_training_median",
            },
            model_family=CANDIDATE_MODELS["p3"][model]["family"],
            estimator=estimator,
            feature_names=PROFILE_COLUMNS,
            input_example=frames["train"].head(5).copy(),
        )
        n_groups = params["n_clusters"]
        assignments = {}
        for split in SPLITS[: 3 if evaluate_test else 2]:
            frame = frames[split]
            labels = pd.Series(estimator.predict(frame), index=frame.index)
            assignments[split] = labels
            n = len(frame)
            observed = int(labels.nunique())
            defined = 2 <= observed < n
            sizes = labels.value_counts().reindex(range(n_groups), fill_value=0)
            result.metrics.update(
                {
                    f"{split}_n_stations": n,
                    f"{split}_n_groups": observed,
                    f"{split}_min_group_share": float(sizes.min() / n),
                    f"{split}_silhouette_defined": int(defined),
                    f"{split}_excluded_stations": total_stations - n,
                    f"{split}_eligible_hours": sum(
                        row["horas_elegibles"] for row in data.coverage if row["conjunto"] == split
                    ),
                    **{f"{split}_group_{group}_size": int(size) for group, size in sizes.items()},
                }
            )
            if defined:
                result.metrics[f"{split}_silhouette"] = silhouette_euclidea(frame, labels)
            baseline = proporcion_fin_semana(frame).gt(threshold).astype("int8")
            if 2 <= baseline.nunique() < n:
                result.metrics[f"{split}_baseline_silhouette"] = silhouette_euclidea(
                    frame, baseline
                )
            if split != "train":
                common = labels.index.intersection(assignments["train"].index)
                train_labels = assignments["train"].loc[common]
                other_labels = labels.loc[common]
                ari_defined = (
                    len(common) >= 2 and train_labels.nunique() >= 2 and other_labels.nunique() >= 2
                )
                result.metrics[f"{split}_common_stations_vs_train"] = len(common)
                result.metrics[f"{split}_ari_vs_train_defined"] = int(ari_defined)
                if ari_defined:
                    result.metrics[f"{split}_ari_vs_train"] = float(
                        adjusted_rand_score(train_labels, other_labels)
                    )

        # Gráficas: PCA por split y perfil medio por grupo (train). Las sube el runner.
        result.artifacts.extend(
            [
                save_pca_scatter_plot(
                    frames, assignments, n_groups, model, self.config.seed, output_dir
                ),
                save_group_profiles_plot(
                    frames["train"], assignments["train"], n_groups, model, "train", output_dir
                ),
            ]
        )
        return result