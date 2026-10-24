"""P3: algoritmos de agrupamiento candidatos sobre perfiles semanales de estación."""

from pathlib import Path

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
        return result
