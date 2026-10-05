"""`ExperimentRunner`: resuelve cache+clase+hiperparametros por las 4 versiones explicitas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data.manifest import DatasetManifest, get_git_sha
from features.cache import FeatureCacheResolver
from features.schema import load_feature_version
from models.registry import (  # noqa: F401 - MODEL_REGISTRY/UnknownModelError se re-exportan
    MODEL_REGISTRY,
    UnknownModelError,
    resolve_model_class,
)
from training.artifacts import save_run_artifacts
from training.callbacks import EpochMetricsCallback, LiveTrainingMonitor
from training.history import EpochHistory


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    best_epoch: Any
    best_valid_pr_auc: float
    history: EpochHistory


class ExperimentRunner:
    def __init__(
        self,
        feature_version: str,
        model_version: str,
        hyperparameters_version: str,
        model: str,
        seed: int = 42,
        data_root: Path = Path("data"),
        configs_root: Path = Path("configs"),
        artifacts_root: Path = Path("artifacts"),
    ):
        self.feature_version = feature_version
        self.model_version = model_version
        self.hyperparameters_version = hyperparameters_version
        self.model = model
        self.seed = seed
        self.data_root = Path(data_root)
        self.configs_root = Path(configs_root)
        self.artifacts_root = Path(artifacts_root)

    def _compute_run_id(self, feature_cache_id: str, hyperparams_raw_text: str) -> str:
        combined = f"{feature_cache_id}{hyperparams_raw_text}{self.model_version}{self.seed}"
        hash8 = hashlib.sha256(combined.encode()).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return (
            f"{self.model}-{self.model_version}-{self.hyperparameters_version}-"
            f"{self.feature_version}-seed{self.seed}-{timestamp}-{hash8}"
        )

    def run(self) -> RunResult:
        model_class = resolve_model_class(self.model_version, self.model)

        manifest = DatasetManifest.from_json(self.data_root / "processed" / "manifest.json")
        feature_yaml_path = self.configs_root / "feature_versions" / f"{self.feature_version}.yml"
        feature_config = load_feature_version(feature_yaml_path)

        cache_root = self.data_root / "cache" / "features"
        cache_paths = FeatureCacheResolver().resolve(
            self.feature_version, manifest, feature_config, cache_root
        )

        X_train, y_train = cache_paths.load_split("train")
        X_valid, y_valid = cache_paths.load_split("valid")
        feature_names = json.loads(cache_paths.feature_names.read_text())

        hp_path = (
            self.configs_root
            / "hyperparameter_versions"
            / self.hyperparameters_version
            / f"{self.model}.yml"
        )
        hyperparameters = yaml.safe_load(hp_path.read_text())

        cache_metadata = cache_paths.load_metadata()
        run_id = self._compute_run_id(cache_metadata["feature_cache_id"], hp_path.read_text())
        run_dir = self.artifacts_root / "runs" / run_id

        model_instance = model_class(
            hyperparameters=hyperparameters,
            seed=self.seed,
            feature_names=feature_names,
            input_dim=X_train.shape[1],
        )

        experiment_cfg = yaml.safe_load((self.configs_root / "experiments" / "v1.yml").read_text())[
            "metrics"
        ]
        callback = EpochMetricsCallback(
            thresholds=experiment_cfg["thresholds"],
            top_k_percent=experiment_cfg["top_k_percent"],
            calibration_bins=experiment_cfg.get("calibration_bins", 10),
            store_scores_by_epoch=experiment_cfg.get("store_scores_by_epoch", True),
        )
        max_epochs = hyperparameters.get("trainer", {}).get("max_epochs")
        monitor = LiveTrainingMonitor(
            total_epochs=max_epochs, desc=f"{self.model} ({self.feature_version})"
        )

        def combined_on_epoch_end(epoch: Any, scores_by_split: dict, extra: dict) -> dict:
            metrics_this_epoch = callback.on_epoch_end(epoch, scores_by_split, extra)
            monitor.on_epoch_end(epoch, metrics_this_epoch)
            return metrics_this_epoch

        try:
            history = model_instance.fit(
                X_train, y_train, X_valid, y_valid, on_epoch_end=combined_on_epoch_end
            )
        finally:
            monitor.close()

        metadata = {
            "run_id": run_id,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "hyperparameters_version": self.hyperparameters_version,
            "model": self.model,
            "seed": self.seed,
            "git_sha": get_git_sha(),
            "feature_cache_id": cache_metadata["feature_cache_id"],
            "hyperparameters": hyperparameters,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_run_artifacts(
            run_dir,
            model_instance,
            cache_paths.preprocessor,
            feature_names,
            metadata,
            history,
            callback,
        )

        best_epoch = history.best_epoch
        best_valid_pr_auc = float("nan")
        if best_epoch is not None:
            best_valid_pr_auc = history.records[best_epoch]["valid"]["average_precision"]

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            best_epoch=best_epoch,
            best_valid_pr_auc=best_valid_pr_auc,
            history=history,
        )
