from __future__ import annotations

import yaml
import pytest

from data.contract import DataContract
from data.prepare import DatasetBuilder
from data.split import TemporalSplitter
from features.cache import FeatureCacheBuilder
from features.schema import FeatureVersionConfig
from training.runner import ExperimentRunner, UnknownModelError, resolve_model_class
from models.versions.v1.sklearn import DummyModel


def _bootstrap_project(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    configs_root = tmp_path / "configs"
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"

    (configs_root / "data_versions").mkdir(parents=True)
    (configs_root / "feature_versions").mkdir(parents=True)
    (configs_root / "hyperparameter_versions" / "v1").mkdir(parents=True)
    (configs_root / "experiments").mkdir(parents=True)

    (configs_root / "data_versions" / "v1.yml").write_text(data_version_yaml_text)
    (configs_root / "feature_versions" / "v1.yml").write_text(feature_version_yaml_text)
    (configs_root / "hyperparameter_versions" / "v1" / "dummy.yml").write_text("parameters:\n  strategy: prior\n")
    (configs_root / "experiments" / "v1.yml").write_text(
        yaml.safe_dump({"metrics": {"thresholds": [0.5], "top_k_percent": [10], "calibration_bins": 5, "store_scores_by_epoch": True}})
    )

    contract = DataContract.from_yaml(configs_root / "data_versions" / "v1.yml")
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    csv_path = tmp_path / "raw.csv"
    synthetic_df.to_csv(csv_path, index=False)
    builder = DatasetBuilder(contract, splitter, data_root / "processed")
    manifest = builder.build(csv_path)

    feature_config = FeatureVersionConfig.model_validate(yaml.safe_load(feature_version_yaml_text))
    parquet_path = data_root / "processed" / "fraud_clean_v1.parquet"
    FeatureCacheBuilder().build(parquet_path, manifest, "v1", feature_config, data_root / "cache" / "features")

    return configs_root, data_root, artifacts_root


def test_run_id_is_deterministic_given_same_inputs(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    configs_root, data_root, artifacts_root = _bootstrap_project(
        tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text
    )
    runner = ExperimentRunner(
        feature_version="v1", model_version="v1", hyperparameters_version="v1", model="dummy",
        seed=0, data_root=data_root, configs_root=configs_root, artifacts_root=artifacts_root,
    )
    run_id_a = runner._compute_run_id("abc123", "parameters:\n  strategy: prior\n")
    run_id_b = runner._compute_run_id("abc123", "parameters:\n  strategy: prior\n")
    # mismo hash8 (misma parte determinista); el timestamp difiere solo si cambia el segundo exacto
    assert run_id_a.rsplit("-", 1)[-1] == run_id_b.rsplit("-", 1)[-1]


def test_run_end_to_end_with_dummy_model(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    configs_root, data_root, artifacts_root = _bootstrap_project(
        tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text
    )
    runner = ExperimentRunner(
        feature_version="v1", model_version="v1", hyperparameters_version="v1", model="dummy",
        seed=0, data_root=data_root, configs_root=configs_root, artifacts_root=artifacts_root,
    )
    result = runner.run()
    assert (result.run_dir / "model" / "model.pkl").exists()
    assert (result.run_dir / "metrics" / "epoch_metrics.json").exists()
    assert result.best_epoch == "final"


def test_unknown_model_raises(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    with pytest.raises(UnknownModelError):
        resolve_model_class("v1", "no_existe")


def test_unknown_model_version_raises():
    with pytest.raises(UnknownModelError):
        resolve_model_class("v99", "dummy")
