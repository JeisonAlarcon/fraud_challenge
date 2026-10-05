"""Smoke test end-to-end de la API de promoción."""

from __future__ import annotations

import json

import yaml

from data.contract import DataContract
from data.prepare import DatasetBuilder
from data.split import TemporalSplitter
from features.cache import FeatureCacheBuilder
from features.schema import FeatureVersionConfig
from promotion import promote_winner
from training.runner import ExperimentRunner


def _bootstrap_two_runs(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    configs_root = tmp_path / "configs"
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"

    (configs_root / "data_versions").mkdir(parents=True)
    (configs_root / "feature_versions").mkdir(parents=True)
    (configs_root / "hyperparameter_versions" / "v1").mkdir(parents=True)
    (configs_root / "experiments").mkdir(parents=True)
    (configs_root / "promotion").mkdir(parents=True)

    (configs_root / "data_versions" / "v1.yml").write_text(data_version_yaml_text)
    (configs_root / "feature_versions" / "v1.yml").write_text(feature_version_yaml_text)
    (configs_root / "hyperparameter_versions" / "v1" / "dummy.yml").write_text("parameters:\n  strategy: prior\n")
    (configs_root / "hyperparameter_versions" / "v1" / "logistic_regression.yml").write_text(
        "parameters:\n  max_iter: 200\n"
    )
    (configs_root / "experiments" / "v1.yml").write_text(
        yaml.safe_dump({"metrics": {"thresholds": [0.5], "top_k_percent": [10], "calibration_bins": 5, "store_scores_by_epoch": True}})
    )
    (configs_root / "promotion" / "v1.yml").write_text(
        yaml.safe_dump(
            {
                "policy": {"type": "default_argmax_f1", "params": {}, "defined_by": "default_no_business_policy"},
                "selection": {"min_segment_support": 5, "segment_columns": ["g"], "bootstrap_n": 50, "temporal_window": "7D"},
                "calibration": {"method": "auto"},
            }
        )
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

    for model_name in ("logistic_regression",):
        runner = ExperimentRunner(
            feature_version="v1", model_version="v1", hyperparameters_version="v1", model=model_name,
            seed=0, data_root=data_root, configs_root=configs_root, artifacts_root=artifacts_root,
        )
        runner.run()

    return configs_root, data_root, artifacts_root


def test_promote_winner_end_to_end_and_test_guard(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    configs_root, data_root, artifacts_root = _bootstrap_two_runs(
        tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text
    )
    promote_winner(
        runs_dir=artifacts_root / "runs",
        config_path=configs_root / "promotion" / "v1.yml",
        data_root=data_root,
        configs_root=configs_root,
        promoted_root=artifacts_root / "promoted",
    )

    promoted_dirs = list((artifacts_root / "promoted").iterdir())
    assert len(promoted_dirs) == 1
    promoted_dir = promoted_dirs[0]

    for filename in (
        "model_raw.onnx", "calibrator.pkl", "calibrator.onnx", "feature_contract.json",
        "threshold.json", "final_test_metrics.json", "model_card.md",
    ):
        assert (promoted_dir / filename).exists(), f"falta {filename}"

    test_metrics_path = promoted_dir / "final_test_metrics.json"
    mtime_before = test_metrics_path.stat().st_mtime

    promote_winner(
        runs_dir=artifacts_root / "runs",
        config_path=configs_root / "promotion" / "v1.yml",
        data_root=data_root,
        configs_root=configs_root,
        promoted_root=artifacts_root / "promoted",
    )

    assert test_metrics_path.stat().st_mtime == mtime_before  # TEST no se recalculo
