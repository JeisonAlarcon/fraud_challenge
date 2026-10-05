from __future__ import annotations

import copy

import yaml

from data.contract import DataContract
from data.prepare import DatasetBuilder
from data.split import TemporalSplitter
from features.cache import (
    FeatureCacheBuilder,
    FeatureCacheNotFoundError,
    FeatureCacheResolver,
)
from features.schema import FeatureVersionConfig


def _build_manifest(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = tmp_path / "data_v1.yml"
    config_path.write_text(data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    csv_path = tmp_path / "raw.csv"
    synthetic_df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    builder = DatasetBuilder(contract, splitter, output_dir)
    manifest = builder.build(csv_path)
    return manifest, output_dir / "fraud_clean_v1.parquet"


def _feature_config(feature_version_yaml_text) -> FeatureVersionConfig:
    return FeatureVersionConfig.model_validate(yaml.safe_load(feature_version_yaml_text))


def test_cache_miss_then_hit_same_directory(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    manifest, parquet_path = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache"
    builder = FeatureCacheBuilder()

    paths_miss = builder.build(parquet_path, manifest, "v1", feature_config, cache_root)
    assert paths_miss.metadata.exists()
    mtime_after_miss = paths_miss.metadata.stat().st_mtime

    paths_hit = builder.build(parquet_path, manifest, "v1", feature_config, cache_root)
    assert paths_hit.root == paths_miss.root
    assert paths_hit.metadata.stat().st_mtime == mtime_after_miss  # no se reescribio


def test_hash_changes_when_feature_yaml_changes(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    manifest, parquet_path = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache"
    builder = FeatureCacheBuilder()

    paths_v1 = builder.build(parquet_path, manifest, "v1", feature_config, cache_root)

    raw = copy.deepcopy(yaml.safe_load(feature_version_yaml_text))
    for feat in raw["features"]:
        if feat["name"] == "g":
            feat["transforms"] = ["missing_token", "rare_category", "ordinal_encoder"]
    feature_config_v2 = FeatureVersionConfig.model_validate(raw)
    paths_v2 = builder.build(parquet_path, manifest, "v1", feature_config_v2, cache_root)

    assert paths_v1.root != paths_v2.root
    assert paths_v1.root.exists()  # el cache anterior no se sobreescribe


def test_force_rewrites_without_changing_cache_id(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    manifest, parquet_path = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache"
    builder = FeatureCacheBuilder()

    paths_1 = builder.build(parquet_path, manifest, "v1", feature_config, cache_root)
    mtime_1 = paths_1.metadata.stat().st_mtime
    paths_2 = builder.build(parquet_path, manifest, "v1", feature_config, cache_root, force=True)

    assert paths_2.root == paths_1.root
    assert paths_2.metadata.stat().st_mtime >= mtime_1


def test_cache_artifacts_are_dense_or_sparse_consistently(
    tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text
):
    manifest, parquet_path = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache"
    builder = FeatureCacheBuilder()
    paths = builder.build(parquet_path, manifest, "v1", feature_config, cache_root)

    metadata = paths.load_metadata()
    assert metadata["is_sparse"] is False  # la densidad combinada supera sparse_threshold
    X_train, y_train = paths.load_split("train")
    assert X_train.shape[0] == y_train.shape[0]
    assert set(y_train.tolist()) <= {0, 1}
    assert paths.feature_names.exists() and paths.input_schema.exists()


def test_resolver_raises_when_cache_missing(tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text):
    manifest, _ = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache_empty"
    resolver = FeatureCacheResolver()
    try:
        resolver.resolve("v1", manifest, feature_config, cache_root)
        raised = False
    except FeatureCacheNotFoundError:
        raised = True
    assert raised


def test_resolver_finds_cache_built_by_builder(
    tmp_path, synthetic_df, data_version_yaml_text, feature_version_yaml_text
):
    manifest, parquet_path = _build_manifest(tmp_path, synthetic_df, data_version_yaml_text)
    feature_config = _feature_config(feature_version_yaml_text)
    cache_root = tmp_path / "cache"
    built_paths = FeatureCacheBuilder().build(parquet_path, manifest, "v1", feature_config, cache_root)
    resolved_paths = FeatureCacheResolver().resolve("v1", manifest, feature_config, cache_root)
    assert resolved_paths.root == built_paths.root
