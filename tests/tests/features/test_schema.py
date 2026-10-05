from __future__ import annotations

import yaml
import pytest
from pydantic import ValidationError

from features.schema import FeatureVersionConfig, load_feature_version


def _parsed(feature_version_yaml_text: str) -> dict:
    return yaml.safe_load(feature_version_yaml_text)


def test_valid_yaml_parses_completely(tmp_path, feature_version_yaml_text):
    path = tmp_path / "v1.yml"
    path.write_text(feature_version_yaml_text)
    known_cols = {"a", "b", "g", "j", "o", "p", "k", "monto", "fecha", "fraude"}
    config = load_feature_version(path, known_columns=known_cols)
    assert config.target == "fraude"
    assert len(config.enabled_features()) == 8  # todas menos 'k' (disabled)


def test_target_as_feature_raises(feature_version_yaml_text):
    raw = _parsed(feature_version_yaml_text)
    raw["features"][0]["name"] = "fraude"
    raw["features"][0]["role"] = "numeric"
    with pytest.raises(ValidationError):
        FeatureVersionConfig.model_validate(raw)


def test_unknown_column_raises(tmp_path, feature_version_yaml_text):
    path = tmp_path / "v1.yml"
    path.write_text(feature_version_yaml_text)
    with pytest.raises(ValueError, match="Columnas desconocidas"):
        load_feature_version(path, known_columns={"a"})  # faltan casi todas


def test_duplicated_transform_raises(feature_version_yaml_text):
    raw = _parsed(feature_version_yaml_text)
    raw["features"][0]["transforms"] = ["median_imputer", "median_imputer"]
    with pytest.raises(ValidationError, match="duplicados"):
        FeatureVersionConfig.model_validate(raw)


def test_incompatible_role_transform_raises(feature_version_yaml_text):
    raw = _parsed(feature_version_yaml_text)
    raw["features"][0]["transforms"] = ["one_hot"]  # one_hot no permitido para role=numeric
    with pytest.raises(ValidationError, match="no permitidos"):
        FeatureVersionConfig.model_validate(raw)


def test_post_decision_availability_raises(feature_version_yaml_text):
    raw = _parsed(feature_version_yaml_text)
    raw["features"][0]["availability"] = "post_decision"
    with pytest.raises(ValidationError, match="post_decision"):
        FeatureVersionConfig.model_validate(raw)


def test_id_like_enabled_raises(feature_version_yaml_text):
    raw = _parsed(feature_version_yaml_text)
    for feat in raw["features"]:
        if feat["name"] == "k":
            feat["enabled"] = True
    with pytest.raises(ValidationError, match="id_like"):
        FeatureVersionConfig.model_validate(raw)
