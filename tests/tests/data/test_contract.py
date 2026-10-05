from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data.contract import DataContract, DataContractError, DataVersionConfig


def _write_config(tmp_path: Path, yaml_text: str) -> Path:
    path = tmp_path / "v1.yml"
    path.write_text(yaml_text)
    return path


def test_valid_synthetic_df_has_no_violations(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    report = contract.validate(synthetic_df)
    assert report.is_valid, report.violations


def test_missing_expected_column_is_violation(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    df = synthetic_df.drop(columns=["monto"])
    report = contract.validate(df)
    assert not report.is_valid
    assert any("monto" in v for v in report.violations)


def test_target_out_of_allowed_values_is_violation(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    df = synthetic_df.copy()
    df.loc[df.index[0], "fraude"] = 2
    report = contract.validate(df)
    assert not report.is_valid
    assert any("fraude" in v for v in report.violations)


def test_duplicates_are_reported_as_warning_not_violation(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    report = contract.validate(synthetic_df)
    assert report.stats["duplicate_rows"] >= 1
    assert any("duplicadas" in w for w in report.warnings)


def test_nulls_within_tolerance_are_warnings(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    report = contract.validate(synthetic_df)
    assert report.is_valid
    assert any("dentro de tolerancia" in w for w in report.warnings)


def test_nulls_beyond_tolerance_is_violation(tmp_path, synthetic_df, data_version_yaml_text):
    raw = yaml.safe_load(data_version_yaml_text)
    raw["validation_rules"]["nulls"]["max_null_fraction"]["b"] = 0.001
    config_path = tmp_path / "v1.yml"
    config_path.write_text(yaml.safe_dump(raw))
    contract = DataContract.from_yaml(config_path)
    report = contract.validate(synthetic_df)
    assert not report.is_valid
    assert any("excede tolerancia" in v for v in report.violations)


def test_unexpected_nulls_without_tolerance_is_violation(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    df = synthetic_df.copy()
    df.loc[df.index[0], "a"] = None
    report = contract.validate(df)
    assert not report.is_valid
    assert any("sin tolerancia configurada" in v for v in report.violations)


def test_validate_or_raise_raises_on_violation(tmp_path, synthetic_df, data_version_yaml_text):
    config_path = _write_config(tmp_path, data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    df = synthetic_df.drop(columns=["monto"])
    with pytest.raises(DataContractError):
        contract.validate_or_raise(df)


def test_invalid_yaml_raises_validation_error(tmp_path):
    bad_path = tmp_path / "bad.yml"
    bad_path.write_text(yaml.safe_dump({"version": "v1"}))  # faltan campos requeridos
    with pytest.raises(Exception):
        DataVersionConfig.from_yaml(bad_path)
