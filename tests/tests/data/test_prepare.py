from __future__ import annotations

import pandas as pd
import pytest

from data.contract import DataContract, DataContractError
from data.prepare import DatasetBuilder
from data.split import TemporalSplitter


def _make_builder(tmp_path, data_version_yaml_text, output_dir):
    config_path = tmp_path / "v1.yml"
    config_path.write_text(data_version_yaml_text)
    contract = DataContract.from_yaml(config_path)
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    return DatasetBuilder(contract, splitter, output_dir)


def test_build_writes_parquet_with_dataset_type_and_ascending_order(
    tmp_path, synthetic_df, data_version_yaml_text
):
    csv_path = tmp_path / "raw.csv"
    synthetic_df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    builder = _make_builder(tmp_path, data_version_yaml_text, output_dir)

    manifest = builder.build(csv_path)

    out_path = output_dir / "fraud_clean_v1.parquet"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert set(df["dataset_type"].unique()) == {"TRAIN", "VALID", "TEST"}
    assert df["fecha"].is_monotonic_increasing
    assert manifest.output_parquet.n_rows == len(df)


def test_build_is_idempotent_without_force(tmp_path, synthetic_df, data_version_yaml_text):
    csv_path = tmp_path / "raw.csv"
    synthetic_df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    builder = _make_builder(tmp_path, data_version_yaml_text, output_dir)

    manifest_1 = builder.build(csv_path)
    parquet_mtime_1 = (output_dir / "fraud_clean_v1.parquet").stat().st_mtime
    manifest_2 = builder.build(csv_path)
    parquet_mtime_2 = (output_dir / "fraud_clean_v1.parquet").stat().st_mtime

    assert manifest_1.output_parquet.sha256 == manifest_2.output_parquet.sha256
    assert parquet_mtime_1 == parquet_mtime_2  # no se reescribio


def test_build_force_rewrites(tmp_path, synthetic_df, data_version_yaml_text):
    csv_path = tmp_path / "raw.csv"
    synthetic_df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    builder = _make_builder(tmp_path, data_version_yaml_text, output_dir)

    builder.build(csv_path)
    manifest_forced = builder.build(csv_path, force=True)
    assert manifest_forced.output_parquet.n_rows > 0


def test_build_propagates_contract_error_on_hard_violation(
    tmp_path, synthetic_df, data_version_yaml_text
):
    df = synthetic_df.drop(columns=["monto"])
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    builder = _make_builder(tmp_path, data_version_yaml_text, output_dir)

    with pytest.raises(DataContractError):
        builder.build(csv_path)
