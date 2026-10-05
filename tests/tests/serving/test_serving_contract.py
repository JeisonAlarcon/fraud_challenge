from __future__ import annotations

from datetime import datetime, timezone

import pytest

from serving.contract import (
    ContractValidationError,
    FeatureColumnSpec,
    FeatureContract,
    validate_raw_features,
)


def _make_contract() -> FeatureContract:
    return FeatureContract(
        feature_version="v1",
        feature_cache_id="abc123",
        target="fraude",
        timestamp_column="fecha",
        input_columns=[
            FeatureColumnSpec(name="monto", dtype="float64", role="numeric", nullable=False, valid_range=(0, None)),
            FeatureColumnSpec(name="o", dtype="string", role="binary", nullable=True),
        ],
        excluded_columns=[{"name": "k", "role": "id_like", "reason": "cardinalidad unica"}],
        preprocessor_output_dim=10,
        preprocessor_hash="deadbeef",
        source_manifest_hash="feedface",
        generated_at=datetime.now(timezone.utc),
    )


def test_valid_raw_features_pass():
    contract = _make_contract()
    validate_raw_features(contract, {"monto": 50.0, "o": "Y"})  # no debe lanzar


def test_missing_column_raises():
    contract = _make_contract()
    with pytest.raises(ContractValidationError, match="Columnas faltantes"):
        validate_raw_features(contract, {"monto": 50.0})


def test_wrong_dtype_raises():
    contract = _make_contract()
    with pytest.raises(ContractValidationError, match="esperaba numero"):
        validate_raw_features(contract, {"monto": "no-es-numero", "o": "Y"})


def test_non_nullable_none_raises():
    contract = _make_contract()
    with pytest.raises(ContractValidationError, match="no admite nulos"):
        validate_raw_features(contract, {"monto": None, "o": "Y"})


def test_out_of_range_raises():
    contract = _make_contract()
    with pytest.raises(ContractValidationError, match="por debajo de min"):
        validate_raw_features(contract, {"monto": -10.0, "o": "Y"})


def test_nullable_column_accepts_none():
    contract = _make_contract()
    validate_raw_features(contract, {"monto": 50.0, "o": None})  # no debe lanzar
