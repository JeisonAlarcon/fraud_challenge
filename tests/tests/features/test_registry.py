from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from features.registry import (
    DatetimeFeaturizer,
    ExplicitBinaryMapper,
    UnknownTransformError,
    build_transform,
)


def test_build_transform_known_name_returns_expected_type():
    transformer = build_transform("robust_scaler")
    assert isinstance(transformer, RobustScaler)


def test_build_transform_unknown_name_raises():
    with pytest.raises(UnknownTransformError):
        build_transform("no_existe")


def test_explicit_binary_mapper_maps_y_n_and_default():
    mapper = ExplicitBinaryMapper()
    X = pd.DataFrame({"o": ["Y", "N", "missing", "N"]})
    out = mapper.fit_transform(X)
    np.testing.assert_array_equal(out.reshape(-1), [1.0, 0.0, 0.5, 0.0])


@pytest.mark.parametrize("kind", ["cyclic_hour", "cyclic_day"])
def test_datetime_featurizer_cyclic_in_unit_circle(kind):
    dates = pd.DataFrame({"fecha": pd.date_range("2020-03-08", periods=48, freq="h")})
    featurizer = DatetimeFeaturizer(kind=kind)
    out = featurizer.fit_transform(dates)
    assert out.shape == (48, 2)
    assert np.all(out >= -1.0 - 1e-9) and np.all(out <= 1.0 + 1e-9)


def test_datetime_featurizer_hour_range():
    dates = pd.DataFrame({"fecha": pd.date_range("2020-03-08", periods=24, freq="h")})
    out = DatetimeFeaturizer(kind="hour").fit_transform(dates)
    assert out.reshape(-1).tolist() == list(range(24))


@pytest.mark.parametrize("name", ["frequency_encoder", "count_encoder", "ordinal_encoder", "hashing_encoder"])
def test_categorical_encoders_produce_numeric_output_for_unseen_category(name):
    X_train = pd.DataFrame({"g": ["A", "A", "B", "C", "C", "C"]})
    transformer = build_transform(name)
    transformer.fit(X_train)
    out = transformer.transform(pd.DataFrame({"g": ["A", "NUNCA_VISTA"]}))
    assert out.shape[0] == 2
    assert np.issubdtype(np.asarray(out).dtype, np.number)
