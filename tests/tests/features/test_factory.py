from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from features.factory import FeaturePipelineFactory
from features.registry import TargetEncoderOOF
from features.schema import FeatureVersionConfig
from data.split import TemporalSplitter


def _split(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    return df[df["dataset_type"] == "TRAIN"], df[df["dataset_type"] == "VALID"], df[df["dataset_type"] == "TEST"]


def test_id_like_feature_excluded_from_output(synthetic_df, feature_version_yaml_text):
    train, valid, _ = _split(synthetic_df)
    config = FeatureVersionConfig.model_validate(yaml.safe_load(feature_version_yaml_text))
    factory = FeaturePipelineFactory(config)
    fitted = factory.fit(train, train["fraude"])
    names = factory.get_feature_names(fitted, train.iloc[:1])
    assert not any(name.startswith("k") for name in names)
    X_valid = factory.transform(valid, fitted)
    assert X_valid.shape[0] == len(valid)


def test_target_encoder_oof_differs_from_naive_full_mean_encoding(synthetic_df):
    train, _, _ = _split(synthetic_df)
    y = train["fraude"].to_numpy()
    X = train[["j"]]

    oof = TargetEncoderOOF(n_splits=5, seed=0)
    oof_values = oof.fit_transform(X, y).reshape(-1)

    # Encoding "ingenuo" (leaky): una sola pasada, sin folds, cada fila ve su propio target.
    naive = TargetEncoderOOF(n_splits=5, seed=0)
    naive.fit(X, y)
    naive_values = naive.transform(X).reshape(-1)

    assert not np.allclose(oof_values, naive_values), (
        "El encoding OOF no deberia coincidir exactamente con el mapeo ingenuo "
        "ajustado sobre el mismo conjunto que codifica (evidencia de proteccion anti-leakage)."
    )


def test_imputer_fits_median_only_on_train(synthetic_df, feature_version_yaml_text):
    train, valid, _ = _split(synthetic_df)
    config = FeatureVersionConfig.model_validate(yaml.safe_load(feature_version_yaml_text))
    factory = FeaturePipelineFactory(config)
    fitted = factory.fit(train, train["fraude"])

    valid_with_outlier = valid.copy()
    valid_with_outlier.loc[valid_with_outlier.index[0], "a"] = 10_000_000.0  # outlier solo en VALID

    pipeline: Pipeline = fitted.named_transformers_["a"]
    imputer = dict(pipeline.steps)["median_imputer"]
    expected_median = train["a"].astype(float).median()
    assert np.isclose(imputer.statistics_[0], expected_median)

    # No debe lanzar y el outlier de VALID no puede haber influido en el estadistico aprendido.
    factory.transform(valid_with_outlier, fitted)
    assert np.isclose(imputer.statistics_[0], expected_median)


def test_unseen_category_in_valid_does_not_raise(synthetic_df, feature_version_yaml_text):
    train, valid, _ = _split(synthetic_df)
    config = FeatureVersionConfig.model_validate(yaml.safe_load(feature_version_yaml_text))
    factory = FeaturePipelineFactory(config)
    fitted = factory.fit(train, train["fraude"])

    valid_with_new_cat = valid.copy()
    valid_with_new_cat.loc[valid_with_new_cat.index[0], "g"] = "CATEGORIA_NUNCA_VISTA"
    X = factory.transform(valid_with_new_cat, fitted)
    assert X.shape[0] == len(valid_with_new_cat)
