from __future__ import annotations

import numpy as np
import pytest

from models.versions.v1.sklearn import (
    DummyModel,
    GradientBoostingModel,
    HistGradientBoostingModel,
    LogisticRegressionModel,
)

MODEL_CLASSES = [DummyModel, LogisticRegressionModel, HistGradientBoostingModel, GradientBoostingModel]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_fit_predict_proba_in_unit_interval(model_class, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = model_class(hyperparameters={}, seed=0, feature_names=[f"f{i}" for i in range(X_train.shape[1])], input_dim=X_train.shape[1])
    history = model.fit(X_train, y_train, X_valid, y_valid)

    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert "final" in history.records
    assert "average_precision" in history.records["final"]["valid"]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_save_load_roundtrip_preserves_predictions(tmp_path, model_class, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = model_class(hyperparameters={}, seed=0, feature_names=[], input_dim=X_train.shape[1])
    model.fit(X_train, y_train, X_valid, y_valid)
    proba_before = model.predict_proba(X_valid)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = model_class.load(path)
    proba_after = loaded.predict_proba(X_valid)

    np.testing.assert_allclose(proba_before, proba_after)


def test_gradient_boosting_balanced_uses_sample_weight_not_uniform(synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    balanced = GradientBoostingModel(
        hyperparameters={"parameters": {"n_estimators": 20, "class_weight": "balanced"}},
        seed=0, feature_names=[], input_dim=X_train.shape[1],
    )
    unweighted = GradientBoostingModel(
        hyperparameters={"parameters": {"n_estimators": 20, "class_weight": None}},
        seed=0, feature_names=[], input_dim=X_train.shape[1],
    )
    balanced.fit(X_train, y_train, X_valid, y_valid)
    unweighted.fit(X_train, y_train, X_valid, y_valid)

    # con pesos distintos por clase, el arbol resultante no puede ser identico al no ponderado
    assert not np.allclose(balanced.predict_proba(X_valid), unweighted.predict_proba(X_valid))
