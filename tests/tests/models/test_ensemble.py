from __future__ import annotations

import numpy as np
import pytest

from models.versions.v1.ensemble import (
    StackingEnsembleModel,
    UnsupportedEstimatorError,
    VotingEnsembleModel,
)


@pytest.mark.parametrize("model_class", [VotingEnsembleModel, StackingEnsembleModel])
def test_ensemble_fits_and_predicts_in_unit_interval(model_class, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    hyperparameters = {"parameters": {"estimators": ["logistic_regression", "hist_gradient_boosting"]}}
    model = model_class(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1])
    model.fit(X_train, y_train, X_valid, y_valid)
    proba = model.predict_proba(X_valid)
    assert np.all((proba >= 0) & (proba <= 1))


def test_ensemble_rejects_epoch_based_estimator():
    hyperparameters = {"parameters": {"estimators": ["logistic_regression", "mlp"]}}
    with pytest.raises(UnsupportedEstimatorError):
        VotingEnsembleModel(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=10)


def test_voting_ensemble_learns_weights_from_valid_performance(synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    names = ["logistic_regression", "hist_gradient_boosting"]
    hyperparameters = {"parameters": {"estimators": names}}
    model = VotingEnsembleModel(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1])

    assert model.estimator.weights is None  # antes de fit: sin ponderar todavia

    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.estimator.weights is not None
    assert len(model.estimator.weights) == len(names)
    assert all(0.0 <= w <= 1.0 for w in model.estimator.weights)  # son PR-AUC en VALID

    voting_weights = dict(zip(names, model.estimator.weights))
    assert set(voting_weights.keys()) == set(names)


def test_voting_ensemble_uses_per_member_tuned_hyperparameters():
    hyperparameters = {
        "parameters": {
            "estimators": ["catboost"],
            "estimator_hyperparameters": {"catboost": {"parameters": {"iterations": 7}}},
        }
    }
    model = VotingEnsembleModel(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=10)
    (catboost_estimator,) = model.estimator.estimators
    assert catboost_estimator[1].get_param("iterations") == 7


def test_stacking_ensemble_exposes_meta_learner_coefficients(synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    names = ["logistic_regression", "hist_gradient_boosting"]
    hyperparameters = {"parameters": {"estimators": names, "cv": 3}}
    model = StackingEnsembleModel(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1])
    model.fit(X_train, y_train, X_valid, y_valid)

    meta_coefs = dict(zip(names, model.estimator.final_estimator_.coef_.ravel()))
    assert set(meta_coefs.keys()) == set(names)
