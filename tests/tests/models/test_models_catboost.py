from __future__ import annotations

import numpy as np

from models.versions.v1.catboost_model import CatBoostModel


def test_catboost_fit_with_eval_set_and_predict_proba(synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = CatBoostModel(
        hyperparameters={"parameters": {"iterations": 50}, "trainer": {"early_stopping_rounds": 10}},
        seed=0,
        feature_names=[],
        input_dim=X_train.shape[1],
    )
    history = model.fit(X_train, y_train, X_valid, y_valid)
    proba = model.predict_proba(X_valid)
    assert np.all((proba >= 0) & (proba <= 1))
    assert "final" in history.records


def test_catboost_save_load_roundtrip(tmp_path, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = CatBoostModel(
        hyperparameters={"parameters": {"iterations": 30}}, seed=0, feature_names=[], input_dim=X_train.shape[1]
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    proba_before = model.predict_proba(X_valid)

    path = tmp_path / "catboost_model.pkl"
    model.save(path)
    loaded = CatBoostModel.load(path)
    proba_after = loaded.predict_proba(X_valid)

    np.testing.assert_allclose(proba_before, proba_after)
