from __future__ import annotations

import numpy as np
import pytest

from models.versions.v1.dcnv2 import DCNv2Model
from models.versions.v1.ft_transformer import FTTransformerModel
from models.versions.v1.mlp import MLPModel

CASES = [
    (MLPModel, {"parameters": {"hidden_layers": [8], "dropout": 0.1, "batch_norm": False},
                "trainer": {"batch_size": 32, "max_epochs": 1, "learning_rate": 0.01}}),
    (DCNv2Model, {"parameters": {"cross_layers": 1, "low_rank": 4, "deep_layers": [8], "dropout": 0.1},
                  "trainer": {"batch_size": 32, "max_epochs": 1, "learning_rate": 0.01}}),
    (FTTransformerModel, {"parameters": {"token_dim": 8, "n_transformer_layers": 1, "n_heads": 2,
                                          "ffn_dim": 16, "attn_dropout": 0.0},
                          "trainer": {"batch_size": 32, "max_epochs": 1, "learning_rate": 0.01}}),
]


@pytest.mark.parametrize("model_class,hyperparameters", CASES)
def test_one_cpu_epoch_runs_without_nan_and_records_history(model_class, hyperparameters, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = model_class(
        hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1]
    )
    history = model.fit(X_train, y_train, X_valid, y_valid)

    assert len(history.records) == 1
    proba = model.predict_proba(X_valid)
    assert not np.isnan(proba).any()
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("model_class,hyperparameters", CASES)
def test_save_load_roundtrip(tmp_path, model_class, hyperparameters, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = model_class(
        hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1]
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    proba_before = model.predict_proba(X_valid)

    path = tmp_path / "model.pt"
    model.save(path)
    loaded = model_class.load(path)
    proba_after = loaded.predict_proba(X_valid)

    np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


def test_lr_scheduler_reduces_learning_rate_on_plateau(synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    hyperparameters = {
        "parameters": {"hidden_layers": [4], "dropout": 0.0, "batch_norm": False},
        "trainer": {
            "batch_size": 32,
            "max_epochs": 15,
            "learning_rate": 0.1,
            "early_stopping": {"patience": 100, "mode": "max"},  # no interfiera con este test
            "lr_scheduler": {"factor": 0.5, "patience": 0, "min_lr": 1e-8},
        },
    }
    model = MLPModel(hyperparameters=hyperparameters, seed=0, feature_names=[], input_dim=X_train.shape[1])

    recorded_lrs: list[float] = []

    def on_epoch_end(epoch, scores_by_split, extra):
        recorded_lrs.append(extra["learning_rate"])
        # metrica constante: nunca "mejora", fuerza al scheduler a reducir apenas se agote su patience.
        return {"valid": {"average_precision": 0.5}}

    model.fit(X_train, y_train, X_valid, y_valid, on_epoch_end=on_epoch_end)

    assert len(recorded_lrs) == 15
    assert recorded_lrs[-1] < recorded_lrs[0]
