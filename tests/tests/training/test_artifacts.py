from __future__ import annotations

import numpy as np

from models.versions.v1.sklearn import DummyModel
from training.artifacts import save_run_artifacts
from training.callbacks import EpochMetricsCallback
from training.history import EpochHistory


def test_save_run_artifacts_produces_expected_files_and_reloadable_model(tmp_path, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = DummyModel(hyperparameters={}, seed=0, feature_names=["f0", "f1"], input_dim=X_train.shape[1])
    callback = EpochMetricsCallback(thresholds=[0.5], top_k_percent=[10])

    def on_epoch_end(epoch, scores_by_split, extra):
        return callback.on_epoch_end(epoch, scores_by_split, extra)

    history: EpochHistory = model.fit(X_train, y_train, X_valid, y_valid, on_epoch_end=on_epoch_end)

    preprocessor_src = tmp_path / "fake_preprocessor.pkl"
    preprocessor_src.write_bytes(b"fake-pickle-bytes")

    run_dir = tmp_path / "artifacts" / "runs" / "run-1"
    save_run_artifacts(
        run_dir, model, preprocessor_src, feature_names=["f0", "f1"],
        metadata={"run_id": "run-1", "model": "dummy"}, history=history, callback=callback,
    )

    assert (run_dir / "model" / "model.pkl").exists()
    assert (run_dir / "model" / "preprocessor.pkl").read_bytes() == b"fake-pickle-bytes"
    assert (run_dir / "model" / "feature_contract.json").exists()
    assert (run_dir / "model" / "threshold.json").exists()
    assert (run_dir / "model" / "metadata.json").exists()
    assert (run_dir / "metrics" / "epoch_metrics.json").exists()
    assert (run_dir / "metrics" / "epoch_metrics.parquet").exists()
    assert (run_dir / "predictions" / "train_scores_by_epoch.npz").exists()
    assert (run_dir / "predictions" / "valid_scores_by_epoch.npz").exists()

    from models.versions.v1.sklearn import DummyModel as ReloadedDummy

    reloaded = ReloadedDummy.load(run_dir / "model" / "model.pkl")
    np.testing.assert_allclose(reloaded.predict_proba(X_valid), model.predict_proba(X_valid))
