from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from evaluation.calibration import PlattScaler
from models.versions.v1.catboost_model import CatBoostModel
from models.versions.v1.sklearn import LogisticRegressionModel
from serving.onnx_exporter import (
    OnnxCalibratorExporter,
    OnnxModelExporter,
    ParityError,
    assert_parity,
)


def test_sklearn_model_parity(tmp_path, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = LogisticRegressionModel(hyperparameters={}, seed=0, feature_names=[], input_dim=X_train.shape[1])
    model.fit(X_train, y_train, X_valid, y_valid)

    onnx_path = OnnxModelExporter().export(model, X_valid[:5], tmp_path / "model_raw.onnx")
    session = ort.InferenceSession(str(onnx_path))

    edge_cases = np.vstack([X_valid, np.zeros((1, X_valid.shape[1]), dtype=np.float32), np.full((1, X_valid.shape[1]), 5.0, dtype=np.float32)])
    report = assert_parity(model.predict_proba, session, edge_cases, atol=1e-4, rtol=1e-3)
    assert report.passed


def test_catboost_model_parity(tmp_path, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = CatBoostModel(
        hyperparameters={"parameters": {"iterations": 2}}, seed=0, feature_names=[], input_dim=X_train.shape[1]
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    onnx_path = OnnxModelExporter().export(model, X_valid[:5], tmp_path / "model_raw.onnx")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(X_valid, dtype=np.float32)})
    probability_output = next(
        output for output in outputs if isinstance(output, list) and output and isinstance(output[0], dict)
    )
    onnx_scores = np.asarray(
        [row.get(1, row.get("1")) for row in probability_output], dtype=np.float32
    )
    assert np.allclose(model.predict_proba(X_valid), onnx_scores, atol=1e-3, rtol=1e-2)


def test_parity_raises_when_mismatched(tmp_path, synthetic_tabular_dataset):
    X_train, y_train, X_valid, y_valid = synthetic_tabular_dataset
    model = LogisticRegressionModel(hyperparameters={}, seed=0, feature_names=[], input_dim=X_train.shape[1])
    model.fit(X_train, y_train, X_valid, y_valid)
    onnx_path = OnnxModelExporter().export(model, X_valid[:5], tmp_path / "model_raw.onnx")
    session = ort.InferenceSession(str(onnx_path))

    def broken_reference(X):
        return model.predict_proba(X) + 0.5  # fuerza un desajuste

    with pytest.raises(ParityError):
        assert_parity(broken_reference, session, X_valid, strict=True)


def test_platt_calibrator_onnx_reproduces_sigmoid(tmp_path):
    calibrator = PlattScaler()
    calibrator.params = {"a": 2.0, "b": -1.0}
    onnx_path = OnnxCalibratorExporter().export(calibrator, tmp_path / "calibrator.onnx")

    raw_scores = np.linspace(-3, 3, 25).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path))
    expected = calibrator.predict_proba(raw_scores)
    report = assert_parity(calibrator.predict_proba, session, raw_scores, atol=1e-5, rtol=1e-4)
    assert report.passed
    assert np.all((expected >= 0) & (expected <= 1))
