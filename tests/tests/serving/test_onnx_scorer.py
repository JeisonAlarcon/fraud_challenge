from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import onnxruntime as ort
import pytest

from evaluation.calibration import PlattScaler
from models.versions.v1.sklearn import LogisticRegressionModel
from serving.contract import (
    FeatureColumnSpec,
    FeatureContract,
    ThresholdContract,
    ThresholdPolicy,
)
from serving.onnx_exporter import OnnxCalibratorExporter, OnnxModelExporter
from serving.onnx_scorer import OnnxScorer, ScoringError


class _PassthroughPreprocessor:
    """Toma 'monto' y 'score' crudos y los devuelve como matriz float32 (sin transformar)."""

    def transform(self, df):
        return df[["monto", "score"]].to_numpy(dtype=np.float32)


def _build_promoted_bundle(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    model = LogisticRegressionModel(hyperparameters={}, seed=0, feature_names=["monto", "score"], input_dim=2)
    model.fit(X[:200], y[:200], X[200:], y[200:])

    promoted_dir = tmp_path / "promoted"
    promoted_dir.mkdir()

    OnnxModelExporter().export(model, X[:5], promoted_dir / "model_raw.onnx")
    raw_scores = model.predict_proba(X)
    calibrator = PlattScaler().fit(raw_scores, y)
    OnnxCalibratorExporter().export(calibrator, promoted_dir / "calibrator.onnx")

    joblib.dump(_PassthroughPreprocessor(), promoted_dir / "preprocessor.pkl")

    feature_contract = FeatureContract(
        feature_version="v1",
        feature_cache_id="abc123",
        target="fraude",
        timestamp_column="fecha",
        input_columns=[
            FeatureColumnSpec(name="monto", dtype="float64", role="numeric", nullable=False),
            FeatureColumnSpec(name="score", dtype="float64", role="numeric", nullable=False),
        ],
        excluded_columns=[],
        preprocessor_output_dim=2,
        preprocessor_hash="deadbeef",
        source_manifest_hash="feedface",
        generated_at=datetime.now(timezone.utc),
    )
    (promoted_dir / "feature_contract.json").write_text(feature_contract.model_dump_json(indent=2))

    threshold_contract = ThresholdContract(
        winner_run_id="run-1",
        feature_version="v1",
        model_version="v1",
        hyperparameters_version="v1",
        model="logistic_regression",
        calibration_method="platt",
        threshold=0.5,
        policy=ThresholdPolicy(type="default_argmax_f1", params={}, defined_by="default_no_business_policy"),
        valid_metrics_at_threshold={"precision": 0.5, "recall": 0.5, "f1": 0.5, "fpr": 0.1},
        reference_curves={},
        calibration_applied=True,
        selected_on="calibrated_probability",
        generated_at=datetime.now(timezone.utc),
        git_sha="unknown",
        seed=0,
    )
    (promoted_dir / "threshold.json").write_text(threshold_contract.model_dump_json(indent=2))
    return promoted_dir, X


def test_scorer_returns_probability_in_unit_interval_and_applies_threshold(tmp_path):
    promoted_dir, X = _build_promoted_bundle(tmp_path)
    scorer = OnnxScorer(promoted_dir)

    result = scorer.score({"monto": 1.5, "score": 1.5})
    assert 0.0 <= result.fraud_probability <= 1.0
    assert result.decision == (result.fraud_probability >= 0.5)
    assert result.latency_ms >= 0.0


def test_scorer_raises_on_missing_column(tmp_path):
    promoted_dir, _ = _build_promoted_bundle(tmp_path)
    scorer = OnnxScorer(promoted_dir)
    with pytest.raises(Exception):
        scorer.score({"monto": 1.5})


def test_score_matrix_works_on_preprocessed_batch(tmp_path):
    promoted_dir, X = _build_promoted_bundle(tmp_path)
    scorer = OnnxScorer(promoted_dir)
    probs = scorer.score_matrix(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))
