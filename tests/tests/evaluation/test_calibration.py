from __future__ import annotations

import numpy as np

from evaluation.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    compute_calibration_metrics,
)


def _separable_scores(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = y * 2.0 + rng.normal(scale=0.5, size=n)  # separable, correlacionado con y
    return raw, y


def test_platt_scaler_is_monotonic_in_raw_score():
    raw, y = _separable_scores()
    calibrator = PlattScaler().fit(raw, y)
    order = np.argsort(raw)
    probs_sorted = calibrator.predict_proba(raw[order])
    assert np.all(np.diff(probs_sorted) >= -1e-9)


def test_isotonic_calibrator_is_monotonic_and_bounded():
    raw, y = _separable_scores()
    calibrator = IsotonicCalibrator().fit(raw, y)
    order = np.argsort(raw)
    probs_sorted = calibrator.predict_proba(raw[order])
    assert np.all(np.diff(probs_sorted) >= -1e-9)
    assert np.all((probs_sorted >= 0) & (probs_sorted <= 1))


def test_temperature_scaler_preserves_ranking_of_logits():
    logits, y = _separable_scores()
    calibrator = TemperatureScaler().fit(logits, y)
    order = np.argsort(logits)
    probs_sorted = calibrator.predict_proba(logits[order])
    assert np.all(np.diff(probs_sorted) >= -1e-9)


def test_compute_calibration_metrics_low_error_when_well_calibrated():
    rng = np.random.default_rng(0)
    n = 2000
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)  # por construccion, bien calibrado
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)
    assert metrics["ece"] < 0.05
    assert metrics["brier"] < 0.3
    assert "reliability_table" in metrics
