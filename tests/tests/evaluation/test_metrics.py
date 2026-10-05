from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import BinaryClassificationMetrics


def test_metrics_match_hand_computed_values():
    y_true = np.array([0, 0, 0, 0, 1])
    y_score = np.array([0.2, 0.9, 0.1, 0.6, 0.4])

    result = BinaryClassificationMetrics.compute(
        y_true, y_score, thresholds=[0.5], top_k_percent=[20, 100], calibration_bins=5
    )

    assert result["average_precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["roc_auc"] == pytest.approx(0.5, abs=1e-6)
    assert result["gini"] == pytest.approx(0.0, abs=1e-6)

    # umbral 0.5 (== y_pred_default): tp=0, fp=2 (idx1,idx3), tn=2 (idx0,idx2), fn=1 (idx4)
    assert result["tp"] == 0
    assert result["fp"] == 2
    assert result["tn"] == 2
    assert result["fn"] == 1

    by_t = result["by_threshold"]["0.5"]
    assert by_t["tp"] == 0 and by_t["fp"] == 2 and by_t["tn"] == 2 and by_t["fn"] == 1
    assert by_t["precision"] == pytest.approx(0.0)
    assert by_t["recall"] == pytest.approx(0.0)

    # top 20% (1 fila) es el score mas alto (idx1, y=0) -> precision@20%=0, recall@20%=0
    top20 = result["by_top_k"]["20"]
    assert top20["precision_at_k"] == pytest.approx(0.0)
    assert top20["recall_at_k"] == pytest.approx(0.0)

    # top 100% (todas las filas) -> recall@100% siempre 1.0 (captura todos los positivos)
    top100 = result["by_top_k"]["100"]
    assert top100["recall_at_k"] == pytest.approx(1.0)
    assert top100["precision_at_k"] == pytest.approx(y_true.mean())


def test_perfect_separation_gives_pr_auc_and_roc_auc_of_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = BinaryClassificationMetrics.compute(y_true, y_score, thresholds=[0.5], top_k_percent=[50])
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["by_top_k"]["50"]["precision_at_k"] == pytest.approx(1.0)


def test_single_class_returns_nan_for_undefined_metrics():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.4, 0.2, 0.3])
    result = BinaryClassificationMetrics.compute(y_true, y_score, thresholds=[0.5], top_k_percent=[50])
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["average_precision"])
