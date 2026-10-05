"""Calibradores (Platt/temperature/isotonic), metricas de calibracion, `CalibrationRunner` y
`ThresholdSelector`. El calibrador se ajusta sobre scores crudos de `model_raw.onnx` en VALID
(nunca sobre TEST)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix

from evaluation.metrics import (
    _expected_calibration_error,
    bin_probabilities,
    economic_profit_metrics,
)
from evaluation.plots import plot_reliability_diagram, save_figure
from serving.contract import ThresholdPolicy
from serving.onnx_exporter import run_onnx_raw_score


class PlattScaler:
    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> PlattScaler:
        lr = LogisticRegression()
        lr.fit(raw_scores.reshape(-1, 1), y)
        self.params = {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        a, b = self.params["a"], self.params["b"]
        return 1.0 / (1.0 + np.exp(-(a * raw_scores + b)))


class TemperatureScaler:
    def fit(self, logits: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        log_temperature = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = criterion(logits_t / temperature, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.params = {"T": float(torch.exp(log_temperature).item())}
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        temperature = self.params["T"]
        return 1.0 / (1.0 + np.exp(-(logits / temperature)))


class IsotonicCalibrator:
    def fit(
        self, raw_scores: np.ndarray, y: np.ndarray, lut_points: int = 1001
    ) -> IsotonicCalibrator:
        self.sklearn_model = IsotonicRegression(out_of_bounds="clip")
        self.sklearn_model.fit(raw_scores, y)
        self.lut_min_ = float(np.min(raw_scores))
        self.lut_max_ = float(np.max(raw_scores))
        grid = np.linspace(self.lut_min_, self.lut_max_, lut_points)
        self.lut_values_ = self.sklearn_model.predict(grid)
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.sklearn_model.predict(raw_scores)


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_prob = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_prob, y_true)

    bin_ids = bin_probabilities(y_prob, n_bins)
    reliability_table = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        reliability_table.append(
            {
                "bin": b,
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _expected_calibration_error(y_true, y_prob, n_bins),
        "slope": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "reliability_table": reliability_table,
    }


def _as_probability(raw_scores: np.ndarray, framework: str) -> np.ndarray:
    if framework == "torch":
        return 1.0 / (1.0 + np.exp(-raw_scores))
    return raw_scores


@dataclass
class CalibrationResult:
    calibrator: Any
    pre_metrics: dict
    post_metrics: dict
    reliability_fig_path: Path
    method_used: str


class CalibrationRunner:
    def __init__(
        self,
        model_raw_onnx_path: Path,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        method: str = "auto",
    ):
        self.model_raw_onnx_path = Path(model_raw_onnx_path)
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.method = method

    def run(self, output_dir: Path, framework: str) -> CalibrationResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        session = ort.InferenceSession(str(self.model_raw_onnx_path))
        raw_scores = run_onnx_raw_score(session, self.X_valid)

        pre_metrics = compute_calibration_metrics(
            self.y_valid, _as_probability(raw_scores, framework)
        )

        method = (
            self.method
            if self.method != "auto"
            else ("temperature" if framework == "torch" else "platt")
        )
        if method == "platt":
            calibrator: Any = PlattScaler().fit(raw_scores, self.y_valid)
        elif method == "temperature":
            calibrator = TemperatureScaler().fit(raw_scores, self.y_valid)
        elif method == "isotonic":
            calibrator = IsotonicCalibrator().fit(raw_scores, self.y_valid)
        else:
            raise ValueError(f"Metodo de calibracion desconocido: '{method}'")

        calibrated_probs = calibrator.predict_proba(raw_scores)
        post_metrics = compute_calibration_metrics(self.y_valid, calibrated_probs)

        fig, ax = plt.subplots()
        plot_reliability_diagram(
            self.y_valid, _as_probability(raw_scores, framework), ax=ax, label="pre-calibracion"
        )
        plot_reliability_diagram(self.y_valid, calibrated_probs, ax=ax, label="post-calibracion")
        fig_path = save_figure(fig, output_dir / "reliability_diagram.png")
        plt.close(fig)

        joblib.dump(calibrator, output_dir / "calibrator.pkl")
        return CalibrationResult(
            calibrator=calibrator,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            reliability_fig_path=fig_path,
            method_used=method,
        )


@dataclass
class ThresholdSelectionResult:
    threshold: float
    policy_used: ThresholdPolicy
    valid_metrics_at_threshold: dict
    threshold_table: pd.DataFrame


class ThresholdSelector:
    def select(
        self,
        y_true: np.ndarray,
        y_prob_calibrated: np.ndarray,
        policy: ThresholdPolicy,
        amounts: np.ndarray | None = None,
    ) -> ThresholdSelectionResult:
        y_true = np.asarray(y_true)
        y_prob_calibrated = np.asarray(y_prob_calibrated, dtype=float)
        if len(y_true) != len(y_prob_calibrated):
            raise ValueError("y_true y y_prob_calibrated deben tener la misma longitud")
        if not np.all(np.isfinite(y_prob_calibrated)):
            raise ValueError("y_prob_calibrated debe contener valores finitos")

        economic = policy.type == "economic_profit"
        if economic:
            if amounts is None:
                raise ValueError("economic_profit requiere amounts")
            amounts = np.asarray(amounts, dtype=float)
            if len(amounts) != len(y_true):
                raise ValueError("amounts y y_true deben tener la misma longitud")
            if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
                raise ValueError("amounts debe contener valores finitos no negativos")
            # Los scores observados evitan perder el máximo entre pasos de 0.01.
            grid = np.unique(np.concatenate(([0.0, 0.20, 1.0], y_prob_calibrated)))
        else:
            grid = np.linspace(0.01, 0.99, 99)

        rows = []
        for t in grid:
            y_pred = (y_prob_calibrated >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            row = {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
            if economic:
                row.update(
                    economic_profit_metrics(
                        y_true,
                        y_prob_calibrated,
                        amounts,
                        float(t),
                        legitimate_gain_rate=policy.params.get("legitimate_gain_rate", 0.25),
                        fraud_loss_rate=policy.params.get("fraud_loss_rate", 1.0),
                    )
                )
            rows.append(row)
        table = pd.DataFrame(rows)

        if policy.type == "economic_profit":
            best_row = table.loc[table["profit_total"].idxmax()]
        elif policy.type == "cost_based":
            cost_fp = policy.params.get("cost_fp", 1.0)
            cost_fn = policy.params.get("cost_fn", 1.0)
            table["expected_cost"] = table["fp"] * cost_fp + table["fn"] * cost_fn
            best_row = table.loc[table["expected_cost"].idxmin()]
        elif policy.type == "recall_at_fpr":
            max_fpr = policy.params.get("max_fpr", 0.05)
            feasible = table[table["fpr"] <= max_fpr]
            best_row = (
                feasible.loc[feasible["recall"].idxmax()] if len(feasible) else table.iloc[-1]
            )
        elif policy.type == "recall_at_precision":
            min_precision = policy.params.get("min_precision", 0.5)
            feasible = table[table["precision"] >= min_precision]
            best_row = feasible.loc[feasible["recall"].idxmax()] if len(feasible) else table.iloc[0]
        else:
            best_row = table.loc[table["f1"].idxmax()]

        threshold = float(best_row["threshold"])
        metrics_at_threshold = {
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "fpr": float(best_row["fpr"]),
        }
        if economic:
            metrics_at_threshold.update(
                {
                    key: float(value) if isinstance(value, (float, np.floating)) else int(value)
                    for key, value in best_row.items()
                    if key
                    not in {"threshold", "precision", "recall", "fpr", "f1", "tp", "fp", "tn", "fn"}
                }
            )
        return ThresholdSelectionResult(
            threshold=threshold,
            policy_used=policy,
            valid_metrics_at_threshold=metrics_at_threshold,
            threshold_table=table,
        )
