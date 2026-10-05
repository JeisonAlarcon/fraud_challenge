"""Unica funcion de metricas para todos los modelos (sklearn/CatBoost/torch)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def bin_probabilities(y_score: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins equi-anchos en [0,1]; reutilizado por ECE, reliability table y reliability diagram."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(y_score, bin_edges[1:-1]), 0, n_bins - 1)


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int) -> float:
    bin_ids = bin_probabilities(y_score, n_bins)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        weight = mask.sum() / n
        ece += weight * abs(float(y_score[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def economic_profit_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    legitimate_gain_rate: float = 0.25,
    fraud_loss_rate: float = 1.0,
) -> dict:
    """Calcula ganancia monetaria usando el threshold de decisión.

    `y_score >= threshold` representa rechazo/bloqueo; por eso una operación se aprueba
    cuando `y_score < threshold`. La selección del threshold debe hacerse en VALID y la
    misma función se reutiliza para evaluar TEST después de congelarlo.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if not (len(y_true) == len(y_score) == len(amounts)):
        raise ValueError("y_true, y_score y amounts deben tener la misma longitud")
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("amounts debe contener valores finitos no negativos")
    if legitimate_gain_rate < 0 or fraud_loss_rate < 0:
        raise ValueError("las tasas económicas deben ser no negativas")

    approved = y_score < threshold
    per_transaction = amounts * np.where(y_true == 1, -fraud_loss_rate, legitimate_gain_rate)
    profit = per_transaction * approved
    approve_all_profit = float(per_transaction.sum())
    profit_total = float(profit.sum())
    return {
        "threshold": float(threshold),
        "profit_total": profit_total,
        "profit_per_transaction": float(profit.mean()) if len(profit) else float("nan"),
        "approved_count": int(approved.sum()),
        "approved_rate": float(approved.mean()) if len(approved) else float("nan"),
        "approved_amount": float(amounts[approved].sum()),
        "fraud_approved_amount": float(amounts[approved & (y_true == 1)].sum()),
        "legitimate_approved_amount": float(amounts[approved & (y_true == 0)].sum()),
        "approve_all_profit": approve_all_profit,
        "reject_all_profit": 0.0,
        "profit_uplift_vs_approve_all": profit_total - approve_all_profit,
    }


class BinaryClassificationMetrics:
    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: list[float],
        top_k_percent: list[float],
        calibration_bins: int = 10,
        amounts: np.ndarray | None = None,
        economic_threshold: float | None = None,
        legitimate_gain_rate: float = 0.25,
        fraud_loss_rate: float = 1.0,
    ) -> dict:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        if amounts is not None:
            amounts = np.asarray(amounts, dtype=float)
            if len(amounts) != len(y_true):
                raise ValueError("amounts y y_true deben tener la misma longitud")
        n = len(y_true)
        has_both_classes = len(set(y_true.tolist())) > 1

        result: dict = {}
        result["average_precision"] = (
            float(average_precision_score(y_true, y_score)) if has_both_classes else float("nan")
        )
        if has_both_classes:
            roc_auc = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ks = float(np.max(tpr - fpr))
        else:
            roc_auc, ks = float("nan"), float("nan")
        result["roc_auc"] = roc_auc
        result["gini"] = 2 * roc_auc - 1 if not np.isnan(roc_auc) else float("nan")
        result["ks"] = ks

        eps = 1e-7
        y_score_clipped = np.clip(y_score, eps, 1 - eps)
        result["log_loss"] = float(log_loss(y_true, y_score_clipped, labels=[0, 1]))
        result["brier"] = float(brier_score_loss(y_true, y_score))
        result["ece"] = _expected_calibration_error(y_true, y_score, calibration_bins)

        y_pred_default = (y_score >= 0.5).astype(int)
        result["accuracy"] = float(accuracy_score(y_true, y_pred_default))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred_default))
        result["mcc"] = (
            float(matthews_corrcoef(y_true, y_pred_default)) if has_both_classes else float("nan")
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default, labels=[0, 1]).ravel()
        result["tp"], result["fp"], result["tn"], result["fn"] = int(tp), int(fp), int(tn), int(fn)

        by_threshold: dict[str, dict] = {}
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], average=None, zero_division=0
            )
            specificity = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else float("nan")
            fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else float("nan")
            fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else float("nan")
            f_beta = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
            threshold_metrics = {
                "tp": int(tp_t),
                "fp": int(fp_t),
                "tn": int(tn_t),
                "fn": int(fn_t),
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "f_beta": f_beta,
                "specificity": float(specificity),
                "fpr": float(fpr_t),
                "fnr": float(fnr_t),
            }
            if amounts is not None:
                threshold_metrics.update(
                    economic_profit_metrics(
                        y_true,
                        y_score,
                        amounts,
                        float(t),
                        legitimate_gain_rate=legitimate_gain_rate,
                        fraud_loss_rate=fraud_loss_rate,
                    )
                )
            by_threshold[str(t)] = threshold_metrics
        result["by_threshold"] = by_threshold

        if amounts is not None and economic_threshold is not None:
            result["economic"] = economic_profit_metrics(
                y_true,
                y_score,
                amounts,
                economic_threshold,
                legitimate_gain_rate=legitimate_gain_rate,
                fraud_loss_rate=fraud_loss_rate,
            )

        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        total_positives = max(int(y_true.sum()), 1)
        prevalence = float(y_true.mean()) if n > 0 else float("nan")
        by_top_k: dict[str, dict] = {}
        for k in top_k_percent:
            k_n = max(1, int(round(n * k / 100.0)))
            top_labels = y_sorted[:k_n]
            precision_at_k = float(top_labels.mean())
            recall_at_k = float(top_labels.sum() / total_positives)  # == capture_rate_at_k
            lift_at_k = float(precision_at_k / prevalence) if prevalence > 0 else float("nan")
            by_top_k[str(k)] = {
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "capture_rate_at_k": recall_at_k,
                "lift_at_k": lift_at_k,
            }
        result["by_top_k"] = by_top_k
        return result
