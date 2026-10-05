"""Funciones puras de grafico (sin estado): PR/ROC, reliability diagram, comparacion de runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from evaluation.metrics import bin_probabilities


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, ax: Axes | None = None) -> Axes:
    ax = ax or plt.gca()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva PR")
    return ax


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, ax: Axes | None = None) -> Axes:
    ax = ax or plt.gca()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    return ax


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    ax = ax or plt.gca()
    bin_ids = bin_probabilities(y_prob, n_bins)
    mean_pred, mean_true = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        mean_true.append(float(y_true[mask].mean()))
    ax.plot(mean_pred, mean_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Probabilidad predicha (bin)")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Reliability diagram")
    if label:
        ax.legend()
    return ax


def plot_candidate_comparison(comparison_df: pd.DataFrame, ax: Axes | None = None) -> Axes:
    ax = ax or plt.gca()
    comparison_df.set_index("run_id")["valid_pr_auc"].plot(kind="bar", ax=ax)
    ax.set_ylabel("PR-AUC (VALID)")
    ax.set_title("Comparacion de candidatos")
    return ax


def save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path
