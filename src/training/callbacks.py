"""Callbacks invocados por el loop de entrenamiento: metricas persistentes + monitor en vivo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from evaluation.metrics import BinaryClassificationMetrics


class EpochMetricsCallback:
    def __init__(
        self,
        thresholds: list[float],
        top_k_percent: list[float],
        calibration_bins: int = 10,
        store_scores_by_epoch: bool = True,
    ):
        self.thresholds = thresholds
        self.top_k_percent = top_k_percent
        self.calibration_bins = calibration_bins
        self.store_scores_by_epoch = store_scores_by_epoch
        self.epoch_metrics: dict[Any, dict] = {}
        self.scores_by_epoch: dict[str, dict[Any, tuple[np.ndarray, np.ndarray]]] = {
            "train": {},
            "valid": {},
        }

    def on_epoch_end(
        self,
        epoch: Any,
        scores_by_split: dict[str, tuple[np.ndarray, np.ndarray]],
        extra: dict,
    ) -> dict:
        metrics_this_epoch: dict[str, Any] = {}
        for split_name, (y_true, y_score) in scores_by_split.items():
            metrics_this_epoch[split_name] = BinaryClassificationMetrics.compute(
                y_true, y_score, self.thresholds, self.top_k_percent, self.calibration_bins
            )

        for key, value in extra.items():
            metrics_this_epoch[key] = value

        train_m = metrics_this_epoch.get("train", {})
        valid_m = metrics_this_epoch.get("valid", {})
        if train_m and valid_m:
            metrics_this_epoch["generalization_gap_loss"] = (
                train_m["log_loss"] - valid_m["log_loss"]
            )
            metrics_this_epoch["generalization_gap_pr_auc"] = (
                train_m["average_precision"] - valid_m["average_precision"]
            )

        self.epoch_metrics[epoch] = metrics_this_epoch
        if self.store_scores_by_epoch:
            for split_name in ("train", "valid"):
                if split_name in scores_by_split:
                    self.scores_by_epoch[split_name][epoch] = scores_by_split[split_name]
        return metrics_this_epoch

    def save_scores_npz(self, train_path: Path, valid_path: Path) -> None:
        for split_name, path in (("train", train_path), ("valid", valid_path)):
            epochs = sorted(self.scores_by_epoch[split_name].keys(), key=str)
            payload = {}
            for epoch in epochs:
                y_true, y_score = self.scores_by_epoch[split_name][epoch]
                payload[f"epoch_{epoch}_y_true"] = y_true
                payload[f"epoch_{epoch}_y_score"] = y_score
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **payload)


class LiveTrainingMonitor:
    """Progreso en vivo del entrenamiento. Con `total_epochs` conocido (modelos torch) muestra
    una barra tqdm con % de avance, ETA y las metricas de la ultima epoch en el postfix. Para
    modelos de un solo fit ("final", sin epochs) no hay barra posible; solo reporta el tiempo
    total transcurrido al terminar."""

    def __init__(self, total_epochs: int | None = None, desc: str = "training"):
        self._start = time.perf_counter()
        self._pbar = tqdm(total=total_epochs, desc=desc, unit="epoch") if total_epochs else None

    def on_epoch_end(self, epoch: Any, metrics_this_epoch: dict) -> None:
        train_m = metrics_this_epoch.get("train", {})
        valid_m = metrics_this_epoch.get("valid", {})
        postfix = {}
        if "average_precision" in train_m:
            postfix["train_pr_auc"] = f"{train_m['average_precision']:.4f}"
        if "average_precision" in valid_m:
            postfix["valid_pr_auc"] = f"{valid_m['average_precision']:.4f}"

        if self._pbar is not None:
            self._pbar.set_postfix(postfix)
            self._pbar.update(1)
        else:
            elapsed = time.perf_counter() - self._start
            parts = [f"epoch={epoch}"] + [f"{k}={v}" for k, v in postfix.items()]
            print(" ".join(parts) + f" (tardo {elapsed:.1f}s)")

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
