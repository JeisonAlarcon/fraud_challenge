"""Ranking/comparacion de runs ya persistidos (nunca reentrena). PR-AUC de validacion elige
al candidato; bootstrap/estabilidad temporal/soporte por segmento/latencia son verificaciones
adicionales que acompanan la decision (para el Model Card), no criterios de re-seleccion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass
class RunSummary:
    run_id: str
    model: str
    model_version: str
    hyperparameters_version: str
    feature_version: str
    valid_pr_auc: float
    valid_roc_auc: float
    valid_brier: float


def load_run_summaries(runs_dir: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return summaries
    for run_dir in sorted(runs_dir.iterdir()):
        metadata_path = run_dir / "model" / "metadata.json"
        metrics_path = run_dir / "metrics" / "epoch_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        epoch_metrics = json.loads(metrics_path.read_text())

        best_val = -float("inf")
        best_metrics: dict[str, Any] = {}
        for _, epoch_data in epoch_metrics.items():
            valid_m = epoch_data.get("valid", {})
            ap = valid_m.get("average_precision")
            if ap is not None and ap > best_val:
                best_val = ap
                best_metrics = valid_m
        if not best_metrics:
            continue

        summaries.append(
            RunSummary(
                run_id=metadata["run_id"],
                model=metadata["model"],
                model_version=metadata["model_version"],
                hyperparameters_version=metadata["hyperparameters_version"],
                feature_version=metadata["feature_version"],
                valid_pr_auc=best_val,
                valid_roc_auc=best_metrics.get("roc_auc", float("nan")),
                valid_brier=best_metrics.get("brier", float("nan")),
            )
        )
    return summaries


def bootstrap_pr_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt.tolist())) < 2:
            continue
        scores.append(average_precision_score(yt, ys))
    if not scores:
        return float("nan"), float("nan")
    arr = np.array(scores)
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def check_temporal_stability(
    y_true: np.ndarray, y_score: np.ndarray, timestamps: pd.Series, window: str = "7D"
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"y_true": y_true, "y_score": y_score}, index=pd.to_datetime(timestamps)
    ).sort_index()

    def _pr_auc(group: pd.DataFrame) -> float:
        if group["y_true"].nunique() < 2:
            return float("nan")
        return float(average_precision_score(group["y_true"], group["y_score"]))

    grouped = df.resample(window).apply(_pr_auc)
    return grouped.rename("pr_auc").to_frame()


def check_segment_support(
    df: pd.DataFrame, segment_cols: list[str], min_support: int = 30
) -> pd.DataFrame:
    rows = []
    for col in segment_cols:
        if col not in df.columns:
            continue
        for value, count in df[col].value_counts().items():
            rows.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "sufficient_support": bool(count >= min_support),
                }
            )
    return pd.DataFrame(rows)


def measure_latency(predict_fn: Callable[[Any], Any], X_sample: Any, n: int = 200) -> dict:
    durations_ms = []
    limit = min(n, X_sample.shape[0])
    for i in range(limit):
        row = X_sample[i : i + 1]
        start = time.perf_counter()
        predict_fn(row)
        durations_ms.append((time.perf_counter() - start) * 1000)
    arr = np.array(durations_ms)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


@dataclass
class WinnerSelection:
    winner_run_id: str
    comparison_table: pd.DataFrame
    checks_passed: dict[str, bool]
    warnings: list[str]


def select_winner(
    runs_dir: Path, min_segment_support: int = 30, override_run_id: str | None = None
) -> WinnerSelection:
    summaries = load_run_summaries(runs_dir)
    if not summaries:
        raise ValueError(f"No hay runs validos en {runs_dir}")

    comparison_table = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(
        "valid_pr_auc", ascending=False
    )
    warnings: list[str] = []
    if override_run_id is not None:
        if override_run_id not in comparison_table["run_id"].values:
            raise ValueError(f"override_run_id '{override_run_id}' no encontrado entre los runs")
        winner_run_id = override_run_id
        if comparison_table.iloc[0]["run_id"] != override_run_id:
            warnings.append(
                f"override_run_id='{override_run_id}' no es el de mayor PR-AUC de validacion."
            )
    else:
        winner_run_id = str(comparison_table.iloc[0]["run_id"])

    return WinnerSelection(
        winner_run_id=winner_run_id,
        comparison_table=comparison_table,
        checks_passed={"has_candidates": True},
        warnings=warnings,
    )
