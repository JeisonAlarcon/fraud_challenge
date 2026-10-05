"""Persistencia de artefactos de una corrida en `artifacts/runs/<run_id>/`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from models.versions.v1.base import BaseModel
from training.callbacks import EpochMetricsCallback
from training.history import EpochHistory


def save_run_artifacts(
    run_dir: Path,
    model: BaseModel,
    preprocessor_src: Path,
    feature_names: list[str],
    metadata: dict,
    history: EpochHistory,
    callback: EpochMetricsCallback,
) -> Path:
    run_dir = Path(run_dir)
    model_dir = run_dir / "model"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    for d in (model_dir, metrics_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_filename = "model.pt" if model.framework == "torch" else "model.pkl"
    model.save(model_dir / model_filename)
    shutil.copy(preprocessor_src, model_dir / "preprocessor.pkl")

    feature_contract = {"feature_names": feature_names, "input_dim": len(feature_names)}
    (model_dir / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2))
    (model_dir / "threshold.json").write_text(
        json.dumps({"threshold": None, "policy": "not_yet_calibrated"}, indent=2)
    )
    (model_dir / "metadata.json").write_text(
        json.dumps({**metadata, "model_filename": model_filename}, indent=2, default=str)
    )

    history.to_json(metrics_dir / "epoch_metrics.json")
    history.to_parquet(metrics_dir / "epoch_metrics.parquet")

    callback.save_scores_npz(
        predictions_dir / "train_scores_by_epoch.npz",
        predictions_dir / "valid_scores_by_epoch.npz",
    )
    return run_dir
