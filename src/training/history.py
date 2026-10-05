"""Historial de metricas por epoch (o `"final"` para modelos de un solo fit) + early stopping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class EpochHistory:
    def __init__(self) -> None:
        self.records: dict[Any, dict] = {}
        self._best_epoch: Any = None
        self._best_value: float | None = None

    def record(self, epoch: Any, metrics: dict) -> None:
        self.records[epoch] = metrics

    def _monitor_value(self, epoch: Any, monitor: str) -> float:
        split, key = monitor.split(".", 1)
        return self.records[epoch][split][key]

    @property
    def best_epoch(self) -> Any:
        return self._best_epoch

    def is_best_so_far(
        self, epoch: Any, monitor: str = "valid.average_precision", mode: str = "max"
    ) -> bool:
        if epoch not in self.records:
            return False
        value = self._monitor_value(epoch, monitor)
        is_better = (
            self._best_value is None
            or (mode == "max" and value > self._best_value)
            or (mode == "min" and value < self._best_value)
        )
        if is_better:
            self._best_value = value
            self._best_epoch = epoch
        return is_better

    def should_early_stop(
        self, patience: int, monitor: str = "valid.average_precision", mode: str = "max"
    ) -> bool:
        int_epochs = sorted(e for e in self.records if isinstance(e, int))
        if not int_epochs or not isinstance(self._best_epoch, int):
            return False
        current_epoch = int_epochs[-1]
        return (current_epoch - self._best_epoch) >= patience

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({str(k): v for k, v in self.records.items()}, indent=2, default=str)
        )

    def to_parquet(self, path: Path) -> None:
        rows = []
        for epoch, metrics in self.records.items():
            row: dict[str, Any] = {"epoch": epoch}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            continue  # by_threshold/by_top_k anidados: solo van al JSON
                        row[f"{key}.{sub_key}"] = sub_value
                else:
                    row[key] = value
            rows.append(row)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
