"""Contratos de serving: `feature_contract.json` (columnas esperadas) y `threshold.json`
(umbral + politica). `OnnxScorer` los usa para validar inputs antes de invocar los ONNX."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class FeatureColumnSpec(BaseModel):
    name: str
    dtype: Literal["float64", "int64", "string", "bool"]
    role: str
    nullable: bool
    valid_range: tuple[float | None, float | None] | None = None


class ThresholdPolicy(BaseModel):
    type: Literal[
        "cost_based",
        "recall_at_fpr",
        "recall_at_precision",
        "default_argmax_f1",
        "economic_profit",
    ]
    params: dict[str, float]
    defined_by: str


class FeatureContract(BaseModel):
    feature_version: str
    feature_cache_id: str
    target: str
    timestamp_column: str
    input_columns: list[FeatureColumnSpec]
    excluded_columns: list[dict]
    preprocessor_output_dim: int
    preprocessor_hash: str
    source_manifest_hash: str
    generated_at: datetime


class ThresholdContract(BaseModel):
    winner_run_id: str
    feature_version: str
    model_version: str
    hyperparameters_version: str
    model: str
    calibration_method: Literal["platt", "temperature", "isotonic"]
    threshold: float
    policy: ThresholdPolicy
    valid_metrics_at_threshold: dict
    reference_curves: dict[str, str]
    calibration_applied: bool
    selected_on: Literal["calibrated_probability"]
    generated_at: datetime
    git_sha: str
    seed: int


class ContractValidationError(ValueError):
    pass


def load_feature_contract(path: Path) -> FeatureContract:
    return FeatureContract.model_validate_json(Path(path).read_text())


def load_threshold_contract(path: Path) -> ThresholdContract:
    return ThresholdContract.model_validate_json(Path(path).read_text())


def validate_raw_features(contract: FeatureContract, raw: dict[str, Any]) -> None:
    missing = [col.name for col in contract.input_columns if col.name not in raw]
    if missing:
        raise ContractValidationError(f"Columnas faltantes en el input: {missing}")

    for col in contract.input_columns:
        value = raw[col.name]
        if value is None:
            if not col.nullable:
                raise ContractValidationError(f"Columna '{col.name}' no admite nulos")
            continue
        if col.dtype in ("float64", "int64") and not isinstance(value, (int, float)):
            raise ContractValidationError(
                f"Columna '{col.name}' esperaba numero, recibio {type(value).__name__}"
            )
        if col.valid_range is not None:
            low, high = col.valid_range
            if low is not None and value < low:
                raise ContractValidationError(
                    f"Columna '{col.name}'={value} por debajo de min={low}"
                )
            if high is not None and value > high:
                raise ContractValidationError(
                    f"Columna '{col.name}'={value} por encima de max={high}"
                )
