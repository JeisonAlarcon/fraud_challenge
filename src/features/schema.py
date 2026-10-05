"""Esquema Pydantic de `configs/feature_versions/<version>.yml`.

Una version de features produce un unico espacio de entrada comun para todos los modelos.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class FeatureRole(str, Enum):
    numeric = "numeric"
    categorical_low = "categorical_low"
    categorical_high = "categorical_high"
    binary = "binary"
    id_like = "id_like"
    datetime = "datetime"


class Availability(str, Enum):
    pre_decision = "pre_decision"
    post_decision = "post_decision"


ROLE_ALLOWED_TRANSFORMS: dict[FeatureRole, set[str]] = {
    FeatureRole.numeric: {
        "drop",
        "passthrough",
        "median_imputer",
        "missing_indicator",
        "clip_quantiles",
        "log1p",
        "standard_scaler",
        "robust_scaler",
    },
    FeatureRole.categorical_low: {
        "drop",
        "passthrough",
        "missing_token",
        "rare_category",
        "one_hot",
        "ordinal_encoder",
        "frequency_encoder",
        "count_encoder",
    },
    FeatureRole.categorical_high: {
        "drop",
        "passthrough",
        "missing_token",
        "target_encoder_oof",
        "standard_scaler",
        "frequency_encoder",
        "hashing_encoder",
        "ordinal_encoder",
    },
    FeatureRole.binary: {
        "drop",
        "passthrough",
        "missing_as_category",
        "explicit_binary_map",
    },
    FeatureRole.id_like: {"drop"},
    FeatureRole.datetime: {
        "hour",
        "day_of_week",
        "is_weekend",
        "cyclic_hour",
        "cyclic_day",
        "drop",
    },
}


class FeatureSpec(BaseModel):
    name: str
    enabled: bool
    role: FeatureRole
    availability: Availability
    reason: str | None = None
    transforms: list[str]

    @model_validator(mode="after")
    def _validate_transforms(self) -> FeatureSpec:
        if self.availability == Availability.post_decision:
            raise ValueError(
                f"Feature '{self.name}': availability 'post_decision' no permitida como input "
                "(riesgo de leakage)."
            )
        if self.role == FeatureRole.id_like and self.enabled:
            raise ValueError(f"Feature '{self.name}': role 'id_like' no puede estar enabled=true.")
        if len(self.transforms) != len(set(self.transforms)):
            raise ValueError(f"Feature '{self.name}': transforms duplicados en {self.transforms}.")
        allowed = ROLE_ALLOWED_TRANSFORMS[self.role]
        unknown = [t for t in self.transforms if t not in allowed]
        if unknown:
            raise ValueError(
                f"Feature '{self.name}' (role={self.role.value}): transforms no permitidos {unknown}. "
                f"Permitidos: {sorted(allowed)}"
            )
        return self


class FeatureVersionConfig(BaseModel):
    target: str
    timestamp: str
    features: list[FeatureSpec]

    @model_validator(mode="after")
    def _validate_features(self) -> FeatureVersionConfig:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"Nombres de features duplicados: {names}")
        if self.target in names:
            raise ValueError(f"El target '{self.target}' no puede aparecer como feature de input.")
        timestamp_specs = [f for f in self.features if f.name == self.timestamp]
        if not timestamp_specs:
            raise ValueError(
                f"La columna de timestamp '{self.timestamp}' debe tener una FeatureSpec."
            )
        if timestamp_specs[0].role != FeatureRole.datetime:
            raise ValueError(
                f"La columna de timestamp '{self.timestamp}' debe tener role='datetime'."
            )
        return self

    def enabled_features(self) -> list[FeatureSpec]:
        return [f for f in self.features if f.enabled]


def load_feature_version(path: Path, known_columns: set[str] | None = None) -> FeatureVersionConfig:
    raw = yaml.safe_load(Path(path).read_text())
    config = FeatureVersionConfig.model_validate(raw)
    if known_columns is not None:
        unknown = [f.name for f in config.features if f.name not in known_columns]
        if unknown:
            raise ValueError(
                f"Columnas desconocidas en feature_version (no estan en el dataset): {unknown}"
            )
    return config
