"""Contrato de datos: valida el CSV crudo contra `configs/data_versions/<version>.yml`.

Las anomalias se reportan (violations/warnings); nunca se descartan filas en silencio aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel


class SourceConfig(BaseModel):
    filename: str


class TargetConfig(BaseModel):
    column: str
    allowed_values: list[int]


class TimestampConfig(BaseModel):
    column: str
    format: str


class ColumnsConfig(BaseModel):
    expected: list[str]
    dtypes: dict[str, str]


class NullRules(BaseModel):
    max_null_fraction: dict[str, float]


class RangeRule(BaseModel):
    min: float | None = None
    max: float | None = None
    values: list[int] | None = None


class DuplicateRule(BaseModel):
    check: bool = True
    subset: list[str] | None = None


class ValidationRulesConfig(BaseModel):
    duplicates: DuplicateRule
    nulls: NullRules
    ranges: dict[str, RangeRule]


class SplitConfig(BaseModel):
    strategy: str
    order_by: str
    validation_days: int
    test_days: int
    entity_column: str | None = None


class ManifestConfig(BaseModel):
    clean_parquet_name: str


class DataVersionConfig(BaseModel):
    version: str
    source: SourceConfig
    target: TargetConfig
    timestamp: TimestampConfig
    columns: ColumnsConfig
    validation_rules: ValidationRulesConfig
    split: SplitConfig
    manifest: ManifestConfig

    @classmethod
    def from_yaml(cls, path: Path) -> DataVersionConfig:
        raw = yaml.safe_load(path.read_text())
        return cls.model_validate(raw)


@dataclass
class ContractReport:
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class DataContractError(Exception):
    def __init__(self, report: ContractReport):
        self.report = report
        message = "Contrato de datos violado:\n" + "\n".join(f"- {v}" for v in report.violations)
        super().__init__(message)


_DTYPE_TO_PANDAS = {
    "int64": "Int64",
    "float64": "float64",
    "category": "category",
    "datetime": "datetime64[ns]",
}


class DataContract:
    def __init__(self, config: DataVersionConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path) -> DataContract:
        return cls(DataVersionConfig.from_yaml(path))

    def validate(self, df: pd.DataFrame) -> ContractReport:
        violations: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {"n_rows": len(df), "n_cols": df.shape[1]}

        expected = self.config.columns.expected
        missing_cols = [c for c in expected if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in expected]
        if missing_cols:
            violations.append(f"Columnas esperadas faltantes: {missing_cols}")
        if extra_cols:
            warnings.append(f"Columnas no esperadas presentes: {extra_cols}")

        # Duplicados: solo se reportan, DatasetBuilder decide si se descartan.
        dup_rule = self.config.validation_rules.duplicates
        if dup_rule.check:
            dup_count = int(df.duplicated(subset=dup_rule.subset).sum())
            stats["duplicate_rows"] = dup_count
            if dup_count > 0:
                warnings.append(f"{dup_count} filas duplicadas detectadas")

        # Nulos: tolerancia configurada -> warning; sin tolerancia configurada -> violation.
        max_null_fraction = self.config.validation_rules.nulls.max_null_fraction
        null_fractions: dict[str, float] = {}
        for col in expected:
            if col not in df.columns:
                continue
            frac = float(df[col].isna().mean())
            if frac <= 0:
                continue
            null_fractions[col] = frac
            if col in max_null_fraction:
                if frac > max_null_fraction[col]:
                    violations.append(
                        f"Columna '{col}': fraccion de nulos {frac:.4f} excede tolerancia "
                        f"{max_null_fraction[col]:.4f}"
                    )
                else:
                    warnings.append(
                        f"Columna '{col}': fraccion de nulos {frac:.4f} dentro de tolerancia"
                    )
            else:
                violations.append(
                    f"Columna '{col}' tiene {frac:.4f} de nulos sin tolerancia configurada "
                    "(anomalia no esperada)"
                )
        stats["null_fractions"] = null_fractions

        # Target: valores permitidos.
        target_col = self.config.target.column
        if target_col in df.columns:
            allowed = set(self.config.target.allowed_values)
            observed = set(
                pd.to_numeric(df[target_col], errors="coerce").dropna().unique().tolist()
            )
            invalid = {int(v) for v in observed if int(v) not in allowed}
            if invalid or df[target_col].isna().any():
                violations.append(
                    f"Columna target '{target_col}' tiene valores fuera de {sorted(allowed)}"
                )

        # Rangos numericos.
        for col, rule in self.config.validation_rules.ranges.items():
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            if rule.values is not None:
                observed = set(numeric.dropna().unique().tolist())
                invalid = {v for v in observed if int(v) not in rule.values}
                if invalid:
                    violations.append(f"Columna '{col}' tiene valores fuera de {rule.values}")
                continue
            if rule.min is not None and (numeric < rule.min).any():
                violations.append(f"Columna '{col}' tiene valores por debajo de min={rule.min}")
            if rule.max is not None and (numeric > rule.max).any():
                violations.append(f"Columna '{col}' tiene valores por encima de max={rule.max}")

        # Timestamp parseable.
        ts_col = self.config.timestamp.column
        if ts_col in df.columns:
            parsed = pd.to_datetime(
                df[ts_col], format=self.config.timestamp.format, errors="coerce"
            )
            n_bad = int(parsed.isna().sum() - df[ts_col].isna().sum())
            if n_bad > 0:
                violations.append(
                    f"{n_bad} valores de '{ts_col}' no parseables con el formato configurado"
                )

        return ContractReport(
            is_valid=len(violations) == 0, violations=violations, warnings=warnings, stats=stats
        )

    def validate_or_raise(self, df: pd.DataFrame) -> ContractReport:
        report = self.validate(df)
        if not report.is_valid:
            raise DataContractError(report)
        return report

    def coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, dtype_name in self.config.columns.dtypes.items():
            if col not in df.columns:
                continue
            if dtype_name == "datetime":
                df[col] = pd.to_datetime(
                    df[col], format=self.config.timestamp.format, errors="coerce"
                )
            elif dtype_name in ("int64", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if dtype_name == "int64":
                    df[col] = df[col].astype("Int64")
            elif dtype_name == "category":
                df[col] = df[col].astype("category")
        return df
