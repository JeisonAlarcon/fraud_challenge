"""Fixtures sinteticas compartidas por toda la suite (sin datos reales)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml


def make_synthetic_df(n: int = 3000, seed: int = 0, p_fraud: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-03-08 00:00:00")
    offsets_minutes = np.sort(rng.integers(0, 45 * 24 * 60, size=n))
    dates = start + pd.to_timedelta(offsets_minutes, unit="m")

    df = pd.DataFrame(
        {
            "a": rng.integers(0, 10, n),
            "b": rng.normal(50000, 10000, n),
            "c": rng.normal(50000, 10000, n),
            "d": rng.integers(0, 100, n),
            "e": rng.normal(0, 1, n),
            "f": rng.integers(0, 30, n),
            "g": rng.choice([f"CAT{i}" for i in range(5)], n),
            "h": rng.integers(0, 2, n),
            "j": rng.choice([f"cat_{i}" for i in range(20)], n),
            "k": np.arange(n).astype(float),
            "l": rng.integers(0, 5000, n),
            "m": rng.integers(0, 2000, n),
            "n": rng.integers(0, 2, n),
            "o": rng.choice(["Y", "N", None], n, p=[0.1, 0.1, 0.8]),
            "p": rng.choice(["Y", "N"], n),
            "fecha": dates,
            "monto": rng.exponential(50, n),
            "score": rng.integers(0, 100, n),
            "fraude": rng.choice([0, 1], n, p=[1 - p_fraud, p_fraud]),
        }
    )
    for col, frac in [("b", 0.05), ("c", 0.05), ("d", 0.01), ("m", 0.01)]:
        idx = rng.choice(n, size=max(1, int(n * frac)), replace=False)
        df.loc[idx, col] = np.nan

    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 1 duplicado exacto a proposito
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


def make_synthetic_tabular_dataset(n: int = 200, d: int = 12, seed: int = 0, p_pos: float = 0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    weights = rng.normal(size=d)
    logits = X @ weights
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < np.clip(probs, 0, p_pos * 2)).astype(np.int64)
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n:
        y[0] = 0
    split = int(n * 0.7)
    return X[:split], y[:split], X[split:], y[split:]


@pytest.fixture
def synthetic_tabular_dataset():
    return make_synthetic_tabular_dataset()


@pytest.fixture
def data_version_yaml_text() -> str:
    return yaml.safe_dump(
        {
            "version": "v1",
            "source": {"filename": "fraud_dataset_2026.csv"},
            "target": {"column": "fraude", "allowed_values": [0, 1]},
            "timestamp": {"column": "fecha", "format": "%Y-%m-%d %H:%M:%S"},
            "columns": {
                "expected": [
                    "a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p",
                    "fecha", "monto", "score", "fraude",
                ],
                "dtypes": {
                    "a": "int64", "b": "float64", "c": "float64", "d": "int64", "e": "float64",
                    "f": "int64", "g": "category", "h": "int64", "j": "category", "k": "float64",
                    "l": "int64", "m": "int64", "n": "int64", "o": "category", "p": "category",
                    "fecha": "datetime", "monto": "float64", "score": "int64", "fraude": "int64",
                },
            },
            "validation_rules": {
                "duplicates": {"check": True, "subset": None},
                "nulls": {"max_null_fraction": {"o": 0.90, "b": 0.12, "c": 0.12, "d": 0.05, "m": 0.05}},
                "ranges": {
                    "monto": {"min": 0},
                    "score": {"min": 0, "max": 100},
                    "fraude": {"values": [0, 1]},
                },
            },
            "split": {
                "strategy": "temporal",
                "order_by": "fecha",
                "validation_days": 7,
                "test_days": 7,
                "entity_column": None,
            },
            "manifest": {"clean_parquet_name": "fraud_clean_v1.parquet"},
        }
    )


@pytest.fixture
def feature_version_yaml_text() -> str:
    return yaml.safe_dump(
        {
            "target": "fraude",
            "timestamp": "fecha",
            "features": [
                {"name": "a", "enabled": True, "role": "numeric", "availability": "pre_decision",
                 "transforms": ["median_imputer", "clip_quantiles", "robust_scaler"]},
                {"name": "b", "enabled": True, "role": "numeric", "availability": "pre_decision",
                 "transforms": ["missing_indicator", "median_imputer", "clip_quantiles", "robust_scaler"]},
                {"name": "g", "enabled": True, "role": "categorical_low", "availability": "pre_decision",
                 "transforms": ["missing_token", "rare_category", "one_hot"]},
                {"name": "j", "enabled": True, "role": "categorical_high", "availability": "pre_decision",
                 "transforms": ["missing_token", "target_encoder_oof", "standard_scaler"]},
                {"name": "o", "enabled": True, "role": "binary", "availability": "pre_decision",
                 "transforms": ["missing_as_category", "explicit_binary_map"]},
                {"name": "p", "enabled": True, "role": "binary", "availability": "pre_decision",
                 "transforms": ["explicit_binary_map"]},
                {"name": "k", "enabled": False, "role": "id_like", "availability": "pre_decision",
                 "reason": "cardinalidad unica", "transforms": ["drop"]},
                {"name": "monto", "enabled": True, "role": "numeric", "availability": "pre_decision",
                 "transforms": ["median_imputer", "clip_quantiles", "log1p", "robust_scaler"]},
                {"name": "fecha", "enabled": True, "role": "datetime", "availability": "pre_decision",
                 "transforms": ["hour", "day_of_week", "is_weekend", "cyclic_hour", "cyclic_day"]},
            ],
        }
    )
