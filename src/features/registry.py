"""Registro de transforms declarativos: nombre (YAML) -> transformer sklearn-compatible.

Todos los transformers exponen fit/transform (BaseEstimator, TransformerMixin) para poder
encadenarse dentro de un `sklearn.pipeline.Pipeline` por feature. `TargetEncoderOOF` es la
unica pieza con `fit_transform` explicito: sklearn invoca ese metodo durante el ajuste del
Pipeline (con `y` propagado por `ColumnTransformer.fit_transform(X, y)`), lo que produce
encoding out-of-fold en TRAIN sin codigo especial en `factory.py`; `transform` (usado en
VALID/TEST) aplica el mapeo ajustado sobre el TRAIN completo.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer


class UnknownTransformError(KeyError):
    pass


def _as_series(X: Any) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    arr = np.asarray(X)
    return pd.Series(arr.reshape(-1))


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip por cuantiles de TRAIN. ponytail: si el input trae una columna indicadora
    binaria (0/1) pegada por SimpleImputer(add_indicator=True), tambien se le calculan
    cuantiles propios (q01=0, q99=1 en la practica) -> el clip es un no-op sobre ella."""

    def __init__(self, low_q: float = 0.01, high_q: float = 0.99):
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X: Any, y: Any = None) -> QuantileClipper:
        arr = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(arr, self.low_q, axis=0)
        self.high_ = np.nanquantile(arr, self.high_q, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.low_, self.high_)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> Log1pTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.log1p(np.clip(arr, a_min=0, a_max=None))


class MissingTokenFiller(BaseEstimator, TransformerMixin):
    def __init__(self, token: str = "__missing__"):
        self.token = token

    def fit(self, X: Any, y: Any = None) -> MissingTokenFiller:
        return self

    def transform(self, X: Any) -> np.ndarray:
        # ponytail: astype(object) primero, no astype(str) primero: pandas Categorical no admite
        # fillna con un valor fuera de sus categorias, y astype(str) directo convertiria los NaN
        # reales en el string literal "nan" en vez de aplicar el token.
        return _as_frame(X).astype(object).fillna(self.token).astype(str).to_numpy()


class MissingAsCategory(BaseEstimator, TransformerMixin):
    def __init__(self, token: str = "missing"):
        self.token = token

    def fit(self, X: Any, y: Any = None) -> MissingAsCategory:
        return self

    def transform(self, X: Any) -> np.ndarray:
        return _as_frame(X).astype(object).fillna(self.token).astype(str).to_numpy()


class ExplicitBinaryMapper(BaseEstimator, TransformerMixin):
    """Y->1.0, N->0.0; cualquier otro valor (incluye 'missing' si viene antes en la cadena)
    se codifica como 0.5 (neutro), en vez de descartar la fila."""

    _MAPPING = {"Y": 1.0, "N": 0.0}

    def fit(self, X: Any, y: Any = None) -> ExplicitBinaryMapper:
        return self

    def transform(self, X: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        return series.map(self._MAPPING).fillna(0.5).to_numpy().reshape(-1, 1)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq: float = 0.01, other_token: str = "__other__"):
        self.min_freq = min_freq
        self.other_token = other_token

    def fit(self, X: Any, y: Any = None) -> RareCategoryGrouper:
        series = _as_series(X).astype(str)
        freq = series.value_counts(normalize=True)
        self.keep_categories_ = set(freq[freq >= self.min_freq].index.tolist())
        return self

    def transform(self, X: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        out = series.where(series.isin(self.keep_categories_), other=self.other_token)
        return out.to_numpy().reshape(-1, 1)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> FrequencyEncoder:
        series = _as_series(X).astype(str)
        self.freq_map_ = (series.value_counts(normalize=True)).to_dict()
        return self

    def transform(self, X: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        return series.map(self.freq_map_).fillna(0.0).to_numpy().reshape(-1, 1)


class CountEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> CountEncoder:
        series = _as_series(X).astype(str)
        self.count_map_ = series.value_counts().to_dict()
        return self

    def transform(self, X: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        return series.map(self.count_map_).fillna(0).to_numpy().reshape(-1, 1)


class HashingEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, X: Any, y: Any = None) -> HashingEncoderWrapper:
        self._encoder = ce.HashingEncoder(n_components=self.n_components, return_df=True)
        df = _as_frame(X).astype(str)
        df.columns = ["col_0"]
        self._encoder.fit(df)
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = _as_frame(X).astype(str)
        df.columns = ["col_0"]
        return self._encoder.transform(df).to_numpy()


class TargetEncoderOOF(BaseEstimator, TransformerMixin):
    """Target encoding con proteccion de leakage: TRAIN se codifica out-of-fold (fit_transform,
    invocado por sklearn durante el ajuste del Pipeline); VALID/TEST usan el mapeo ajustado
    sobre TRAIN completo (transform)."""

    def __init__(self, n_splits: int = 5, smoothing: float = 10.0, seed: int = 42):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed

    def _smoothed_means(self, series: pd.Series, y: np.ndarray, global_mean: float) -> dict:
        df = pd.DataFrame({"cat": series.to_numpy(), "y": y})
        agg = df.groupby("cat")["y"].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * self.smoothing) / (
            agg["count"] + self.smoothing
        )
        return smoothed.to_dict()

    def fit(self, X: Any, y: Any = None) -> TargetEncoderOOF:
        series = _as_series(X).astype(str)
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(y_arr.mean())
        self.mapping_ = self._smoothed_means(series, y_arr, self.global_mean_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        return series.map(self.mapping_).fillna(self.global_mean_).to_numpy().reshape(-1, 1)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        series = _as_series(X).astype(str)
        y_arr = np.asarray(y, dtype=float)
        n = len(series)
        oof = np.empty(n, dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for train_idx, holdout_idx in kf.split(series):
            fold_mean = float(y_arr[train_idx].mean())
            fold_mapping = self._smoothed_means(series.iloc[train_idx], y_arr[train_idx], fold_mean)
            holdout_series = series.iloc[holdout_idx]
            oof[holdout_idx] = holdout_series.map(fold_mapping).fillna(fold_mean).to_numpy()
        # Ajusta tambien el mapeo sobre TRAIN completo, usado luego por `transform` en VALID/TEST.
        self.fit(X, y)
        return oof.reshape(-1, 1)


class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    """Una instancia por sub-feature solicitada (`kind`); se combinan en paralelo via
    `FeatureUnion` en factory.py (no son pasos secuenciales de un mismo Pipeline)."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, X: Any, y: Any = None) -> DatetimeFeaturizer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        dt = pd.to_datetime(_as_series(X))
        if self.kind == "hour":
            return dt.dt.hour.to_numpy().reshape(-1, 1).astype(float)
        if self.kind == "day_of_week":
            return dt.dt.dayofweek.to_numpy().reshape(-1, 1).astype(float)
        if self.kind == "is_weekend":
            return (dt.dt.dayofweek >= 5).to_numpy().reshape(-1, 1).astype(float)
        if self.kind == "cyclic_hour":
            radians = 2 * np.pi * dt.dt.hour.to_numpy() / 24.0
            return np.column_stack([np.sin(radians), np.cos(radians)])
        if self.kind == "cyclic_day":
            radians = 2 * np.pi * dt.dt.dayofweek.to_numpy() / 7.0
            return np.column_stack([np.sin(radians), np.cos(radians)])
        raise UnknownTransformError(f"DatetimeFeaturizer: kind desconocido '{self.kind}'")


def _unsupported_drop(**kwargs: Any) -> Any:
    raise ValueError(
        "'drop' no debe instanciarse como transformer: el feature debe quedar excluido "
        "del pipeline (enabled=false), no incluido con un paso 'drop'."
    )


TRANSFORM_REGISTRY: dict[str, Callable[..., Any]] = {
    "drop": _unsupported_drop,
    "passthrough": FunctionTransformer,
    "median_imputer": partial(SimpleImputer, strategy="median"),
    "missing_indicator": partial(
        SimpleImputer, strategy="constant", add_indicator=True, fill_value=0
    ),
    "clip_quantiles": QuantileClipper,
    "log1p": Log1pTransformer,
    "standard_scaler": StandardScaler,
    "robust_scaler": RobustScaler,
    "missing_token": MissingTokenFiller,
    "rare_category": RareCategoryGrouper,
    "one_hot": partial(OneHotEncoder, handle_unknown="ignore", sparse_output=True),
    "ordinal_encoder": partial(
        OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1
    ),
    "frequency_encoder": FrequencyEncoder,
    "count_encoder": CountEncoder,
    "hashing_encoder": HashingEncoderWrapper,
    "target_encoder_oof": TargetEncoderOOF,
    "missing_as_category": MissingAsCategory,
    "explicit_binary_map": ExplicitBinaryMapper,
    "hour": partial(DatetimeFeaturizer, kind="hour"),
    "day_of_week": partial(DatetimeFeaturizer, kind="day_of_week"),
    "is_weekend": partial(DatetimeFeaturizer, kind="is_weekend"),
    "cyclic_hour": partial(DatetimeFeaturizer, kind="cyclic_hour"),
    "cyclic_day": partial(DatetimeFeaturizer, kind="cyclic_day"),
}

DATETIME_TRANSFORM_NAMES = {"hour", "day_of_week", "is_weekend", "cyclic_hour", "cyclic_day"}


def build_transform(name: str, **kwargs: Any) -> Any:
    if name not in TRANSFORM_REGISTRY:
        raise UnknownTransformError(f"Transform desconocido: '{name}'")
    factory = TRANSFORM_REGISTRY[name]
    return factory(**kwargs)
