"""Construye el `ColumnTransformer` unico a partir de una `FeatureVersionConfig` ya validada."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from features.registry import build_transform
from features.schema import FeatureRole, FeatureSpec, FeatureVersionConfig


class FeaturePipelineFactory:
    def __init__(self, config: FeatureVersionConfig):
        self.config = config

    def _build_feature_pipeline(self, spec: FeatureSpec) -> Pipeline | FeatureUnion:
        if spec.role == FeatureRole.datetime:
            # Sub-features de fecha son independientes entre si (todas parten del mismo dato
            # crudo), no secuenciales: se combinan en paralelo, no se encadenan.
            return FeatureUnion([(name, build_transform(name)) for name in spec.transforms])

        transforms = list(spec.transforms)
        steps: list[tuple[str, object]] = []
        if "missing_indicator" in transforms and "median_imputer" in transforms:
            # ponytail: SimpleImputer(add_indicator=True) ya resuelve "imputar + marcar nulo"
            # en un solo paso stdlib; no se encadenan dos transformers separados para esto.
            transforms = [t for t in transforms if t != "missing_indicator"]
            idx = transforms.index("median_imputer")
            for name in transforms[:idx]:
                steps.append((name, build_transform(name)))
            steps.append(("median_imputer", build_transform("median_imputer", add_indicator=True)))
            for name in transforms[idx + 1 :]:
                steps.append((name, build_transform(name)))
        else:
            for name in transforms:
                steps.append((name, build_transform(name)))
        return Pipeline(steps)

    def build(self) -> ColumnTransformer:
        transformers = [
            (spec.name, self._build_feature_pipeline(spec), [spec.name])
            for spec in self.config.enabled_features()
        ]
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series) -> ColumnTransformer:
        ct = self.build()
        ct.fit(df_train, y_train)
        return ct

    def transform(self, df: pd.DataFrame, fitted: ColumnTransformer):
        return fitted.transform(df)

    def get_feature_names(self, fitted: ColumnTransformer, sample_row: pd.DataFrame) -> list[str]:
        """Nombres de columnas de salida, determinados por sondeo empirico (transformar una
        fila ya ajustada) en vez de exigir `get_feature_names_out` en cada transformer custom."""
        names: list[str] = []
        for feature_name, transformer, columns in fitted.transformers_:
            if transformer in ("drop", "passthrough"):
                continue
            output = transformer.transform(sample_row[list(columns)])
            width = output.shape[1] if hasattr(output, "shape") else 1
            last_step = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else None
            if isinstance(last_step, OneHotEncoder):
                categories = last_step.categories_[0]
                names.extend(f"{feature_name}__{c}" for c in categories)
            elif width == 1:
                names.append(feature_name)
            else:
                names.extend(f"{feature_name}_{i}" for i in range(width))
        return names
