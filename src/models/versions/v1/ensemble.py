"""Ensambles Voting/Stacking sobre modelos `supports_epochs=False` (LogReg/HistGBM/CatBoost).

Los adapters torch (MLP/DCN-V2/FT-Transformer) no son clonables via `sklearn.clone` (ciclo de
vida manual por epoch), por lo que quedan fuera de los ensambles en v1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from models.versions.v1.base import BaseModel, SklearnLikeModel
from models.versions.v1.catboost_model import CatBoostModel
from models.versions.v1.sklearn import (
    DummyModel,
    HistGradientBoostingModel,
    LogisticRegressionModel,
)

_BASE_MODEL_BUILDERS: dict[str, type[BaseModel]] = {
    "dummy": DummyModel,
    "logistic_regression": LogisticRegressionModel,
    "hist_gradient_boosting": HistGradientBoostingModel,
    "catboost": CatBoostModel,
}


class UnsupportedEstimatorError(TypeError):
    pass


def _resolve_base_estimator(
    name: str, seed: int, member_hyperparameters: dict | None
) -> tuple[str, Any]:
    if name not in _BASE_MODEL_BUILDERS:
        # Import perezoso: solo si alguien pide un modelo torch por nombre en un ensamble.
        from models.versions.v1.dcnv2 import DCNv2Model
        from models.versions.v1.ft_transformer import FTTransformerModel
        from models.versions.v1.mlp import MLPModel

        torch_builders = {
            "mlp": MLPModel,
            "dcnv2": DCNv2Model,
            "ft_transformer": FTTransformerModel,
        }
        if name in torch_builders:
            raise UnsupportedEstimatorError(
                f"'{name}' tiene supports_epochs=True; no puede usarse como base de un ensamble v1."
            )
        raise UnsupportedEstimatorError(f"Estimador base desconocido para ensamble: '{name}'")
    wrapper = _BASE_MODEL_BUILDERS[name](
        hyperparameters=member_hyperparameters or {}, seed=seed, feature_names=[], input_dim=0
    )
    if wrapper.supports_epochs:
        raise UnsupportedEstimatorError(
            f"'{name}' tiene supports_epochs=True; no puede usarse como base de un ensamble v1."
        )
    return name, wrapper.estimator


def _base_estimators(hyperparameters: dict, seed: int) -> list[tuple[str, Any]]:
    """Construye los miembros base. `estimator_hyperparameters` (opcional, en el YAML del
    ensamble) permite que cada miembro use SU PROPIA config tuneada (p.ej. la de
    `hyperparameter_versions/v2/catboost.yml`) en vez de los defaults de su clase; sin esa
    entrada, el miembro usa los defaults de siempre."""
    params = hyperparameters.get("parameters", {})
    names = params.get("estimators", ["logistic_regression", "hist_gradient_boosting", "catboost"])
    member_hyperparameters = params.get("estimator_hyperparameters", {})
    return [_resolve_base_estimator(name, seed, member_hyperparameters.get(name)) for name in names]


class VotingEnsembleModel(SklearnLikeModel):
    model_name = "voting_ensemble"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        voting = self.hyperparameters.get("parameters", {}).get("voting", "soft")
        return VotingClassifier(
            estimators=_base_estimators(self.hyperparameters, self.seed), voting=voting
        )

    def _fit_estimator(
        self, X_train: Any, y_train: np.ndarray, X_valid: Any, y_valid: np.ndarray
    ) -> dict:
        self.estimator.fit(X_train, y_train)
        member_names = [name for name, _ in self.estimator.estimators]
        member_weights = [
            float(average_precision_score(y_valid, member.predict_proba(X_valid)[:, 1]))
            for member in self.estimator.estimators_
        ]
        self.estimator.weights = member_weights
        return {"voting_weights": dict(zip(member_names, member_weights))}


class StackingEnsembleModel(SklearnLikeModel):
    model_name = "stacking_ensemble"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        cv = self.hyperparameters.get("parameters", {}).get("cv", 5)
        return StackingClassifier(
            estimators=_base_estimators(self.hyperparameters, self.seed),
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.seed),
            cv=cv,
        )

    def _fit_estimator(
        self, X_train: Any, y_train: np.ndarray, X_valid: Any, y_valid: np.ndarray
    ) -> dict:
        self.estimator.fit(X_train, y_train)
        member_names = [name for name, _ in self.estimator.estimators]
        meta_coefs = self.estimator.final_estimator_.coef_.ravel().tolist()
        if len(meta_coefs) == len(member_names):
            stacking_meta_coefficients = dict(zip(member_names, meta_coefs))
        elif len(meta_coefs) == 2 * len(member_names):
            stacking_meta_coefficients = dict(zip(member_names, meta_coefs[1::2]))
        else:
            stacking_meta_coefficients = {"raw_coefficients": meta_coefs}
        return {"stacking_meta_coefficients": stacking_meta_coefficients}
