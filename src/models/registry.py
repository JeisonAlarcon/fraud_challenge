"""Registro de modelos disponibles por model_version."""

from __future__ import annotations

from models.versions.v1.base import BaseModel
from models.versions.v1.catboost_model import CatBoostModel
from models.versions.v1.dcnv2 import DCNv2Model
from models.versions.v1.ensemble import StackingEnsembleModel, VotingEnsembleModel
from models.versions.v1.ft_transformer import FTTransformerModel
from models.versions.v1.mlp import MLPModel
from models.versions.v1.sklearn import (
    DummyModel,
    GradientBoostingModel,
    HistGradientBoostingModel,
    LogisticRegressionModel,
)

MODEL_REGISTRY: dict[str, dict[str, type[BaseModel]]] = {
    "v1": {
        "dummy": DummyModel,
        "logistic_regression": LogisticRegressionModel,
        "hist_gradient_boosting": HistGradientBoostingModel,
        "gradient_boosting": GradientBoostingModel,
        "catboost": CatBoostModel,
        "voting_ensemble": VotingEnsembleModel,
        "stacking_ensemble": StackingEnsembleModel,
        "mlp": MLPModel,
        "dcnv2": DCNv2Model,
        "ft_transformer": FTTransformerModel,
    }
}


class UnknownModelError(KeyError):
    pass


def resolve_model_class(model_version: str, model: str) -> type[BaseModel]:
    if model_version not in MODEL_REGISTRY or model not in MODEL_REGISTRY[model_version]:
        available = sorted(MODEL_REGISTRY.get(model_version, {}).keys())
        raise UnknownModelError(
            f"No existe modelo '{model}' registrado para model_version='{model_version}'. "
            f"Disponibles: {available}"
        )
    return MODEL_REGISTRY[model_version][model]
