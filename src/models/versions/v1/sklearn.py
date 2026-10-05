"""Modelos con API sklearn nativa: Dummy, LogisticRegression, HistGradientBoosting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from models.versions.v1.base import SklearnLikeModel


class DummyModel(SklearnLikeModel):
    model_name = "dummy"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        hp = self.hyperparameters.get("parameters", {})
        strategy = hp.get("strategy", "prior")
        return DummyClassifier(strategy=strategy, random_state=self.seed)


class LogisticRegressionModel(SklearnLikeModel):
    model_name = "logistic_regression"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        hp = self.hyperparameters.get("parameters", {})
        return LogisticRegression(
            penalty=hp.get("penalty", "l2"),
            C=hp.get("C", 1.0),
            class_weight=hp.get("class_weight", "balanced"),
            max_iter=hp.get("max_iter", 1000),
            solver=hp.get("solver", "lbfgs"),
            random_state=self.seed,
        )


class HistGradientBoostingModel(SklearnLikeModel):
    model_name = "hist_gradient_boosting"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        hp = self.hyperparameters.get("parameters", {})
        return HistGradientBoostingClassifier(
            max_iter=hp.get("max_iter", 300),
            learning_rate=hp.get("learning_rate", 0.05),
            max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            class_weight=hp.get("class_weight", "balanced"),
            early_stopping=hp.get("early_stopping", True),
            n_iter_no_change=hp.get("n_iter_no_change", 20),
            random_state=self.seed,
        )


class GradientBoostingModel(SklearnLikeModel):
    """Boosting clasico (no histograma). Sustituto de `hist_gradient_boosting` cuando este no
    exporta a ONNX en el entorno (bug de skl2onnx con HistGradientBoostingClassifier, ver
    notebooks/02_orquestar_entrenamiento.ipynb, seccion de diagnostico)."""

    model_name = "gradient_boosting"
    framework = "sklearn"

    def _build_estimator(self) -> Any:
        hp = self.hyperparameters.get("parameters", {})
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", 3),
            learning_rate=hp.get("learning_rate", 0.05),
            subsample=hp.get("subsample", 1.0),
            n_iter_no_change=hp.get("n_iter_no_change", 20),
            random_state=self.seed,
        )

    def _fit_estimator(
        self, X_train: Any, y_train: np.ndarray, X_valid: Any, y_valid: np.ndarray
    ) -> dict:
        # GradientBoostingClassifier no tiene class_weight en el constructor (a diferencia de
        # HistGradientBoostingClassifier/LogisticRegression); se reproduce "balanced" a mano via
        # sample_weight: la clase positiva pesa n_neg/n_pos, la negativa pesa 1.0.
        hp = self.hyperparameters.get("parameters", {})
        sample_weight = None
        if hp.get("class_weight", "balanced") == "balanced":
            n_pos = max(int(np.asarray(y_train).sum()), 1)
            n_neg = max(len(y_train) - n_pos, 1)
            sample_weight = np.where(np.asarray(y_train) == 1, n_neg / n_pos, 1.0)
        self.estimator.fit(X_train, y_train, sample_weight=sample_weight)
        return {}
