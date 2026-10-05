"""CatBoost sobre la misma matriz numerica canonica del cache (sin `cat_features` nativo)."""

from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostClassifier, Pool

from models.versions.v1.base import SklearnLikeModel


class CatBoostModel(SklearnLikeModel):
    model_name = "catboost"
    framework = "catboost"

    def _build_estimator(self) -> Any:
        hp = self.hyperparameters.get("parameters", {})
        kwargs: dict[str, Any] = dict(
            iterations=hp.get("iterations", 2000),
            depth=hp.get("depth", 6),
            learning_rate=hp.get("learning_rate", 0.03),
            l2_leaf_reg=hp.get("l2_leaf_reg", 3.0),
            auto_class_weights=hp.get("auto_class_weights", "Balanced"),
            random_seed=self.seed,
            verbose=hp.get("verbose", 200),  # progreso nativo cada N iteraciones, no silencioso
        )
        if "bagging_temperature" in hp:
            # bagging_temperature solo aplica con bootstrap_type="Bayesian" (regularizacion extra
            # contra el overfitting temprano que se vio en v1 con los hiperparametros default).
            kwargs["bootstrap_type"] = "Bayesian"
            kwargs["bagging_temperature"] = hp["bagging_temperature"]
        return CatBoostClassifier(**kwargs)

    def _fit_estimator(
        self, X_train: Any, y_train: np.ndarray, X_valid: Any, y_valid: np.ndarray
    ) -> dict:
        trainer_cfg = self.hyperparameters.get("trainer", {})
        self.estimator.fit(
            Pool(X_train, y_train),
            eval_set=Pool(X_valid, y_valid),
            early_stopping_rounds=trainer_cfg.get("early_stopping_rounds", 100),
            use_best_model=trainer_cfg.get("use_best_model", True),
        )
        return {"best_iteration": self.estimator.get_best_iteration()}
