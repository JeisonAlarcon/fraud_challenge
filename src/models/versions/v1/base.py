"""Interfaz comun a todos los modelos v1 + loop de entrenamiento compartido para PyTorch."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.history import EpochHistory

OnEpochEnd = Callable[[int | str, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]], None]


class BaseModel(ABC):
    model_name: str
    supports_epochs: bool
    framework: Literal["sklearn", "catboost", "torch"]

    def __init__(self, hyperparameters: dict, seed: int, feature_names: list[str], input_dim: int):
        self.hyperparameters = hyperparameters
        self.seed = seed
        self.feature_names = feature_names
        self.input_dim = input_dim

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any,
        y_valid: np.ndarray,
        on_epoch_end: OnEpochEnd | None = None,
    ) -> EpochHistory: ...

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray: ...

    def raw_score(self, X: Any) -> np.ndarray:
        """Score exactamente equivalente a lo que exporta `model_raw.onnx` (para paridad y
        calibracion). Por defecto es `predict_proba` (sklearn/CatBoost); `TorchModelAdapter` lo
        sobreescribe para devolver el logit pre-sigmoid, que es lo que el ONNX exportado
        realmente produce."""
        return self.predict_proba(X)

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel": ...


def _to_dense_float32(X: Any) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _fallback_metrics(y_valid: np.ndarray, valid_scores: np.ndarray) -> dict:
    """Metricas minimas cuando no hay callback (p.ej. tests): solo lo necesario para que
    `EpochHistory.is_best_so_far`/`should_early_stop` funcionen."""
    from sklearn.metrics import average_precision_score

    return {"valid": {"average_precision": average_precision_score(y_valid, valid_scores)}}


class TorchModelAdapter(BaseModel):
    """Loop de entrenamiento compartido por MLP/DCN-V2/FT-Transformer. Las subclases solo
    implementan `build_network(input_dim) -> nn.Module`."""

    supports_epochs = True
    framework = "torch"

    def __init__(self, hyperparameters: dict, seed: int, feature_names: list[str], input_dim: int):
        super().__init__(hyperparameters, seed, feature_names, input_dim)
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.network: nn.Module = self.build_network(input_dim)
        self.network.to(self.device)

    @abstractmethod
    def build_network(self, input_dim: int) -> nn.Module: ...

    def _trainer_config(self) -> dict:
        return self.hyperparameters.get("trainer", {})

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any,
        y_valid: np.ndarray,
        on_epoch_end: OnEpochEnd | None = None,
    ) -> EpochHistory:
        trainer_cfg = self._trainer_config()
        batch_size = int(trainer_cfg.get("batch_size", 512))
        max_epochs = int(trainer_cfg.get("max_epochs", 50))
        learning_rate = float(trainer_cfg.get("learning_rate", 1e-3))
        weight_decay = float(trainer_cfg.get("weight_decay", 0.0))
        early_stopping_cfg = trainer_cfg.get(
            "early_stopping", {"monitor": "valid.average_precision", "patience": 10, "mode": "max"}
        )
        lr_scheduler_cfg = trainer_cfg.get("lr_scheduler", {})

        X_train_t = torch.from_numpy(_to_dense_float32(X_train))
        y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        X_valid_t = torch.from_numpy(_to_dense_float32(X_valid))
        y_valid_np = np.asarray(y_valid, dtype=np.float32)

        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        if trainer_cfg.get("pos_weight") == "balanced":
            n_pos = max(float(y_train_t.sum().item()), 1.0)
            n_neg = max(float(len(y_train_t)) - n_pos, 1.0)
            pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # LR adaptativo: si valid.average_precision no mejora en `patience` epochs, reduce el LR
        # por `factor`. Complementa (no reemplaza) el early stopping, que sigue monitoreando lo
        # mismo con su propia paciencia (mayor a la del scheduler para dar margen a la reduccion).
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_scheduler_cfg.get("factor", 0.5),
            patience=lr_scheduler_cfg.get("patience", 5),
            min_lr=lr_scheduler_cfg.get("min_lr", 1e-6),
        )

        history = EpochHistory()
        best_state: dict[str, Any] | None = None
        cumulative_seconds = 0.0

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            self.network.train()
            total_loss = 0.0
            grad_norm_sum = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += float(loss.item())
                grad_norm_sum += float(grad_norm)
                n_batches += 1

            epoch_seconds = time.perf_counter() - epoch_start
            cumulative_seconds += epoch_seconds

            self.network.eval()
            with torch.no_grad():
                train_scores = (
                    torch.sigmoid(self.network(X_train_t.to(self.device)).squeeze(-1)).cpu().numpy()
                )
                valid_scores = (
                    torch.sigmoid(self.network(X_valid_t.to(self.device)).squeeze(-1)).cpu().numpy()
                )

            extra = {
                "learning_rate": current_lr,
                "grad_norm": grad_norm_sum / max(n_batches, 1),
                "epoch_seconds": epoch_seconds,
                "cumulative_seconds": cumulative_seconds,
                "throughput_samples_per_sec": len(X_train_t) / max(epoch_seconds, 1e-9),
                "gpu_memory_mb": None,
                "train_loss": total_loss / max(n_batches, 1),
            }
            scores_by_split = {
                "train": (np.asarray(y_train_t.numpy()), train_scores),
                "valid": (y_valid_np, valid_scores),
            }
            if on_epoch_end is not None:
                metrics_this_epoch = on_epoch_end(epoch, scores_by_split, extra)
            else:
                metrics_this_epoch = _fallback_metrics(y_valid_np, valid_scores)
            history.record(epoch, metrics_this_epoch)

            valid_pr_auc = metrics_this_epoch.get("valid", {}).get("average_precision")
            if valid_pr_auc is not None:
                scheduler.step(valid_pr_auc)

            if history.is_best_so_far(epoch):
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}

            if history.should_early_stop(
                patience=int(early_stopping_cfg.get("patience", 10)),
                mode=early_stopping_cfg.get("mode", "max"),
            ):
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        return history

    def raw_score(self, X: Any) -> np.ndarray:
        self.network.eval()
        X_t = torch.from_numpy(_to_dense_float32(X))
        with torch.no_grad():
            return self.network(X_t.to(self.device)).squeeze(-1).cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw_score(X)))

    def save(self, path: Path) -> None:
        payload = {
            "state_dict": self.network.state_dict(),
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "feature_names": self.feature_names,
            "input_dim": self.input_dim,
            "class_name": self.__class__.__name__,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "TorchModelAdapter":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            hyperparameters=payload["hyperparameters"],
            seed=payload["seed"],
            feature_names=payload["feature_names"],
            input_dim=payload["input_dim"],
        )
        instance.network.load_state_dict(payload["state_dict"])
        return instance


class SklearnLikeModel(BaseModel):
    """Base para modelos de un solo `fit` (sklearn/CatBoost/ensembles): `self.estimator` expone
    `fit`/`predict_proba` nativos. Centraliza construccion, fit -> scores -> callback -> history
    ("final") comun a las cuatro familias; una subclase solo sobreescribe `_build_estimator`
    (que estimator nativo construir) y, si su `.fit` no es un `estimator.fit(X, y)` liso (caso
    CatBoost: necesita `eval_set`/`early_stopping_rounds`), `_fit_estimator`."""

    supports_epochs = False

    def __init__(self, hyperparameters: dict, seed: int, feature_names: list[str], input_dim: int):
        super().__init__(hyperparameters, seed, feature_names, input_dim)
        self.estimator: Any = self._build_estimator()

    def _build_estimator(self) -> Any:
        raise NotImplementedError

    def _fit_estimator(
        self, X_train: Any, y_train: np.ndarray, X_valid: Any, y_valid: np.ndarray
    ) -> dict:
        self.estimator.fit(X_train, y_train)
        return {}

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any,
        y_valid: np.ndarray,
        on_epoch_end: OnEpochEnd | None = None,
    ) -> EpochHistory:
        print(f"[{self.model_name}] fit() sin epochs (un solo llamado bloqueante) - iniciando...")
        start = time.perf_counter()
        extra = self._fit_estimator(X_train, y_train, X_valid, y_valid)
        print(f"[{self.model_name}] fit() termino en {time.perf_counter() - start:.1f}s")
        scores_by_split = {
            "train": (y_train, self.predict_proba(X_train)),
            "valid": (y_valid, self.predict_proba(X_valid)),
        }
        if on_epoch_end is not None:
            metrics_final = on_epoch_end("final", scores_by_split, extra)
        else:
            metrics_final = _fallback_metrics(y_valid, scores_by_split["valid"][1])

        history = EpochHistory()
        history.record("final", metrics_final)
        history.is_best_so_far("final")
        return history

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnLikeModel":
        return joblib.load(path)
