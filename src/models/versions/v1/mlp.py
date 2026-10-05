"""MLP tabular: baseline neural sobre el mismo vector denso del cache comun."""

from __future__ import annotations

from torch import nn

from models.versions.v1.base import TorchModelAdapter


class MLPModel(TorchModelAdapter):
    model_name = "mlp"

    def build_network(self, input_dim: int) -> nn.Module:
        hp = self.hyperparameters.get("parameters", {})
        hidden_layers = hp.get("hidden_layers", [256, 128, 64])
        dropout = hp.get("dropout", 0.2)
        use_batch_norm = hp.get("batch_norm", True)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
