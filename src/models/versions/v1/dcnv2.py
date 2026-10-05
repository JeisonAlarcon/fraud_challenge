"""DCN-V2 (arquitectura *parallel*): cross network de bajo rango + deep MLP, concatenados
antes del logit final. https://arxiv.org/abs/2008.13535
"""

from __future__ import annotations

import torch
from torch import nn

from models.versions.v1.base import TorchModelAdapter


class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, low_rank: int):
        super().__init__()
        self.U = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, low_rank) * 0.01) for _ in range(num_layers)]
        )
        self.V = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, low_rank) * 0.01) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for U, V, b in zip(self.U, self.V, self.bias):
            projected = x @ V
            crossed = projected @ U.t()
            x = x0 * (crossed + b) + x
        return x


class DCNv2Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cross_layers: int,
        low_rank: int,
        deep_layers: list[int],
        dropout: float,
    ):
        super().__init__()
        self.cross = CrossNetworkV2(input_dim, cross_layers, low_rank)
        deep_modules: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in deep_layers:
            deep_modules += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        self.deep = nn.Sequential(*deep_modules)
        self.head = nn.Linear(input_dim + prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_out = self.cross(x)
        deep_out = self.deep(x)
        return self.head(torch.cat([cross_out, deep_out], dim=-1))


class DCNv2Model(TorchModelAdapter):
    model_name = "dcnv2"

    def build_network(self, input_dim: int) -> nn.Module:
        hp = self.hyperparameters.get("parameters", {})
        return DCNv2Network(
            input_dim=input_dim,
            cross_layers=hp.get("cross_layers", 3),
            low_rank=hp.get("low_rank", 32),
            deep_layers=hp.get("deep_layers", [256, 128]),
            dropout=hp.get("dropout", 0.15),
        )
