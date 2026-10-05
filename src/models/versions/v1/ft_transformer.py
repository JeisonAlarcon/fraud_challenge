"""FT-Transformer: un token por columna final del cache (no por feature semantica original,
para no acoplarse a convenciones de nombres de `feature_names.json`).
https://arxiv.org/abs/2106.11959
"""

from __future__ import annotations

import torch
from torch import nn

from models.versions.v1.base import TorchModelAdapter


class FTTransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        ffn_dim: int,
        attn_dropout: float,
    ):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(input_dim, token_dim) * 0.01)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = (
            x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        )  # (batch, input_dim, token_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(sequence)
        cls_out = self.norm(encoded[:, 0, :])
        return self.head(cls_out)


class FTTransformerModel(TorchModelAdapter):
    model_name = "ft_transformer"

    def build_network(self, input_dim: int) -> nn.Module:
        hp = self.hyperparameters.get("parameters", {})
        return FTTransformerNetwork(
            input_dim=input_dim,
            token_dim=hp.get("token_dim", 32),
            n_transformer_layers=hp.get("n_transformer_layers", 3),
            n_heads=hp.get("n_heads", 8),
            ffn_dim=hp.get("ffn_dim", 128),
            attn_dropout=hp.get("attn_dropout", 0.1),
        )
