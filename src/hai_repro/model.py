from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "full": ("contrastive", "reconstruction", "prediction"),
    "tcn_autoencoder": ("reconstruction",),
    "tcn_predictor": ("prediction",),
    "contrastive": ("contrastive",),
    "t_minus_contrastive": ("reconstruction", "prediction"),
    "t_minus_reconstruction": ("contrastive", "prediction"),
    "t_minus_prediction": ("contrastive", "reconstruction"),
}

MODEL_SCORES: dict[str, tuple[str, ...]] = {
    "full": ("representation", "reconstruction", "prediction"),
    "tcn_autoencoder": ("reconstruction",),
    "tcn_predictor": ("prediction",),
    "contrastive": ("representation",),
    "t_minus_contrastive": ("reconstruction", "prediction"),
    "t_minus_reconstruction": ("representation", "prediction"),
    "t_minus_prediction": ("representation", "reconstruction"),
}


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self.left_padding = dilation * (kernel_size - 1)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        if self.left_padding:
            output = output[..., : -self.left_padding]
        return output


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(x.transpose(1, 2)).transpose(1, 2)


class ResidualCausalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.dropout(self.activation(x))
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(self.activation(x))
        return self.activation(x + residual)


class MultiObjectiveTCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        embedding_dim: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_channels = hidden_channels
        self.input_projection = nn.Conv1d(input_features, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualCausalBlock(
                    hidden_channels, kernel_size, dilation, dropout
                )
                for dilation in dilations
            ]
        )
        self.projection_head = nn.Linear(hidden_channels, embedding_dim)
        self.reconstruction_head = nn.Linear(hidden_channels, input_features)
        self.prediction_head = nn.Linear(hidden_channels, input_features)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.transpose(1, 2)

    def embed(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        return F.normalize(self.projection_head(pooled), dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode_sequence(x)
        return {
            "hidden": hidden,
            "embedding": self.embed(hidden),
            "reconstruction": self.reconstruction_head(hidden),
            "prediction": self.prediction_head(hidden[:, :-1]),
        }


def build_model(config: dict[str, Any], input_features: int) -> MultiObjectiveTCN:
    model = config["model"]
    return MultiObjectiveTCN(
        input_features=input_features,
        hidden_channels=int(model["hidden_channels"]),
        embedding_dim=int(model["embedding_dim"]),
        kernel_size=int(model["kernel_size"]),
        dilations=[int(value) for value in model["dilations"]],
        dropout=float(model["dropout"]),
    )


def objectives_for(model_name: str) -> tuple[str, ...]:
    try:
        return MODEL_OBJECTIVES[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model configuration: {model_name}") from error


def scores_for(model_name: str) -> tuple[str, ...]:
    try:
        return MODEL_SCORES[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model configuration: {model_name}") from error


def trainable_parameters(
    model: MultiObjectiveTCN, model_name: str
) -> list[nn.Parameter]:
    active = set(objectives_for(model_name))
    modules: list[nn.Module] = [model.input_projection, model.blocks]
    if "contrastive" in active:
        modules.append(model.projection_head)
    if "reconstruction" in active:
        modules.append(model.reconstruction_head)
    if "prediction" in active:
        modules.append(model.prediction_head)
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameter.requires_grad_(True)
                parameters.append(parameter)
                seen.add(id(parameter))
    for parameter in model.parameters():
        if id(parameter) not in seen:
            parameter.requires_grad_(False)
    return parameters


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }

