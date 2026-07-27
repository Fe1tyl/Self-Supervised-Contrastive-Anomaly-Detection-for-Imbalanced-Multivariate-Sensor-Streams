from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import MultiObjectiveTCN, objectives_for


@dataclass(frozen=True)
class AugmentationConfig:
    jitter_sigma: float
    temporal_mask_probability: float
    channel_dropout_probability: float
    scaling_low: float
    scaling_high: float


def augmentation_config(config: dict[str, Any]) -> AugmentationConfig:
    values = config["augmentation"]
    return AugmentationConfig(
        jitter_sigma=float(values["jitter_sigma"]),
        temporal_mask_probability=float(values["temporal_mask_probability"]),
        channel_dropout_probability=float(values["channel_dropout_probability"]),
        scaling_low=float(values["scaling_low"]),
        scaling_high=float(values["scaling_high"]),
    )


def augment_windows(
    x: torch.Tensor,
    continuous_mask: torch.Tensor,
    settings: AugmentationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    output = x.clone()
    batch, length, features = output.shape
    continuous = continuous_mask.view(1, 1, features)
    scaling = torch.empty(
        (batch, 1, features), device=x.device, dtype=x.dtype
    ).uniform_(settings.scaling_low, settings.scaling_high, generator=generator)
    scaling = torch.where(continuous, scaling, torch.ones_like(scaling))
    output = output * scaling
    jitter = torch.randn(
        output.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    ) * settings.jitter_sigma
    output = output + jitter * continuous
    temporal_keep = (
        torch.rand(
            (batch, length, 1),
            device=x.device,
            generator=generator,
        )
        >= settings.temporal_mask_probability
    )
    channel_keep = (
        torch.rand(
            (batch, 1, features),
            device=x.device,
            generator=generator,
        )
        >= settings.channel_dropout_probability
    )
    return output * temporal_keep.to(output.dtype) * channel_keep.to(output.dtype)


def nt_xent(z_first: torch.Tensor, z_second: torch.Tensor, temperature: float) -> torch.Tensor:
    if z_first.shape != z_second.shape:
        raise ValueError("Contrastive views have different embedding shapes")
    batch = z_first.shape[0]
    if batch < 2:
        raise ValueError("NT-Xent requires at least two windows per batch")
    embeddings = torch.cat([z_first, z_second], dim=0).float()
    similarity = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=similarity.device)
    similarity = similarity.masked_fill(diagonal, -torch.inf)
    positive = (torch.arange(2 * batch, device=similarity.device) + batch) % (
        2 * batch
    )
    positive_logits = similarity[
        torch.arange(2 * batch, device=similarity.device), positive
    ]
    return (-positive_logits + torch.logsumexp(similarity, dim=1)).mean()


def compute_objective(
    model: MultiObjectiveTCN,
    x: torch.Tensor,
    model_name: str,
    config: dict[str, Any],
    continuous_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = set(objectives_for(model_name))
    zero = torch.zeros((), device=x.device, dtype=torch.float32)
    raw = {
        "contrastive": zero,
        "reconstruction": zero,
        "prediction": zero,
    }

    if active & {"reconstruction", "prediction"}:
        original = model(x)
        if "reconstruction" in active:
            raw["reconstruction"] = F.mse_loss(original["reconstruction"], x)
        if "prediction" in active:
            raw["prediction"] = F.mse_loss(original["prediction"], x[:, 1:])

    if "contrastive" in active:
        settings = augmentation_config(config)
        first = augment_windows(x, continuous_mask, settings, generator)
        second = augment_windows(x, continuous_mask, settings, generator)
        views = torch.cat([first, second], dim=0)
        hidden = model.encode_sequence(views)
        embeddings = model.embed(hidden)
        z_first, z_second = embeddings.chunk(2, dim=0)
        raw["contrastive"] = nt_xent(
            z_first, z_second, float(config["loss"]["temperature"])
        )

    weights = {
        "contrastive": float(config["loss"]["contrastive_weight"]),
        "reconstruction": float(config["loss"]["reconstruction_weight"]),
        "prediction": float(config["loss"]["prediction_weight"]),
    }
    weighted = {name: raw[name] * weights[name] for name in raw}
    total = sum(weighted.values(), start=zero)
    values = {
        **{f"raw_{name}": value for name, value in raw.items()},
        **{f"weighted_{name}": value for name, value in weighted.items()},
        "total": total,
    }
    return total, values

