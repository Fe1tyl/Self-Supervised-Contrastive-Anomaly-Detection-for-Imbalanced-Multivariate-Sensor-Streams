import copy

import torch

from hai_repro.losses import augment_windows, augmentation_config, compute_objective
from hai_repro.model import build_model, trainable_parameters


def tiny_config() -> dict:
    return {
        "model": {
            "hidden_channels": 8,
            "embedding_dim": 8,
            "kernel_size": 3,
            "dilations": [1, 2],
            "dropout": 0.0,
        },
        "augmentation": {
            "jitter_sigma": 0.02,
            "temporal_mask_probability": 0.0,
            "channel_dropout_probability": 0.0,
            "scaling_low": 0.9,
            "scaling_high": 1.1,
        },
        "loss": {
            "contrastive_weight": 1.0,
            "reconstruction_weight": 0.5,
            "prediction_weight": 0.5,
            "temperature": 0.1,
        },
    }


def test_causal_encoder_has_no_future_access() -> None:
    config = tiny_config()
    model = build_model(config, 4).eval()
    x = torch.randn(2, 16, 4)
    changed = x.clone()
    changed[:, 8:] += 100.0
    with torch.inference_mode():
        original = model.encode_sequence(x)
        perturbed = model.encode_sequence(changed)
    torch.testing.assert_close(original[:, :8], perturbed[:, :8], atol=0, rtol=0)


def test_prediction_alignment_and_shapes() -> None:
    model = build_model(tiny_config(), 4)
    x = torch.randn(3, 16, 4)
    output = model(x)
    assert output["hidden"].shape == (3, 16, 8)
    assert output["embedding"].shape == (3, 8)
    assert output["reconstruction"].shape == x.shape
    assert output["prediction"].shape == (3, 15, 4)
    assert x[:, 1:].shape == output["prediction"].shape


def test_binary_features_are_not_jittered_or_scaled() -> None:
    config = tiny_config()
    settings = augmentation_config(config)
    x = torch.ones(2, 8, 4)
    continuous = torch.tensor([True, False, True, False])
    augmented = augment_windows(x, continuous, settings)
    torch.testing.assert_close(augmented[:, :, 1], x[:, :, 1])
    torch.testing.assert_close(augmented[:, :, 3], x[:, :, 3])
    assert not torch.equal(augmented[:, :, 0], x[:, :, 0])


def test_removed_objective_is_exactly_zero_and_head_is_frozen() -> None:
    config = tiny_config()
    model = build_model(config, 4)
    parameters = trainable_parameters(model, "t_minus_reconstruction")
    assert parameters
    assert not any(
        parameter.requires_grad for parameter in model.reconstruction_head.parameters()
    )
    x = torch.randn(4, 16, 4)
    continuous = torch.ones(4, dtype=torch.bool)
    total, values = compute_objective(
        model, x, "t_minus_reconstruction", config, continuous
    )
    total.backward()
    assert values["weighted_reconstruction"].item() == 0.0

