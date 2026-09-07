from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dinov3.data.global_bridge_target_bank import GlobalBridgeTargetBank
from dinov3.loss.global_bridge_transport_loss import (
    GlobalBridgeTransportLoss,
    compose_global_bridge_readout,
)
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.train.ssl_meta_arch import SSLMetaArch, _renormalize_rgb_images


def _write_bank(path: Path) -> None:
    anchors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float16)
    targets = np.asarray([[0.5, 0.866, 0.0], [0.0, 0.5, 0.866]], dtype=np.float16)
    controls = targets[::-1].copy()
    np.savez(
        path,
        keys=np.asarray(["a", "c"]),
        anchor_features=anchors,
        target_features=targets,
        control_target_features=controls,
        feature_protocol=np.asarray("final_cls"),
        input_normalization=np.asarray("train"),
        observation_crop_size=np.asarray(0, dtype=np.int32),
    )


def test_bridge_target_bank_intersects_batch_keys(tmp_path: Path) -> None:
    path = tmp_path / "targets.npz"
    _write_bank(path)
    bank = GlobalBridgeTargetBank(path)
    batch = bank.lookup(["missing", "c", "a"], device="cpu")
    assert batch.sample_indices.tolist() == [1, 2]
    assert batch.anchor_features.shape == (2, 3)
    assert bank.feature_dim == 3
    assert bank.feature_protocol == "final_cls"
    assert bank.input_normalization == "train"
    assert bank.observation_crop_size == 0
    assert torch.allclose(
        batch.anchor_features[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float16)
    )

    control = GlobalBridgeTargetBank(path, control=True).lookup(["a"], device="cpu")
    assert torch.allclose(
        control.target_features[0],
        torch.tensor([0.0, 0.5, 0.866], dtype=torch.float16),
    )


def test_bridge_transport_preserves_offline_angle_and_backpropagates() -> None:
    student = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    current_anchor = torch.tensor([[1.0, 0.0, 0.0]])
    bank_anchor = torch.tensor([[0.0, 1.0, 0.0]])
    bank_target = torch.tensor([[0.0, 0.5, 0.8660254]])
    loss, metrics = GlobalBridgeTransportLoss()(
        student_features=student,
        current_anchor_features=current_anchor,
        bank_anchor_features=bank_anchor,
        bank_target_features=bank_target,
    )
    assert torch.allclose(loss, torch.tensor(0.5), atol=1e-5)
    assert torch.allclose(metrics["gbt_target_angle"], torch.tensor(torch.pi / 3), atol=1e-5)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.norm() > 0


def test_bridge_transport_empty_batch_has_stable_metric_schema() -> None:
    empty = torch.empty((0, 3), requires_grad=True)
    loss, metrics = GlobalBridgeTransportLoss()(
        student_features=empty,
        current_anchor_features=empty.detach(),
        bank_anchor_features=empty.detach(),
        bank_target_features=empty.detach(),
    )
    assert set(metrics) == {
        "gbt_loss",
        "gbt_active_rows",
        "gbt_valid_direction_fraction",
        "gbt_target_angle",
        "gbt_student_target_cosine",
        "gbt_anchor_target_cosine",
    }
    assert all(float(value) == 0.0 for value in metrics.values())
    loss.backward()
    assert empty.grad is not None


def test_nlb2_avg_composition_matches_frozen_eval_convention() -> None:
    penultimate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    final = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    patches = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    actual = compose_global_bridge_readout(
        final_cls=final,
        penultimate_cls=penultimate,
        final_patches=patches,
        feature_protocol="nlb2_avg",
    )
    expected = torch.cat((penultimate, final, patches.mean(dim=1)), dim=-1)
    assert torch.equal(actual, expected)


def test_training_penultimate_output_matches_intermediate_layer_evaluator() -> None:
    torch.manual_seed(0)
    model = DinoVisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        ffn_ratio=2.0,
        pos_embed_rope_dtype="fp32",
    )
    model.init_weights()
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(images, is_training=True, return_penultimate=True)
        intermediate = model.get_intermediate_layers(
            images,
            n=2,
            reshape=False,
            return_class_token=True,
        )
    expected = torch.cat(
        (
            intermediate[0][1],
            intermediate[1][1],
            intermediate[1][0].mean(dim=1),
        ),
        dim=-1,
    )
    actual = compose_global_bridge_readout(
        final_cls=output["x_norm_clstoken"],
        penultimate_cls=output["x_norm_penultimate_clstoken"],
        final_patches=output["x_norm_patchtokens"],
        feature_protocol="nlb2_avg",
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5)


def test_bridge_rgb_renormalization_preserves_underlying_pixels() -> None:
    raw = torch.tensor([[[[[0.2]], [[0.4]], [[0.8]]]]])
    source_mean = (0.5, 0.4, 0.3)
    source_std = (0.25, 0.5, 0.2)
    target_mean = (0.1, 0.2, 0.3)
    target_std = (0.5, 0.25, 0.1)
    source = (raw - raw.new_tensor(source_mean).view(1, 1, 3, 1, 1)) / raw.new_tensor(
        source_std
    ).view(1, 1, 3, 1, 1)
    actual = _renormalize_rgb_images(
        source,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    expected = (raw - raw.new_tensor(target_mean).view(1, 1, 3, 1, 1)) / raw.new_tensor(
        target_std
    ).view(1, 1, 3, 1, 1)
    assert torch.allclose(actual, expected)


def test_bridge_observation_uses_center_crop_without_changing_main_input() -> None:
    images = torch.arange(1 * 1 * 3 * 4 * 6).reshape(1, 1, 3, 4, 6).float()
    bridge = SimpleNamespace(
        global_bridge_input_normalization="train",
        global_bridge_observation_crop_size=2,
    )
    actual = SSLMetaArch._prepare_global_bridge_images(bridge, images)
    assert torch.equal(actual, images[..., 1:3, 2:4])
    assert actual.is_contiguous()
    assert images.shape == (1, 1, 3, 4, 6)
