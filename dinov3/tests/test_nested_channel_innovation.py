import torch

from dinov3.loss.nested_channel_innovation_loss import (
    ConditionalFeaturePredictor,
    NestedChannelInnovationLoss,
    NestedChannelInnovationWeights,
    conditional_innovation_residual,
    martingale_increment_orthogonality,
)
from dinov3.train.ssl_meta_arch import (
    _make_low_resolution_observation,
    _require_rgb_backbone_for_nri,
    _sample_channel_subset_mask,
    _sample_nested_channel_masks,
)


def test_conditional_predictor_is_identity_at_initialization():
    predictor = ConditionalFeaturePredictor(dim=12, hidden_dim=24)
    predictor.reset_parameters()
    features = torch.randn(5, 12)

    torch.testing.assert_close(predictor(features), features)


def test_strict_nested_sampler_omits_a_channel_when_possible():
    torch.manual_seed(0)
    valid = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    subset = _sample_channel_subset_mask(
        valid,
        min_channels=1,
        max_channels=3,
        require_omission=True,
    )

    assert torch.equal(subset[0], valid[0])
    assert torch.all(subset.sum(dim=1)[1:] < valid.sum(dim=1)[1:])
    assert torch.all(subset.sum(dim=1) >= 1)
    assert torch.all(~subset | valid)


def test_martingale_sampler_builds_a_strict_channel_filtration():
    torch.manual_seed(0)
    valid = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    middle, lower = _sample_nested_channel_masks(valid)

    assert torch.all(~middle | valid)
    assert torch.all(~lower | middle)
    assert torch.equal(middle[0], valid[0])
    assert torch.equal(lower[0], valid[0])
    assert middle[1].sum() < valid[1].sum()
    assert lower[1].sum() == middle[1].sum()
    assert middle[2].sum() < valid[2].sum()
    assert lower[2].sum() < middle[2].sum()
    assert middle[3].sum() < valid[3].sum()
    assert lower[3].sum() < middle[3].sum()


class _DummyBackbone:
    def __init__(self, *, in_chans=3, stem_type=None, enable_channelvit=False):
        self.in_chans = in_chans
        self.stem_type = stem_type
        self.enable_channelvit = enable_channelvit


def test_nri_accepts_rgb_patch_embed_and_rejects_residual_mc():
    _require_rgb_backbone_for_nri(_DummyBackbone())
    try:
        _require_rgb_backbone_for_nri(_DummyBackbone(stem_type="residual_mc_v2", in_chans=8))
    except ValueError as exc:
        assert "stem_type='residual_mc_v2'" in str(exc)
    else:
        raise AssertionError("Expected Residual-MC NRI to be rejected")
    try:
        _require_rgb_backbone_for_nri(_DummyBackbone(in_chans=8))
    except ValueError as exc:
        assert "in_chans=8" in str(exc)
    else:
        raise AssertionError("Expected in_chans=8 NRI to be rejected")


def test_nested_low_resolution_observation_preserves_shape_and_removes_checkerboard():
    checkerboard = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).repeat(8, 8)
    images = checkerboard[None, None].repeat(2, 3, 1, 1)

    low = _make_low_resolution_observation(images, downsample_factor=2)

    assert low.shape == images.shape
    assert low.dtype == images.dtype
    assert low.var() < images.var() * 0.05


def test_innovation_metrics_support_a_resolution_prefix():
    predictor = ConditionalFeaturePredictor(dim=4, hidden_dim=8)
    predictor.reset_parameters()
    loss_fn = NestedChannelInnovationLoss(min_std=0.0, metric_prefix="nri")
    full = torch.randn(2, 3, 4)
    subset = torch.randn(2, 3, 4)

    _, metrics = loss_fn(
        full_features=full,
        subset_features=subset,
        active_samples=torch.ones(3, dtype=torch.bool),
        predictor=predictor,
    )

    assert "nri_predictor_loss" in metrics
    assert "nci_predictor_loss" not in metrics


def test_nci_stop_gradient_prevents_subset_deletion_pressure():
    torch.manual_seed(0)
    predictor = ConditionalFeaturePredictor(dim=8, hidden_dim=16)
    predictor.reset_parameters()
    loss_fn = NestedChannelInnovationLoss(
        min_std=0.05,
        weights=NestedChannelInnovationWeights(
            predictor=1.0,
            invariance=1.0,
            variance=1.0,
            orthogonality=1.0,
        ),
    )
    subset = torch.randn(2, 6, 8, requires_grad=True)
    shared_innovation = torch.randn(1, 6, 8).expand(2, -1, -1)
    full = (subset.detach() + shared_innovation).requires_grad_(True)

    loss, metrics = loss_fn(
        full_features=full,
        subset_features=subset,
        active_samples=torch.ones(6, dtype=torch.bool),
        predictor=predictor,
    )
    loss.backward()

    assert full.grad is not None
    assert full.grad.abs().sum() > 0
    assert subset.grad is None
    assert predictor.mlp[-1].weight.grad is not None
    assert predictor.mlp[-1].weight.grad.abs().sum() > 0
    assert metrics["nci_positive_cosine"] > 0.99
    assert metrics["nci_alignment_margin"] > 0


def test_nci_ignores_samples_without_omitted_channels():
    predictor = ConditionalFeaturePredictor(dim=4, hidden_dim=8)
    predictor.reset_parameters()
    loss_fn = NestedChannelInnovationLoss(min_std=0.1)
    full = torch.randn(2, 3, 4, requires_grad=True)
    subset = full.detach().clone().requires_grad_(True)

    loss, metrics = loss_fn(
        full_features=full,
        subset_features=subset,
        active_samples=torch.zeros(3, dtype=torch.bool),
        predictor=predictor,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert full.grad is not None
    assert full.grad.abs().sum() == 0
    assert subset.grad is None
    assert metrics["nci_active_fraction"].item() == 0.0


def test_martingale_increment_orthogonality_detects_disjoint_coordinates():
    # Centered upper/lower increments occupy separate coordinates, so their
    # feature-space cross-covariance is exactly zero.
    upper = torch.zeros(2, 4, 3, requires_grad=True)
    lower = torch.zeros(2, 4, 3, requires_grad=True)
    with torch.no_grad():
        values = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        upper[:, :, 0] = values
        lower[:, :, 1] = torch.tensor([1.0, -2.0, 2.0, -1.0])

    loss, metrics = martingale_increment_orthogonality(
        upper_increment=upper,
        lower_increment=lower,
        active_samples=torch.ones(4, dtype=torch.bool),
    )

    assert loss.item() < 1.0e-8
    assert metrics["nci_martingale_active_fraction"].item() == 1.0


def test_conditional_innovation_residual_keeps_predictor_behind_firewall():
    predictor = ConditionalFeaturePredictor(dim=4, hidden_dim=8)
    predictor.reset_parameters()
    full = torch.randn(2, 3, 4, requires_grad=True)
    subset = torch.randn(2, 3, 4, requires_grad=True)

    residual = conditional_innovation_residual(
        full_features=full,
        subset_features=subset,
        predictor=predictor,
        stop_gradient=True,
    )
    residual.square().mean().backward()

    assert full.grad is not None and full.grad.abs().sum() > 0
    assert subset.grad is None
    assert predictor.mlp[-1].weight.grad is None


def test_disabling_firewall_reintroduces_subset_and_full_prediction_gradients():
    predictor = ConditionalFeaturePredictor(dim=6, hidden_dim=12)
    predictor.reset_parameters()
    loss_fn = NestedChannelInnovationLoss(
        min_std=0.0,
        stop_gradient=False,
        weights=NestedChannelInnovationWeights(
            predictor=1.0,
            invariance=0.0,
            variance=0.0,
            orthogonality=0.0,
        ),
    )
    subset = torch.randn(2, 5, 6, requires_grad=True)
    full = torch.randn(2, 5, 6, requires_grad=True)

    loss, _ = loss_fn(
        full_features=full,
        subset_features=subset,
        active_samples=torch.ones(5, dtype=torch.bool),
        predictor=predictor,
    )
    loss.backward()

    assert subset.grad is not None and subset.grad.abs().sum() > 0
    assert full.grad is not None and full.grad.abs().sum() > 0


def test_nci_rejects_mismatched_shapes():
    predictor = ConditionalFeaturePredictor(dim=4, hidden_dim=8)
    loss_fn = NestedChannelInnovationLoss()

    try:
        loss_fn(
            full_features=torch.randn(2, 3, 4),
            subset_features=torch.randn(2, 4, 4),
            active_samples=torch.ones(3, dtype=torch.bool),
            predictor=predictor,
        )
    except ValueError as exc:
        assert "same [V, B, D] shape" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched NCI feature shapes")
