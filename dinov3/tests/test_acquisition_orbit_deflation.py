import torch

from dinov3.loss.acquisition_orbit_deflation_loss import AcquisitionOrbitDeflationLoss
from dinov3.loss.acquisition_tangent_projection import (
    apply_acquisition_tangent_gradient_projection,
    build_acquisition_tangent_basis,
    project_onto_acquisition_tangent,
    rank_matched_random_tangent_basis,
    shift_tangent_correspondence,
)
from dinov3.train.ssl_meta_arch import _make_acquisition_orbit_views


def _anchors(batch_size=4, dim=7):
    torch.manual_seed(11)
    anchor = torch.randn(batch_size, dim)
    direction = torch.randn(batch_size, dim)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    perturbed = torch.stack((anchor + direction, anchor + 2.0 * direction), dim=1)
    return anchor, direction, perturbed


def test_orbit_deflation_penalizes_an_adaptation_along_the_nuisance_orbit():
    anchor, direction, perturbed = _anchors()
    current = (anchor + 0.5 * direction).requires_grad_(True)
    loss_fn = AcquisitionOrbitDeflationLoss(relative_singular_value=0.01)

    loss, normal, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
    )

    assert loss.item() > 0.99
    assert normal.norm().item() < 1.0e-4
    assert metrics["orbit_nuisance_rank"].item() == 1.0
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_orbit_deflation_retains_a_normal_adaptation_component():
    anchor, direction, perturbed = _anchors(dim=9)
    candidate = torch.randn_like(direction)
    normal_direction = candidate - (candidate * direction).sum(dim=-1, keepdim=True) * direction
    normal_direction = normal_direction / normal_direction.norm(dim=-1, keepdim=True)
    current = anchor + 0.5 * normal_direction
    loss_fn = AcquisitionOrbitDeflationLoss(relative_singular_value=0.01)

    loss, normal, _ = loss_fn(
        current_features=current,
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
    )

    assert loss.item() < 1.0e-5
    assert torch.allclose(normal, 0.5 * normal_direction, atol=2.0e-5)


def test_orbit_deflation_detaches_anchor_and_perturbation_artifacts():
    anchor, direction, perturbed = _anchors()
    anchor.requires_grad_(True)
    perturbed.requires_grad_(True)
    current = (anchor.detach() + 0.5 * direction).requires_grad_(True)
    loss, _, _ = AcquisitionOrbitDeflationLoss(relative_singular_value=0.01)(
        current_features=current,
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
    )
    loss.backward()

    assert current.grad is not None
    assert anchor.grad is None
    assert perturbed.grad is None


def test_zero_orbit_has_no_arbitrary_projection_direction():
    torch.manual_seed(13)
    anchor = torch.randn(3, 5)
    current = anchor + torch.randn_like(anchor)
    repeated = anchor.unsqueeze(1).expand(-1, 2, -1).clone()
    loss, normal, metrics = AcquisitionOrbitDeflationLoss()(
        current_features=current,
        anchor_features=anchor,
        perturbed_anchor_features=repeated,
    )

    assert loss.item() == 0.0
    assert torch.allclose(normal, current - anchor)
    assert metrics["orbit_nuisance_rank"].item() == 0.0


def test_acquisition_orbit_views_keep_the_input_shape_and_change_each_orbit_member():
    torch.manual_seed(17)
    images = torch.randn(3, 2, 12, 10)
    views = _make_acquisition_orbit_views(
        images,
        contrast_scale=1.15,
        background_scale=0.10,
        blur_mix=0.50,
        num_perturbations=3,
    )

    assert views.shape == (3, 3, 2, 12, 10)
    assert not torch.allclose(views[:, 0], images)
    assert not torch.allclose(views[:, 1], images)
    assert not torch.allclose(views[:, 2], images)


def test_tangent_gradient_projection_is_forward_identity_and_exactly_orthogonal():
    anchor, _, perturbed = _anchors(batch_size=3, dim=11)
    basis, active, _ = build_acquisition_tangent_basis(
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
        relative_singular_value=0.01,
    )
    features = torch.randn_like(anchor, requires_grad=True)
    incoming_gradient = torch.randn_like(features)

    projected_features = apply_acquisition_tangent_gradient_projection(
        features,
        tangent_basis=basis,
        active_rows=active,
        strength=1.0,
    )
    assert torch.equal(projected_features, features)
    projected_features.backward(incoming_gradient)

    expected = incoming_gradient - project_onto_acquisition_tangent(
        incoming_gradient,
        tangent_basis=basis,
        active_rows=active,
    )
    assert torch.allclose(features.grad, expected, atol=2.0e-6)
    coefficients = torch.einsum("bd,bmd->bm", features.grad, basis) * active
    assert coefficients.abs().max().item() < 2.0e-6


def test_tangent_gradient_projection_strength_zero_is_an_identity_backward():
    anchor, _, perturbed = _anchors(batch_size=2, dim=8)
    basis, active, _ = build_acquisition_tangent_basis(
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
        relative_singular_value=0.01,
    )
    features = torch.randn_like(anchor, requires_grad=True)
    incoming_gradient = torch.randn_like(features)

    apply_acquisition_tangent_gradient_projection(
        features,
        tangent_basis=basis,
        active_rows=active,
        strength=0.0,
    ).backward(incoming_gradient)

    assert torch.equal(features.grad, incoming_gradient)


def test_rank_matched_random_tangent_preserves_active_rank_and_orthonormality():
    anchor, _, perturbed = _anchors(batch_size=3, dim=11)
    basis, active, _ = build_acquisition_tangent_basis(
        anchor_features=anchor,
        perturbed_anchor_features=perturbed,
        relative_singular_value=0.01,
    )
    random_basis = rank_matched_random_tangent_basis(basis)

    assert active.sum(dim=-1).tolist() == [1, 1, 1]
    gram = random_basis @ random_basis.transpose(-2, -1)
    identity = torch.eye(random_basis.shape[1]).expand_as(gram)
    assert torch.allclose(gram, identity, atol=2.0e-6)


def test_shift_tangent_correspondence_preserves_bank_but_breaks_pairs():
    basis = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    active = torch.tensor(
        [[True, False], [True, True], [False, True], [False, False]]
    )

    shifted_basis, shifted_active = shift_tangent_correspondence(
        basis,
        active,
        shift=1,
    )

    assert torch.equal(shifted_basis, torch.roll(basis, shifts=1, dims=0))
    assert torch.equal(shifted_active, torch.roll(active, shifts=1, dims=0))
    assert not torch.equal(shifted_basis, basis)


def test_shift_tangent_correspondence_rejects_single_sample_controls():
    basis = torch.randn(1, 2, 5)
    active = torch.ones(1, 2, dtype=torch.bool)

    try:
        shift_tangent_correspondence(basis, active)
    except ValueError as error:
        assert "At least two samples" in str(error)
    else:
        raise AssertionError("Expected a single-sample shuffled control to fail")
