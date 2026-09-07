import torch

from dinov3.loss.scout_kernel_delta_loss import (
    ScoutKernelDeltaTransportLoss,
    centered_cosine_kernel,
    cross_view_stable_kernel_delta,
)


def test_centered_cosine_kernel_is_invariant_to_orthogonal_feature_coordinates():
    torch.manual_seed(0)
    features = torch.randn(9, 7)
    rotation, _ = torch.linalg.qr(torch.randn(7, 7))

    torch.testing.assert_close(
        centered_cosine_kernel(features),
        centered_cosine_kernel(features @ rotation),
        atol=1e-6,
        rtol=1e-6,
    )


def test_centered_cosine_kernel_supports_batched_spatial_tokens():
    torch.manual_seed(10)
    features = torch.randn(4, 9, 6)
    kernels = centered_cosine_kernel(features)

    assert kernels.shape == (4, 9, 9)
    torch.testing.assert_close(kernels.sum(dim=-1), torch.zeros(4, 9), atol=1e-5, rtol=0)
    torch.testing.assert_close(kernels.sum(dim=-2), torch.zeros(4, 9), atol=1e-5, rtol=0)


def test_kernel_delta_transport_has_zero_loss_for_the_matched_direction():
    torch.manual_seed(1)
    anchor = torch.randn(8, 6)
    current = (anchor + 0.2 * torch.randn_like(anchor)).requires_grad_(True)
    scout_delta = centered_cosine_kernel(current.detach()) - centered_cosine_kernel(anchor)
    loss_fn = ScoutKernelDeltaTransportLoss()

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    loss.backward()

    assert loss.item() < 1e-5
    assert metrics["skdt_alignment"].item() > 0.99999
    assert current.grad is not None


def test_kernel_delta_transport_is_width_agnostic_for_matched_sample_geometry():
    torch.manual_seed(2)
    anchor = torch.randn(7, 5)
    current = anchor + 0.3 * torch.randn_like(anchor)
    scout_delta = centered_cosine_kernel(current) - centered_cosine_kernel(anchor)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5))
    loss_fn = ScoutKernelDeltaTransportLoss()

    original_loss, _ = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    rotated_loss, _ = loss_fn(
        current_features=current @ rotation,
        anchor_features=anchor @ rotation,
        scout_delta=scout_delta,
    )

    torch.testing.assert_close(original_loss, rotated_loss, atol=1e-6, rtol=1e-6)


def test_kernel_delta_transport_matches_batched_spatial_displacements():
    torch.manual_seed(11)
    anchor = torch.randn(3, 12, 7)
    current = (anchor + 0.2 * torch.randn_like(anchor)).requires_grad_(True)
    scout_delta = centered_cosine_kernel(current.detach()) - centered_cosine_kernel(anchor)
    loss_fn = ScoutKernelDeltaTransportLoss(metric_prefix="spatial")

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    loss.backward()

    assert loss.item() < 1e-5
    assert metrics["spatial_alignment"].item() > 0.99999
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_absolute_final_kernel_control_matches_the_current_relation_not_its_delta():
    torch.manual_seed(13)
    anchor = torch.randn(8, 6)
    current = (anchor + 0.2 * torch.randn_like(anchor)).requires_grad_(True)
    final_kernel = centered_cosine_kernel(current.detach())
    loss_fn = ScoutKernelDeltaTransportLoss()

    absolute_loss, _ = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=final_kernel,
        absolute_target=True,
    )
    legacy_delta_loss, _ = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=final_kernel,
    )

    assert absolute_loss.item() < 1e-5
    assert legacy_delta_loss.item() > 1e-3


def test_neighborhood_transport_uses_sparse_signed_extreme_edges():
    torch.manual_seed(12)
    anchor = torch.randn(10, 8)
    current = (anchor + 0.25 * torch.randn_like(anchor)).requires_grad_(True)
    scout_delta = centered_cosine_kernel(current.detach()) - centered_cosine_kernel(anchor)
    loss_fn = ScoutKernelDeltaTransportLoss(neighborhood_topk=2)

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    loss.backward()

    assert loss.item() < 1e-5
    assert 0 < metrics["skdt_selected_edges"].item() <= 40
    assert 0 < metrics["skdt_selected_fraction"].item() < 1
    assert 0 < metrics["skdt_target_energy_ratio"].item() <= 1.000001
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_kernel_delta_transport_has_a_finite_initial_direction_gradient():
    torch.manual_seed(4)
    anchor = torch.randn(7, 5)
    # The current model starts exactly at the large-model anchor, while a
    # nonzero cached scout delta gives it a direction in function space.
    current = anchor.detach().clone().requires_grad_(True)
    scout_current = anchor + 0.25 * torch.randn_like(anchor)
    scout_delta = centered_cosine_kernel(scout_current) - centered_cosine_kernel(anchor)
    loss_fn = ScoutKernelDeltaTransportLoss()

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["skdt_alignment"].item() == 0.0
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum() > 0


def test_directional_damping_does_not_disable_a_small_but_valid_target():
    torch.manual_seed(7)
    anchor = torch.randn(6, 10)
    current = anchor.detach().clone().requires_grad_(True)
    scout_delta = torch.randn(6, 6) * 0.01
    loss_fn = ScoutKernelDeltaTransportLoss(eps=1e-4, directional_damping=0.1)

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    loss.backward()

    assert metrics["skdt_active"].item() == 1.0
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum() > 0


def test_displacement_budget_leaves_local_transport_unchanged_and_decays_outside():
    torch.manual_seed(9)
    anchor = torch.randn(7, 6)
    current = anchor + 0.5 * torch.randn_like(anchor)
    scout_current = anchor + 0.3 * torch.randn_like(anchor)
    scout_delta = centered_cosine_kernel(scout_current) - centered_cosine_kernel(anchor)

    unbudgeted = ScoutKernelDeltaTransportLoss(directional_damping=0.1)
    local_budget = ScoutKernelDeltaTransportLoss(
        directional_damping=0.1,
        displacement_budget_ratio=1.0e6,
    )
    tiny_budget = ScoutKernelDeltaTransportLoss(
        directional_damping=0.1,
        displacement_budget_ratio=1.0e-6,
    )
    base_loss, _ = unbudgeted(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    local_loss, local_metrics = local_budget(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )
    clipped_loss, clipped_metrics = tiny_budget(
        current_features=current,
        anchor_features=anchor,
        scout_delta=scout_delta,
    )

    torch.testing.assert_close(local_loss, base_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_metrics["skdt_budget_gate"], torch.ones(()))
    assert clipped_metrics["skdt_budget_gate"].item() < 1.0e-3
    assert clipped_loss.item() < base_loss.item()


def test_zero_scout_delta_disables_transport_and_detaches_target():
    torch.manual_seed(3)
    anchor = torch.randn(6, 4)
    current = (anchor + 0.1 * torch.randn_like(anchor)).requires_grad_(True)
    zero_scout_delta = torch.zeros(6, 6, requires_grad=True)
    loss_fn = ScoutKernelDeltaTransportLoss()

    loss, metrics = loss_fn(
        current_features=current,
        anchor_features=anchor,
        scout_delta=zero_scout_delta,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert current.grad is not None and current.grad.abs().sum() == 0
    assert zero_scout_delta.grad is None
    assert metrics["skdt_active"].item() == 0.0


def test_cross_view_stable_delta_preserves_a_shared_relation_change():
    torch.manual_seed(5)
    delta = centered_cosine_kernel(torch.randn(9, 6))
    stable, metrics = cross_view_stable_kernel_delta(
        delta,
        delta,
        relative_eigenvalue=0.0,
        min_eigenvalue=0.0,
    )

    torch.testing.assert_close(stable, delta, atol=2.0e-5, rtol=2.0e-5)
    assert metrics["skdt_cross_view_alignment"].item() > 0.99999
    assert metrics["skdt_stable_rank"].item() > 0
    assert metrics["skdt_stable_active"].item() == 1.0


def test_cross_view_stable_delta_rejects_an_anti_aligned_view_change():
    torch.manual_seed(6)
    delta = centered_cosine_kernel(torch.randn(8, 5))
    stable, metrics = cross_view_stable_kernel_delta(delta, -delta, relative_eigenvalue=0.01)

    assert stable.abs().max().item() < 1.0e-6
    assert metrics["skdt_cross_view_alignment"].item() < -0.99999
    assert metrics["skdt_stable_rank"].item() == 0
    assert metrics["skdt_stable_active"].item() == 0.0


def test_stable_target_shuffle_preserves_strength_but_breaks_sample_alignment():
    torch.manual_seed(8)
    first = centered_cosine_kernel(torch.randn(8, 6))
    second = first + 0.02 * centered_cosine_kernel(torch.randn(8, 6))
    stable, _ = cross_view_stable_kernel_delta(first, second)
    permutation = torch.tensor([3, 0, 6, 1, 7, 4, 2, 5])
    shuffled = stable[permutation][:, permutation]

    # A simultaneous row/column permutation is an isometry of the target:
    # it leaves energy and spectrum intact but no longer assigns a relation to
    # the original sample pair. That makes it a strength-matched causal control.
    torch.testing.assert_close(shuffled.norm(), stable.norm(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.linalg.eigvalsh(shuffled), torch.linalg.eigvalsh(stable), atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(shuffled, stable)
