import torch
import torch.nn.functional as F

from dinov3.loss.intervention_factorized_topology_loss import (
    InterventionContextHead,
    InterventionFactorizedTopologyLoss,
    make_balanced_intervention_assignments,
    value_and_gradient_matched_shuffled_mse,
)


def test_balanced_intervention_assignments_are_deterministic_and_histogram_matched():
    true_ids, shuffled_ids = make_balanced_intervention_assignments(
        64,
        num_interventions=3,
        iteration=7,
        rank=2,
    )
    repeated_true, repeated_shuffled = make_balanced_intervention_assignments(
        64,
        num_interventions=3,
        iteration=7,
        rank=2,
    )

    assert torch.equal(true_ids, repeated_true)
    assert torch.equal(shuffled_ids, repeated_shuffled)
    assert torch.equal(torch.bincount(true_ids), torch.bincount(shuffled_ids))
    assert int(torch.bincount(true_ids).max() - torch.bincount(true_ids).min()) <= 1
    assert (true_ids == shuffled_ids).float().mean().item() < 0.5


def test_shuffled_mse_matches_true_value_and_prediction_gradient_norm():
    torch.manual_seed(5)
    prediction_true = torch.randn(11, 3, requires_grad=True)
    prediction_control = prediction_true.detach().clone().requires_grad_(True)
    ids = torch.arange(11) % 3
    shuffled_ids = ids[torch.tensor([4, 8, 1, 9, 0, 6, 10, 3, 7, 2, 5])]
    true_target = F.one_hot(ids, num_classes=3).float()
    shuffled_target = F.one_hot(shuffled_ids, num_classes=3).float()

    true_loss = F.mse_loss(prediction_true, true_target)
    control_loss, metrics = value_and_gradient_matched_shuffled_mse(
        prediction_control,
        true_target=true_target,
        shuffled_target=shuffled_target,
    )
    true_loss.backward()
    control_loss.backward()

    assert torch.allclose(control_loss, true_loss)
    assert torch.allclose(prediction_control.grad.norm(), prediction_true.grad.norm(), atol=1.0e-6)
    assert not torch.allclose(prediction_control.grad, prediction_true.grad)
    assert metrics["ift_context_gradient_scale"].item() > 0


def _loss_inputs(batch_size=6, dim=12, patches=4):
    torch.manual_seed(13)
    base = torch.randn(batch_size, dim, requires_grad=True)
    intervention = (base.detach() + 0.1 * torch.randn(batch_size, dim)).requires_grad_(True)
    anchor = base.detach().clone()
    base_patches = torch.randn(batch_size, patches, dim, requires_grad=True)
    anchor_patches = base_patches.detach().clone()
    ids, shuffled_ids = make_balanced_intervention_assignments(
        batch_size,
        num_interventions=3,
        iteration=2,
    )
    return base, intervention, anchor, base_patches, anchor_patches, ids, shuffled_ids


def test_factorized_topology_true_mode_backpropagates_without_anchor_gradients():
    inputs = _loss_inputs()
    head = InterventionContextHead(12, context_dim=5, hidden_dim=9, num_interventions=3)
    head.reset_parameters()
    loss_fn = InterventionFactorizedTopologyLoss(
        num_interventions=3,
        sample_topk=2,
        patch_radius=1,
    )

    loss, metrics = loss_fn(
        base_features=inputs[0],
        intervention_features=inputs[1],
        anchor_features=inputs[2],
        base_patches=inputs[3],
        anchor_patches=inputs[4],
        intervention_ids=inputs[5],
        shuffled_intervention_ids=inputs[6],
        context_head=head,
        mode="true",
        patch_masks=torch.zeros(6, 4, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() > 0
    assert inputs[0].grad is not None
    assert inputs[1].grad is not None
    assert inputs[3].grad is not None
    assert inputs[2].grad is None
    assert inputs[4].grad is None
    assert metrics["ift_sample_topology_loss"].item() < 1.0e-8
    assert metrics["ift_patch_topology_loss"].item() < 1.0e-8
    assert torch.isfinite(inputs[0].grad).all()


def test_factorized_topology_baseline_matches_compute_but_has_zero_auxiliary_gradient():
    inputs = _loss_inputs()
    head = InterventionContextHead(12, context_dim=5, hidden_dim=9, num_interventions=3)
    head.reset_parameters()
    loss_fn = InterventionFactorizedTopologyLoss(
        num_interventions=3,
        sample_topk=2,
        patch_radius=1,
    )

    loss, metrics = loss_fn(
        base_features=inputs[0],
        intervention_features=inputs[1],
        anchor_features=inputs[2],
        base_patches=inputs[3],
        anchor_patches=inputs[4],
        intervention_ids=inputs[5],
        shuffled_intervention_ids=inputs[6],
        context_head=head,
        mode="baseline",
    )
    loss.backward()

    assert loss.item() == 0.0
    assert metrics["ift_raw_total_loss"].item() > 0
    assert inputs[0].grad is not None and inputs[0].grad.abs().max().item() == 0.0
    assert inputs[1].grad is not None and inputs[1].grad.abs().max().item() == 0.0
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert all(parameter.grad.abs().max().item() == 0.0 for parameter in head.parameters())
