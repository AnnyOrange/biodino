import pytest
import torch

from dinov3.loss import ExpertConsensusResidualLoss


def _features_for_pairs(pairs: list[tuple[int, int]], size: int = 6) -> torch.Tensor:
    features = torch.eye(size)
    for left, right in pairs:
        features[right] = features[left]
    return features


def test_no_correction_when_experts_match_anchor():
    anchor = torch.randn(8, 12)
    student = anchor.clone().requires_grad_(True)
    loss_fn = ExpertConsensusResidualLoss(neighborhood_topk=2, min_experts=2)
    loss, metrics = loss_fn(
        student_features=student,
        anchor_features=anchor,
        expert_features=[anchor.clone(), anchor.clone()],
    )
    assert loss.item() == 0.0
    assert metrics["ecr_selected_edges"].item() == 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_consensus_missed_edge_produces_gradient():
    anchor = torch.eye(6)
    experts = _features_for_pairs([(0, 1), (2, 3)])
    student = anchor.clone().requires_grad_(True)
    loss_fn = ExpertConsensusResidualLoss(
        neighborhood_topk=1,
        min_experts=2,
        temperature=0.2,
    )
    loss, metrics = loss_fn(
        student_features=student,
        anchor_features=anchor,
        expert_features=[experts, experts.clone()],
    )
    assert loss.item() > 0
    assert metrics["ecr_selected_edges"].item() >= 4
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad.norm().item() > 0


def test_single_expert_edge_is_rejected_by_consensus():
    anchor = torch.eye(6)
    first = _features_for_pairs([(0, 1)])
    second = _features_for_pairs([(2, 3)])
    loss_fn = ExpertConsensusResidualLoss(neighborhood_topk=1, min_experts=2)
    loss, metrics = loss_fn(
        student_features=anchor.clone().requires_grad_(True),
        anchor_features=anchor,
        expert_features=[first, second],
    )
    assert loss.item() == 0.0
    assert metrics["ecr_consensus_edges"].item() == 0.0


def test_routing_requires_both_endpoints_to_be_active():
    anchor = torch.eye(6)
    experts = _features_for_pairs([(0, 1), (2, 3)])
    weights = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss_fn = ExpertConsensusResidualLoss(neighborhood_topk=1, min_experts=2)
    _loss, metrics = loss_fn(
        student_features=anchor.clone().requires_grad_(True),
        anchor_features=anchor,
        expert_features=[experts, experts.clone()],
        expert_weights=weights,
    )
    assert metrics["ecr_selected_edges"].item() == 2.0


def test_shuffled_control_changes_the_target_graph():
    torch.manual_seed(4)
    anchor = torch.randn(12, 16)
    experts = [torch.randn(12, 10), torch.randn(12, 14)]
    kwargs = dict(
        student_features=anchor.clone().requires_grad_(True),
        anchor_features=anchor,
        expert_features=experts,
    )
    true_loss, true_metrics = ExpertConsensusResidualLoss(
        neighborhood_topk=4, min_experts=2
    )(**kwargs)
    shuffled_loss, shuffled_metrics = ExpertConsensusResidualLoss(
        neighborhood_topk=4, min_experts=2, shuffled_control=True
    )(**kwargs)
    assert shuffled_metrics["ecr_shuffled_control"].item() == 1.0
    assert (
        true_metrics["ecr_selected_edges"].item()
        != shuffled_metrics["ecr_selected_edges"].item()
        or not torch.allclose(true_loss, shuffled_loss)
    )


def test_invalid_expert_weight_shape_is_rejected():
    anchor = torch.randn(5, 4)
    loss_fn = ExpertConsensusResidualLoss(min_experts=1)
    with pytest.raises(ValueError, match="expert_weights"):
        loss_fn(
            student_features=anchor,
            anchor_features=anchor,
            expert_features=[anchor],
            expert_weights=torch.ones(2, 5),
        )


def test_edge_mask_restricts_consensus_to_cross_domain_candidates():
    anchor = torch.eye(6)
    experts = _features_for_pairs([(0, 1), (2, 3)])
    edge_mask = torch.zeros(6, 6, dtype=torch.bool)
    edge_mask[2, 3] = edge_mask[3, 2] = True
    loss_fn = ExpertConsensusResidualLoss(neighborhood_topk=1, min_experts=2)
    _loss, metrics = loss_fn(
        student_features=anchor.clone().requires_grad_(True),
        anchor_features=anchor,
        expert_features=[experts, experts.clone()],
        edge_mask=edge_mask,
    )
    assert metrics["ecr_candidate_edges"].item() == 2.0
    assert metrics["ecr_selected_edges"].item() == 2.0


def test_asymmetric_edge_mask_is_rejected():
    features = torch.randn(4, 5)
    edge_mask = torch.zeros(4, 4, dtype=torch.bool)
    edge_mask[0, 1] = True
    with pytest.raises(ValueError, match="symmetric"):
        ExpertConsensusResidualLoss(min_experts=1)(
            student_features=features,
            anchor_features=features,
            expert_features=[features],
            edge_mask=edge_mask,
        )
