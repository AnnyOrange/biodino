import torch

from dinov3.loss.conditional_morphology_graph_loss import (
    ConditionalEdgeGraphPredictor,
    ConditionalMorphologyGraphLoss,
    ConditionalMorphologyGraphWeights,
)
from dinov3.loss.nested_channel_innovation_loss import ConditionalFeaturePredictor


def _predictor(dim: int) -> ConditionalFeaturePredictor:
    predictor = ConditionalFeaturePredictor(dim=dim, hidden_dim=2 * dim)
    predictor.reset_parameters()
    return predictor


def _edge_predictor(dim: int) -> ConditionalEdgeGraphPredictor:
    predictor = ConditionalEdgeGraphPredictor(dim=dim, edge_dim=4, hidden_dim=8)
    predictor.reset_parameters()
    return predictor


def test_direct_edge_predictor_starts_at_the_raw_subset_graph():
    torch.manual_seed(17)
    subset = torch.randn(2, 3, 4, 6)
    loss_fn = ConditionalMorphologyGraphLoss(local_radius=1, predictor_mode="edge")
    source, target = loss_fn._local_edges(4, torch.device("cpu"))
    subset_graph = loss_fn._edge_cosine(subset, source, target)

    predicted_graph = _edge_predictor(6)(subset, source, target, subset_graph)

    torch.testing.assert_close(predicted_graph, subset_graph)


def test_cmgi_preserves_the_subset_gradient_firewall_and_updates_full_graph():
    torch.manual_seed(0)
    full = torch.randn(2, 3, 9, 8, requires_grad=True)
    teacher = full.detach() + 0.2 * torch.randn_like(full)
    subset = torch.randn(2, 3, 9, 8, requires_grad=True)
    masks = torch.zeros(2, 3, 9, dtype=torch.bool)
    loss_fn = ConditionalMorphologyGraphLoss(
        local_radius=1,
        weights=ConditionalMorphologyGraphWeights(predictor=1.0, graph=1.0),
    )
    predictor = _predictor(8)

    loss, metrics = loss_fn(
        full_features=full,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.ones(3, dtype=torch.bool),
        predictor=predictor,
        masks=masks,
    )
    loss.backward()

    assert full.grad is not None and full.grad.abs().sum() > 0
    assert subset.grad is None
    assert predictor.mlp[-1].weight.grad is not None
    assert predictor.mlp[-1].weight.grad.abs().sum() > 0
    assert metrics["cmgi_graph_loss"] > 0
    assert metrics["cmgi_edge_gate_fraction"] > 0


def test_cmgi_direct_edge_mode_preserves_the_gradient_firewall():
    torch.manual_seed(11)
    full = torch.randn(2, 3, 9, 8, requires_grad=True)
    teacher = full.detach() + 0.2 * torch.randn_like(full)
    subset = torch.randn(2, 3, 9, 8, requires_grad=True)
    masks = torch.zeros(2, 3, 9, dtype=torch.bool)
    predictor = _edge_predictor(8)
    loss_fn = ConditionalMorphologyGraphLoss(
        local_radius=1,
        predictor_mode="edge",
        weights=ConditionalMorphologyGraphWeights(predictor=1.0, graph=1.0),
    )

    loss, metrics = loss_fn(
        full_features=full,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.ones(3, dtype=torch.bool),
        predictor=predictor,
        masks=masks,
    )
    loss.backward()

    assert full.grad is not None and full.grad.abs().sum() > 0
    assert subset.grad is None
    assert predictor.mlp[-1].weight.grad is not None
    assert predictor.mlp[-1].weight.grad.abs().sum() > 0
    assert metrics["cmgi_predictor_mode_edge"].item() == 1.0
    # A zero-gated direct predictor must not initially degrade the raw S graph.
    assert torch.isclose(metrics["cmgi_predictor_vs_subset_edge_delta"], torch.tensor(0.0))


def test_cmgi_ignores_inactive_samples():
    full = torch.randn(2, 2, 4, 6, requires_grad=True)
    teacher = torch.randn(2, 2, 4, 6)
    subset = torch.randn(2, 2, 4, 6, requires_grad=True)
    loss_fn = ConditionalMorphologyGraphLoss(local_radius=1)

    loss, metrics = loss_fn(
        full_features=full,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.zeros(2, dtype=torch.bool),
        predictor=_predictor(6),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert full.grad is not None and full.grad.abs().sum() == 0
    assert subset.grad is None
    assert metrics["cmgi_active_fraction"].item() == 0.0


def test_cmgi_uses_only_local_non_diagonal_edges():
    loss_fn = ConditionalMorphologyGraphLoss(local_radius=1)
    source, target = loss_fn._local_edges(9, torch.device("cpu"))

    assert source.numel() == 40  # 3x3 grid, directed 8-neighbour graph.
    assert torch.all(source != target)
    assert torch.all((source // 3 - target // 3).abs() <= 1)
    assert torch.all((source % 3 - target % 3).abs() <= 1)


def test_cmgi_uniform_control_keeps_every_visible_edge():
    torch.manual_seed(1)
    features = torch.randn(1, 2, 4, 6, requires_grad=True)
    teacher = features.detach() + 0.1 * torch.randn_like(features)
    subset = torch.randn(1, 2, 4, 6)
    loss_fn = ConditionalMorphologyGraphLoss(local_radius=1, gate_mode="uniform")

    loss, metrics = loss_fn(
        full_features=features,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.ones(2, dtype=torch.bool),
        predictor=_predictor(6),
    )
    loss.backward()

    assert features.grad is not None and features.grad.abs().sum() > 0
    assert torch.isclose(metrics["cmgi_edge_gate_fraction"], torch.tensor(1.0))


def test_cmgi_sparse_gate_matches_the_requested_edge_density():
    torch.manual_seed(2)
    features = torch.randn(1, 2, 4, 6, requires_grad=True)
    teacher = features.detach() + 0.1 * torch.randn_like(features)
    subset = torch.randn(1, 2, 4, 6)
    loss_fn = ConditionalMorphologyGraphLoss(
        local_radius=1,
        gate_mode="conditional",
        selection_fraction=0.25,
    )

    loss, metrics = loss_fn(
        full_features=features,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.ones(2, dtype=torch.bool),
        predictor=_predictor(6),
    )
    loss.backward()

    # A 2x2 grid has 12 directed local edges; ceil(0.25 * 12) = 3 per view.
    assert torch.isclose(metrics["cmgi_edge_gate_fraction"], torch.tensor(0.25))
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_cmgi_uniform_control_can_match_sparse_gate_density_without_rng():
    torch.manual_seed(3)
    features = torch.randn(1, 2, 4, 6, requires_grad=True)
    teacher = features.detach() + 0.1 * torch.randn_like(features)
    subset = torch.randn(1, 2, 4, 6)
    loss_fn = ConditionalMorphologyGraphLoss(
        local_radius=1,
        gate_mode="uniform",
        selection_fraction=0.25,
    )

    loss, metrics = loss_fn(
        full_features=features,
        teacher_features=teacher,
        subset_features=subset,
        active_samples=torch.ones(2, dtype=torch.bool),
        predictor=_predictor(6),
    )
    loss.backward()

    assert torch.isclose(metrics["cmgi_edge_gate_fraction"], torch.tensor(0.25))
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_cmgi_rejects_mismatched_mask_shape():
    loss_fn = ConditionalMorphologyGraphLoss(local_radius=1)
    features = torch.randn(2, 2, 4, 6)
    try:
        loss_fn(
            full_features=features,
            teacher_features=features,
            subset_features=features,
            active_samples=torch.ones(2, dtype=torch.bool),
            predictor=_predictor(6),
            masks=torch.zeros(2, 2, 3, dtype=torch.bool),
        )
    except ValueError as exc:
        assert "masks must have" in str(exc)
    else:
        raise AssertionError("Expected mask-shape validation to fail")
