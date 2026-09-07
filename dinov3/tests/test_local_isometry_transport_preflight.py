import numpy as np
import torch

from scripts.analyze_local_isometry_transport_preflight import (
    angle_match_control_to_true,
    build_same_axis_graph,
    geodesic_directions,
    geodesic_targets,
    local_kernel_distortion,
    retained_tangent_gain,
    smooth_tangent_fields,
)


def _synthetic_anchors() -> torch.Tensor:
    anchors = torch.tensor(
        [
            [1.0, -0.12, 0.0, 0.0],
            [1.0, -0.04, 0.0, 0.0],
            [1.0, 0.04, 0.0, 0.0],
            [1.0, 0.12, 0.0, 0.0],
        ]
    )
    return torch.nn.functional.normalize(anchors, dim=-1)


def test_geodesic_round_trip_preserves_per_sample_angle() -> None:
    anchors = _synthetic_anchors()
    directions = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    angles = torch.tensor([0.05, 0.10, 0.15, 0.20])
    targets = geodesic_targets(anchors, directions, angles)
    recovered, valid = geodesic_directions(anchors, targets, angles)
    rebuilt = geodesic_targets(anchors, recovered, angles)

    assert valid.all()
    assert torch.allclose(targets, rebuilt, atol=1.0e-6)
    assert torch.allclose((anchors * rebuilt).sum(dim=-1), torch.cos(angles), atol=1.0e-6)


def test_same_axis_graph_never_crosses_axis_labels() -> None:
    anchors = _synthetic_anchors()
    labels = np.asarray(["human", "human", "mouse", "mouse"])
    _, _, edges, report = build_same_axis_graph(anchors, labels, local_k=3, temperature=0.1, block_size=2)
    source, target = edges.cpu().numpy()

    assert np.all(labels[source] == labels[target])
    assert report["undirected_edges_after_symmetrization"] == 2


def test_control_output_angles_are_matched_per_sample() -> None:
    true = torch.tensor([[0.3, 0.0], [0.0, 0.1], [0.2, 0.2]])
    shuffled = torch.tensor([[0.0, 0.1], [0.4, 0.0], [-0.3, 0.1]])
    calibrated, operator_angles = angle_match_control_to_true(torch.stack((true, shuffled)))

    assert torch.allclose(operator_angles[1], shuffled.norm(dim=-1))
    assert torch.allclose(calibrated[0].norm(dim=-1), calibrated[1].norm(dim=-1))
    assert not torch.allclose(calibrated[1], true)


def test_smoothing_reduces_local_kernel_tearing_and_keeps_gain() -> None:
    anchors = _synthetic_anchors()
    labels = np.asarray(["human"] * len(anchors))
    adjacency, degree, edges, _ = build_same_axis_graph(anchors, labels, local_k=3, temperature=0.1, block_size=4)
    mostly_coherent = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.2],
        ]
    )
    mostly_coherent = torch.nn.functional.normalize(mostly_coherent, dim=-1)
    raw_directions = torch.stack((mostly_coherent, torch.roll(mostly_coherent, 1, 0)))
    confidence = torch.ones(len(anchors))
    angles = torch.full((len(anchors),), 0.25)
    raw_fields = raw_directions * angles[None, :, None]
    smoothed = smooth_tangent_fields(
        anchors,
        raw_fields,
        adjacency,
        degree,
        confidence,
        smoothing_lambda=0.5,
        iterations=80,
    )
    raw_targets = geodesic_targets(anchors, raw_fields[0], raw_fields[0].norm(dim=-1))
    smooth_targets = geodesic_targets(anchors, smoothed[0], smoothed[0].norm(dim=-1))
    raw_distortion = local_kernel_distortion(anchors, raw_targets, edges)
    smooth_distortion = local_kernel_distortion(anchors, smooth_targets, edges)
    gain = retained_tangent_gain(raw_fields[0], smoothed[0], confidence)

    assert torch.quantile(smooth_distortion, 0.9) < 0.5 * torch.quantile(raw_distortion, 0.9)
    assert gain > 0.5
    assert torch.allclose(
        (anchors * smooth_targets).sum(dim=-1),
        torch.cos(smoothed[0].norm(dim=-1)),
        atol=1.0e-6,
    )
