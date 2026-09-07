from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.analyze_global_bridge_anchor_residual import (
    geodesic_targets,
    normalize,
    shuffled_geodesic_targets,
)


def test_readout_protocol_is_propagated_to_target_bank(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.npz"
    anchor_path = tmp_path / "anchor.npz"
    target_path = tmp_path / "target.npz"
    report_path = tmp_path / "report.json"
    keys = np.asarray(["a", "b"])
    np.savez(
        graph_path,
        keys=keys,
        offsets=np.asarray([0, 1, 2]),
        neighbor_indices=np.asarray([1, 0]),
        confidence=np.asarray([0.9, 0.9]),
    )
    np.savez(
        anchor_path,
        keys=keys,
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        feature_protocol=np.asarray("nlb2_avg"),
        normalization_protocol=np.asarray("eval_imagenet"),
        transform_resize_crop=np.asarray([256, 224]),
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_global_bridge_anchor_residual.py",
            "--graph",
            str(graph_path),
            "--anchor-bank",
            str(anchor_path),
            "--margin",
            "0.0",
            "--output-targets",
            str(target_path),
            "--output",
            str(report_path),
        ],
        check=True,
    )
    with np.load(target_path, allow_pickle=False) as payload:
        assert str(payload["feature_protocol"].item()) == "nlb2_avg"
        assert str(payload["input_normalization"].item()) == "eval_imagenet"
        assert int(payload["observation_crop_size"].item()) == 224


def test_geodesic_targets_move_a_fraction_toward_prototype() -> None:
    anchors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prototypes = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    targets, tangent, angle = geodesic_targets(anchors, prototypes, strength=0.25)
    assert np.allclose(np.linalg.norm(targets, axis=1), 1.0)
    assert np.allclose(angle, np.pi / 8)
    assert np.allclose(np.sum(targets * anchors, axis=1), np.cos(np.pi / 8))
    assert np.allclose(np.sum(tangent * anchors, axis=1), 0.0)


def test_shuffled_control_preserves_each_sample_angle() -> None:
    anchors = normalize(
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )
    prototypes = normalize(
        np.asarray(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    targets, tangent, angle = geodesic_targets(anchors, prototypes, strength=0.3)
    control, permutation = shuffled_geodesic_targets(anchors, tangent, angle)
    true_cosine = np.sum(targets * anchors, axis=1)
    control_cosine = np.sum(control * anchors, axis=1)
    assert permutation.tolist() == [2, 0, 1]
    assert np.allclose(true_cosine, control_cosine, atol=1e-6)
    assert not np.allclose(targets, control)
