import json
from pathlib import Path

import pytest

from scripts.summarize_local_dino_weight_dense_followup import _load_arm


def _segmentation_command(dataset: str, checkpoint_iter: int) -> list[str]:
    return [
        "python",
        "pipeline",
        "--datasets",
        dataset,
        "--checkpoint-iters",
        str(checkpoint_iter),
        "--protocol",
        "best",
        "--dataset-split-protocol",
        "formal-v1",
        "--feature-batch-size",
        "32",
        "--probe-batch-size",
        "32",
        "--probe-epochs",
        "50",
        "--probe-eval-every",
        "50",
        "--probe-seed",
        "0",
        "--channel-policy",
        "auto",
        "--channel-policy-seed",
        "0",
    ]


def _write_arm(root: Path, checkpoint_iter: int = 255) -> None:
    jobs = [
        {"task": "segmentation", "dataset": dataset, "cmd": _segmentation_command(dataset, checkpoint_iter)}
        for dataset in ("cellpose", "conic")
    ]
    jobs.append(
        {
            "task": "detection",
            "dataset": "bbbc038",
            "cmd": [
                "python",
                "center_probe",
                "--epochs",
                "5",
                "--batch-size",
                "8",
                "--channel-policy",
                "auto",
                "--conic-split-protocol",
                "official-baseline-fold0-nested-v1",
            ],
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "command_manifest.json").write_text(json.dumps(jobs), encoding="utf-8")

    paths = {
        "cellpose": root
        / "bio_segmentation"
        / "run__best__last1__pad__s512"
        / "cellpose"
        / str(checkpoint_iter),
        "conic": root
        / "bio_segmentation"
        / "run__best__custom_4_11_17_23__s256__cw_sqrt_inverse_spofficial_baseline_fold0_nested_v1"
        / "conic"
        / str(checkpoint_iter),
    }
    for index, path in enumerate(paths.values()):
        path.mkdir(parents=True)
        payload = {
            "_meta": {"seed": 0, "probe_rng_seeded": True, "probe_batch_size": 32, "probe_epochs": 50},
            "test": {"mDice": 0.7 + index / 10, "mIoU": 0.6 + index / 10},
        }
        (path / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    detection_path = root / "bio_detection" / "bbbc038" / str(checkpoint_iter)
    detection_path.mkdir(parents=True)
    detection = {
        "dataset": "bbbc038",
        "checkpoint": str(checkpoint_iter),
        "epochs": 5,
        "batch_size": 8,
        "seed": 0,
        "test_patch_f1": 71.5,
    }
    (detection_path / "results_bio_detection.json").write_text(json.dumps(detection), encoding="utf-8")


def test_load_arm_audits_protocol_and_reads_fixed_metrics(tmp_path: Path) -> None:
    _write_arm(tmp_path)

    result = _load_arm(tmp_path, 255)

    assert result["segmentation"]["cellpose"]["test_mDice"] == pytest.approx(0.7)
    assert result["segmentation"]["conic"]["test_mDice"] == pytest.approx(0.8)
    assert result["bbbc038_detection_observation"]["test_patch_f1"] == pytest.approx(71.5)
    assert result["manifest_audit"]["pass"] is True


def test_load_arm_rejects_wrong_probe_batch(tmp_path: Path) -> None:
    _write_arm(tmp_path)
    manifest_path = tmp_path / "command_manifest.json"
    jobs = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs[0]["cmd"][jobs[0]["cmd"].index("--probe-batch-size") + 1] = "16"
    manifest_path.write_text(json.dumps(jobs), encoding="utf-8")

    with pytest.raises(ValueError, match="probe-batch-size=16"):
        _load_arm(tmp_path, 255)
