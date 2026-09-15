#!/usr/bin/env python3
"""Summarize and audit the preregistered local-DINO dense follow-up."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("w1", "w025")
SEGMENTATION = {
    "cellpose": {
        "path_tokens": ("__best__last1__pad__s512",),
        "layers": "last1",
        "resolution": 512,
        "resize": "pad",
        "class_weight": "none",
        "split": "legacy",
    },
    "conic": {
        "path_tokens": (
            "__best__custom_4_11_17_23__s256__cw_sqrt_inverse_",
            "spofficial_baseline_fold0_nested_v1",
        ),
        "layers": [4, 11, 17, 23],
        "resolution": 256,
        "resize": "stretch",
        "class_weight": "sqrt_inverse",
        "split": "official-baseline-fold0-nested-v1",
    },
}


def _one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {description}, found {len(paths)}: {paths}")
    return paths[0]


def _finite(value: Any, description: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {description}: {parsed}")
    return parsed


def _flag_value(cmd: list[str], flag: str) -> str:
    matches = [index for index, token in enumerate(cmd) if token == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(cmd):
        raise ValueError(f"Expected one value for {flag} in command: {cmd}")
    return cmd[matches[0] + 1]


def _audit_manifest(root: Path, checkpoint_iter: int) -> dict[str, Any]:
    path = root / "command_manifest.json"
    jobs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or len(jobs) != 3:
        raise ValueError(f"Expected three dense jobs in {path}")
    by_key = {(job["task"], job["dataset"]): job for job in jobs}
    expected = {("segmentation", "cellpose"), ("segmentation", "conic"), ("detection", "bbbc038")}
    if set(by_key) != expected:
        raise ValueError(f"Unexpected dense matrix in {path}: {sorted(by_key)}")

    for dataset in SEGMENTATION:
        cmd = list(by_key[("segmentation", dataset)]["cmd"])
        required = {
            "--checkpoint-iters": str(checkpoint_iter),
            "--protocol": "best",
            "--dataset-split-protocol": "formal-v1",
            "--feature-batch-size": "32",
            "--probe-batch-size": "32",
            "--probe-epochs": "50",
            "--probe-eval-every": "50",
            "--probe-seed": "0",
            "--channel-policy": "auto",
            "--channel-policy-seed": "0",
        }
        for flag, expected_value in required.items():
            actual = _flag_value(cmd, flag)
            if actual != expected_value:
                raise ValueError(f"{dataset}: {flag}={actual}, expected {expected_value}")

    detection_cmd = list(by_key[("detection", "bbbc038")]["cmd"])
    detection_required = {
        "--epochs": "5",
        "--batch-size": "8",
        "--channel-policy": "auto",
        "--conic-split-protocol": "official-baseline-fold0-nested-v1",
    }
    for flag, expected_value in detection_required.items():
        actual = _flag_value(detection_cmd, flag)
        if actual != expected_value:
            raise ValueError(f"bbbc038 detection: {flag}={actual}, expected {expected_value}")
    return {"path": str(path.resolve()), "jobs": len(jobs), "pass": True}


def _load_arm(root: Path, checkpoint_iter: int) -> dict[str, Any]:
    manifest = _audit_manifest(root, checkpoint_iter)
    segmentation: dict[str, Any] = {}
    for dataset, protocol in SEGMENTATION.items():
        candidates = [
            path
            for path in (root / "bio_segmentation").rglob("results.json")
            if path.parent.name == str(checkpoint_iter)
            and path.parent.parent.name == dataset
            and all(token in str(path) for token in protocol["path_tokens"])
        ]
        path = _one(candidates, f"{dataset} segmentation result")
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("_meta", {})
        if meta.get("probe_seeded_rng", meta.get("probe_rng_seeded")) is not True:
            raise ValueError(f"Probe RNG was not recorded as seeded at {path}")
        if int(meta.get("seed", -1)) != 0 or int(meta.get("probe_batch_size", -1)) != 32:
            raise ValueError(f"Unexpected seed or probe batch at {path}")
        if int(meta.get("probe_epochs", -1)) != 50:
            raise ValueError(f"Unexpected probe epochs at {path}")
        segmentation[dataset] = {
            "test_mDice": _finite(payload["test"]["mDice"], f"{dataset} test mDice"),
            "test_mIoU": _finite(payload["test"]["mIoU"], f"{dataset} test mIoU"),
            "protocol": protocol,
            "result_path": str(path.resolve()),
        }

    detection_path = root / "bio_detection" / "bbbc038" / str(checkpoint_iter) / "results_bio_detection.json"
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    required_detection = {
        "dataset": "bbbc038",
        "checkpoint": str(checkpoint_iter),
        "epochs": 5,
        "batch_size": 8,
        "seed": 0,
    }
    for key, expected in required_detection.items():
        if detection.get(key) != expected:
            raise ValueError(f"Detection {key}={detection.get(key)!r}, expected {expected!r}")
    return {
        "segmentation": segmentation,
        "bbbc038_detection_observation": {
            "test_patch_f1": _finite(detection["test_patch_f1"], "BBBC038 test patch F1"),
            "result_path": str(detection_path.resolve()),
        },
        "manifest_audit": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iter", type=int, default=255)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = {arm: _load_arm(args.eval_root / arm, args.checkpoint_iter) for arm in ARMS}
    segmentation_deltas = {
        dataset: arms["w025"]["segmentation"][dataset]["test_mDice"]
        - arms["w1"]["segmentation"][dataset]["test_mDice"]
        for dataset in SEGMENTATION
    }
    observation_delta = (
        arms["w025"]["bbbc038_detection_observation"]["test_patch_f1"]
        - arms["w1"]["bbbc038_detection_observation"]["test_patch_f1"]
    )
    report = {
        "summary": "local_dino_weight_dense_followup_v1",
        "checkpoint_iter": args.checkpoint_iter,
        "primary_formal_metric": "test_mDice",
        "formal_datasets": list(SEGMENTATION),
        "observation_only": "bbbc038_detection/test_patch_f1",
        "arms": arms,
        "w025_minus_w1": {
            "segmentation_mDice_deltas": segmentation_deltas,
            "segmentation_mean_delta": statistics.fmean(segmentation_deltas.values()),
            "segmentation_win_count": sum(value > 0 for value in segmentation_deltas.values()),
            "bbbc038_detection_observation_delta": observation_delta,
        },
        "note": "No dense pass threshold was preregistered; these deltas are descriptive and BBBC038 is excluded from the formal mean.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["w025_minus_w1"]}, indent=2))


if __name__ == "__main__":
    main()
