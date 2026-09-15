#!/usr/bin/env python3
"""Apply the locked native 2-D CTC observation to one matched Gram branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import run_ctc_native_2d_observation as implementation


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "outputs/02_eval_runs/ctc_native_2d_hs6_l5_gram_branches_observation_20260912"
)
PLAN = ROOT / "Evaluation Rules/plans/hs6_l5_gram_new_v3_datasets_addendum_20260912.md"
RUNS: dict[str, dict[str, Any]] = {
    "control": {
        "run": "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
        "model": "hs6_l5_control_ck20495",
        "intervention": {"patch_gram": False, "global_relation_gram": False},
    },
    "anchor7807": {
        "run": "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
        "model": "hs6_l5_anchor7807_ck20495",
        "intervention": {
            "patch_gram": True,
            "patch_anchor_checkpoint": 7807,
            "patch_weight": 2.0,
            "global_relation_gram": False,
        },
    },
    "anchor17079": {
        "run": "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911",
        "model": "hs6_l5_anchor17079_ck20495",
        "intervention": {
            "patch_gram": True,
            "patch_anchor_checkpoint": 17079,
            "patch_weight": 2.0,
            "global_relation_gram": False,
        },
    },
    "dual": {
        "run": "HS6_L5_ck20007_dualgram_pa7807_ga20007_u488_gb64_4x3090qi_dual_v3_20260912",
        "model": "hs6_l5_dual_anchor_ck20495",
        "intervention": {
            "patch_gram": True,
            "patch_anchor_checkpoint": 7807,
            "patch_weight": 2.0,
            "global_relation_gram": True,
            "global_relation_anchor_checkpoint": 20007,
            "global_relation_weight": 0.5,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=sorted(RUNS), required=True)
    selected, remaining = parser.parse_known_args()
    specification = RUNS[selected.arm]

    implementation.TRAIN_RUN = ROOT / "outputs/01_training_runs" / specification["run"]
    implementation.CANDIDATE_STEPS = (20495,)
    implementation.PLAN = PLAN
    implementation.DEFAULT_CAMPAIGN = OUTPUT_ROOT / selected.arm

    original_prepare = implementation.prepare_campaign_manifest

    def prepare_branch_manifest(campaign: Path, cache_manifest_path: Path) -> Path:
        manifest_path = campaign / "campaign_manifest.json"
        existed = manifest_path.exists()
        result = original_prepare(campaign, cache_manifest_path)
        payload = json.loads(result.read_text())
        if existed:
            if payload.get("branch_arm") != selected.arm:
                raise RuntimeError(f"campaign is locked for another arm: {result}")
            return result

        if len(payload["models"]) != 1:
            raise RuntimeError("branch campaign must contain exactly one endpoint")
        payload["branch_arm"] = selected.arm
        payload["candidate_selection"] = (
            "matched Gram branch fixed before its CTC result; see the v3 dataset addendum"
        )
        payload["models"][0]["model"] = specification["model"]
        payload["models"][0]["arm"] = selected.arm
        payload["ssl_training"] = {
            "student_start_checkpoint": 20007,
            "endpoint_checkpoint": 20495,
            "matched_updates": 488,
            "effective_global_batch": 64,
            "intervention": specification["intervention"],
            "labels_used": False,
        }
        payload["code"].append(implementation.file_record(Path(__file__).resolve()))
        implementation.atomic_json(result, payload)
        return result

    implementation.prepare_campaign_manifest = prepare_branch_manifest
    sys.argv = [sys.argv[0], *remaining]
    implementation.main()


if __name__ == "__main__":
    main()
